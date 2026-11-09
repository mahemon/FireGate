#!/usr/bin/env python
"""Review add-ons on stored features and outputs (CPU only: numpy + scipy; no GPU, no new runs).

    python scripts/review_analyses.py --config configs/v2/overlap_q15.yaml --name Nq [--out paper/v2_results/<exp>/switch]
        [--parts zoo,middle,redraws,regioncov,ci]

Needs <runs>/analysis/gate_features.npz and the switch report of the experiment (switch_summary.json in --out).
Writes into --out/review/ (tables *.tex, review.json) and review_numbers.tex (macros \\rv<name>...):
  zoo        (single task) other gates built from the same unlabeled prompts and trunk features, each at its own natural
             threshold (uncalibrated, as a router would use it) and PAC-calibrated at eps 0.03 / 0.1: the deployed probe
             (natural threshold: logit > 0, the router), a Gaussian classifier with a shared covariance (LDA), an
             Expert-Gate-style gate (one linear autoencoder = PCA per side, route to the side that reconstructs the
             prompt better), k nearest neighbours, and a one-class gate from the new task's prompts only (the region).
             For each: AUROC, firing on every pool, and every ungated model behind it. Separates "gating helps" from
             "calibration helps": the natural thresholds give whatever drift they give; calibration bounds it.
  middle     (single task) the middle of the separability scale, simulated: the trunk features projected on k random
             directions (k = 1 ... 128, three draws) make the probe weaker (AUROC from about 0.6 up); for each, the
             calibrated gate at eps 0.03 / 0.1, its power, and Seq-GRPO behind it, measured against beta * G.
  redraws    (single task) certificate variability: the reference fitting / calibration split drawn again and the
             new-task fitting prompts resampled, the probe refitted and recalibrated (R times): held-out Dolly firing,
             power and the switched gain over draws (the paper's gate is one draw, fixed over seeds).
  regioncov  (every stage) held-out coverage of the two-sided tripwire's region against the number n of unlabeled
             new-task prompts its threshold is set on (random subsets, and skewed subsets: one half of a random
             projection), against the nominal 1 - alpha.
  ci         paired bootstrap over evaluation prompts and seeds: 95% intervals of every method's gain and of the
             paired differences that the text compares (Tripwire-prompt - Seq-GRPO, router - Tripwire-prompt, Seq-GRPO
             behind the switch - Tripwire-prompt, ...), with the share of draws at or below 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402
from switch_analyses import f2, finished_runs, pv  # noqa: E402
from switch_report import GATED, SUBSTANTIVE, Experiment, Support, build_variant, fit_probe, readable_label  # noqa: E402
from separability import auroc  # noqa: E402

POOLS_SHOWN = ("dolly", "advbench", "xstest", "mmlu")
FAMN = {"seq": "Seq", "tripwire_prompt": "Twp", "router": "Router", "refkl": "Refkl", "kl": "Kl", "nullspace": "Null",
        "tripwire": "Tw", "tripwire_prompt_seq": "Tws", "seq@sw": "Seqsw", "ewc": "Ewc"}


# ----------------------------------------------------------------------------- per-prompt arrays of the runs
def run_arrays(runs_list, tasks, lens):
    """{family: {"method", "seeds": {seed: {"d": {task: delta correct}, "base": {task: base correct}, "chg": Dolly
    changed, "sub": Dolly substantive (None unless fully judged), "mmlu": delta correct}}}}"""
    out = {}
    for sd, fam, method, B, R, labels in runs_list:
        if any(t not in R or t not in B or R[t] is None or len(R[t]) != lens.get(t, -1) for t in tasks):
            continue
        e = {"d": {t: np.array([float(r["m"]["correct"]) - float(b["m"]["correct"]) for r, b in zip(R[t], B[t])])
                   for t in tasks},
             "base": {t: np.array([float(b["m"]["correct"]) for b in B[t]]) for t in tasks},
             "chg": None, "sub": None, "mmlu": None}
        if R.get("dolly") and B.get("dolly") and len(R["dolly"]) == lens.get("dolly", -1):
            e["chg"] = np.array([r["out"] != b["out"] for r, b in zip(R["dolly"], B["dolly"])])
            lab = (labels or {}).get("dolly")
            if lab is not None:
                ls = [lab.get(str(r["i"])) for r, c in zip(R["dolly"], e["chg"]) if c]
                if all(x is not None for x in ls):
                    e["sub"] = np.array([(lab.get(str(r["i"])) in SUBSTANTIVE) if c else False
                                         for r, c in zip(R["dolly"], e["chg"])])
        if R.get("mmlu") and B.get("mmlu") and len(R["mmlu"]) == lens.get("mmlu", -1):
            e["mmlu"] = np.array([float(r["m"]["correct"]) - float(b["m"]["correct"]) for r, b in zip(R["mmlu"], B["mmlu"])])
        out.setdefault(fam, {"method": method, "seeds": {}})["seeds"][sd] = e
    return out


def switched(fa, fire, tasks):
    """an ungated family behind a gate with firing vectors fire[pool]: gain, Dolly changed / substantive, MMLU"""
    gs, cs, ss, ms = [], [], [], []
    for e in fa["seeds"].values():
        gs.append(np.mean([(e["d"][t] * fire[t]).mean() for t in tasks]))
        if e["chg"] is not None and "dolly" in fire:
            cs.append((e["chg"] & fire["dolly"]).mean())
        if e["sub"] is not None and "dolly" in fire:
            ss.append((e["sub"] & fire["dolly"]).mean())
        if e["mmlu"] is not None and "mmlu" in fire:
            ms.append((e["mmlu"] * fire["mmlu"]).mean())
    n = len(fa["seeds"])
    return {"gain": float(np.mean(gs)), "gain_sd": float(np.std(gs, ddof=1)) if n > 1 else 0.0, "n": n,
            "chg": float(np.mean(cs)) if cs else None, "subst": float(np.mean(ss)) if len(ss) == n else None,
            "mmlu": float(np.mean(ms)) if ms else None,
            "own_gain": float(np.mean([np.mean([e["d"][t].mean() for t in tasks]) for e in fa["seeds"].values()]))}


# ----------------------------------------------------------------------------- gates from scores
def pac_kappa(scores_cal, eps, delta):
    return float(calibrate({c: v.tolist() for c, v in scores_cal.items()}, eps, delta)["kappa"])


def gate_eval(ex, score_fn, kappa, task, fams, tasks):
    fire = {p: score_fn(ex.X(f"eval__{p}")) > kappa for p in ex.eval_keys()}
    s_rl = score_fn(ex.X(f"rl__{task}"))
    st = {"kappa": kappa, "power": float((s_rl > kappa).mean()),
          "fire": {p: float(f.mean()) for p, f in fire.items()},
          "cp_dolly": clopper_pearson_upper(int(fire["dolly"].sum()), len(fire["dolly"])) if "dolly" in fire else None,
          "families": {fam: switched(fa, fire, tasks) for fam, fa in fams.items() if fa["method"] not in GATED}}
    return st


class Standardizer:
    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0, ddof=1) + 1e-6

    def __call__(self, X):
        return (np.asarray(X, np.float64) - self.mu) / self.sd


def lda_score(pos, neg, gamma=0.1):
    zs = Standardizer(np.concatenate([pos, neg]).astype(np.float64))
    P, N = zs(pos), zs(neg)
    mp, mn = P.mean(0), N.mean(0)
    S = ((P - mp).T @ (P - mp) + (N - mn).T @ (N - mn)) / (len(P) + len(N) - 2)
    d = S.shape[0]
    S = (1 - gamma) * S + gamma * (np.trace(S) / d) * np.eye(d)
    w = np.linalg.solve(S, mp - mn)
    b = -float(w @ (mp + mn)) / 2
    return lambda X: zs(X) @ w + b


def expertgate_score(pos, neg, k=32):
    """Expert Gate with linear autoencoders: one PCA (k components) per side; score = log of the reference side's
    reconstruction error over the new task side's (> 0: the new task's autoencoder reconstructs the prompt better)"""
    zs = Standardizer(np.concatenate([pos, neg]).astype(np.float64))
    sides = []
    for X in (zs(pos), zs(neg)):
        m = X.mean(0)
        _, _, Vt = np.linalg.svd(X - m, full_matrices=False)
        sides.append((m, Vt[:max(1, min(k, len(X) - 1, X.shape[1] // 2))].T))   # a code smaller than the input

    def err(Z, m, U):
        D = Z - m
        return ((D - (D @ U) @ U.T) ** 2).sum(1) + 1e-9

    return lambda X: (lambda Z: np.log(err(Z, *sides[1])) - np.log(err(Z, *sides[0])))(zs(X))


def knn_score(pos, neg, k=15, chunk=2048):
    """k nearest neighbours (cosine, standardized features), class-balanced: share of the k neighbours from the new
    task over its fitting share minus the same for the reference (> 0: the new task's side is closer)"""
    zs = Standardizer(np.concatenate([pos, neg]).astype(np.float64))
    F = np.concatenate([zs(pos), zs(neg)])
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    npos, nneg = len(pos), len(neg)

    def score(X):
        Z = zs(X)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        out = []
        for i in range(0, len(Z), chunk):
            S = Z[i:i + chunk] @ F.T
            idx = np.argpartition(-S, k, axis=1)[:, :k]
            t = lab[idx].sum(1)
            out.append(t / npos - (k - t) / nneg)
        return np.concatenate(out)

    return score


# ----------------------------------------------------------------------------- parts
def part_zoo(ex, fams, eps_list, alpha, delta):
    task, pos, neg, cal, _ = ex.stage_data(1, "ratio", None)
    probe, _ = ex.probe(1, "ratio", None, pos, neg)
    sup = Support(probe, pos, ex.X(f"rl__{task}"), alpha)
    gates = {"probe": (probe.score, 0.0), "lda": (lda_score(pos, neg), 0.0), "expertgate": (expertgate_score(pos, neg), 0.0),
             "knn": (knn_score(pos, neg), 0.0), "oneclass": (lambda X: -sup.dist(X), -sup.tau)}
    out = {}
    for name, (fn, natural) in gates.items():
        a = auroc(fn(ex.X(f"eval__{task}")), fn(ex.X("eval__dolly"))) if ex.has("eval__dolly") else None
        cs = {c: fn(X) for c, X in cal.items()}
        rows = {"natural": gate_eval(ex, fn, natural, task, fams, [task])}
        for e in eps_list:
            rows[f"{e:g}"] = gate_eval(ex, fn, pac_kappa(cs, e, delta), task, fams, [task])
        out[name] = {"auroc": a, "thresholds": rows}
        print(f"  zoo {name}: AUROC {a if a is None else round(a, 3)}, natural threshold fires on "
              f"{100 * rows['natural']['fire'].get('dolly', float('nan')):.1f}% of Dolly")
    return out


def part_middle(ex, fams, dims, draws, eps_list, delta, rng):
    task, pos, neg, cal, _ = ex.stage_data(1, "ratio", None)
    d = pos.shape[1]
    sets = {k: ex.X(k) for k in ex.z.files if k.startswith(("eval__", "rl__"))}
    fa = fams.get("seq")
    rows = []
    for k in [k for k in dims if k < d]:
        for r in range(draws):
            P = rng.normal(size=(d, k)).astype(np.float32) / math.sqrt(k)
            pr = fit_probe(pos @ P, neg @ P, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"]))
            sc = {key: pr.score(X @ P) for key, X in sets.items()}
            a = auroc(sc[f"eval__{task}"], sc["eval__dolly"]) if "eval__dolly" in sc else None
            cs = {c: pr.score(X @ P) for c, X in cal.items()}
            for e in eps_list:
                kap = pac_kappa(cs, e, delta)
                fire = {key[len("eval__"):]: v > kap for key, v in sc.items() if key.startswith("eval__")}
                beta = float((sc[f"rl__{task}"] > kap).mean())
                row = {"k": k, "draw": r, "eps": e, "auroc": a, "beta_rl": beta, "beta_eval": float(fire[task].mean()),
                       "dolly_fire": float(fire["dolly"].mean()) if "dolly" in fire else None}
                if fa is not None:
                    sw = switched(fa, fire, [task])
                    row.update({"meas": sw["gain"], "pred": beta * sw["own_gain"], "own": sw["own_gain"],
                                "chg": sw["chg"], "subst": sw["subst"]})
                rows.append(row)
    agg = []
    for k in sorted({r["k"] for r in rows}):
        for e in eps_list:
            sel = [r for r in rows if r["k"] == k and r["eps"] == e]
            g = {"k": k, "eps": e}
            for f in ("auroc", "beta_rl", "beta_eval", "dolly_fire", "meas", "pred", "chg", "subst"):
                v = [r[f] for r in sel if r.get(f) is not None]
                g[f] = float(np.mean(v)) if v else None
                g[f + "_sd"] = float(np.std(v)) if len(v) > 1 else 0.0
            agg.append(g)
    return {"rows": rows, "agg": agg}


def part_redraws(ex, fams, R, eps_list, delta, rng):
    task, pos0, _, _, _ = ex.stage_data(1, "ratio", None)
    fit_ref, cal_ref = ex.fit_ref, ex.cal_ref
    draws = []
    for r in range(R):
        neg, cal = [], {}
        for c in sorted(set(fit_ref) | set(cal_ref)):
            Xf = ex.X(f"fit__ref__{c}") if c in fit_ref else np.zeros((0, pos0.shape[1]), np.float32)
            Xc = ex.X(f"cal__ref__{c}") if c in cal_ref else np.zeros((0, pos0.shape[1]), np.float32)
            Xa = np.concatenate([Xf, Xc])
            perm = rng.permutation(len(Xa))
            if c in fit_ref:
                neg.append(Xa[perm[:len(Xf)]])
            if c in cal_ref:
                cal[c] = Xa[perm[len(Xf):len(Xf) + len(Xc)]]
        pos = pos0[rng.integers(0, len(pos0), len(pos0))]
        pr = fit_probe(pos, np.concatenate(neg), float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"]))
        cs = {c: pr.score(X) for c, X in cal.items()}
        d = {"draw": r}
        for e in eps_list:
            st = gate_eval(ex, pr.score, pac_kappa(cs, e, delta), task, {f: v for f, v in fams.items() if f == "seq"}, [task])
            d[f"{e:g}"] = {"power": st["power"], "fire": st["fire"],
                           "seq": st["families"].get("seq")}
        draws.append(d)
    summ = {}
    for e in eps_list:
        key = f"{e:g}"
        dol = [d[key]["fire"].get("dolly") for d in draws if d[key]["fire"].get("dolly") is not None]
        mm = [d[key]["fire"].get("mmlu") for d in draws if d[key]["fire"].get("mmlu") is not None]
        pw = [d[key]["power"] for d in draws]
        sg = [d[key]["seq"]["gain"] for d in draws if d[key]["seq"]]
        sc = [d[key]["seq"]["chg"] for d in draws if d[key]["seq"] and d[key]["seq"]["chg"] is not None]

        def ms(v):
            return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "min": float(np.min(v)), "max": float(np.max(v))} if v else None
        summ[key] = {"dolly": ms(dol), "dolly_above_eps": float(np.mean([x > e for x in dol])) if dol else None,
                     "mmlu": ms(mm), "power": ms(pw), "seq_gain": ms(sg), "seq_chg": ms(sc), "R": len(draws)}
    return {"draws": draws, "summary": summ}


def part_regioncov(ex, sizes, draws, alpha, rng):
    out = []
    for k in range(1, ex.K + 1):
        task, pos, neg, _, _ = ex.stage_data(k, "ratio", None)
        if not ex.has(f"eval__{task}"):
            continue
        probe, _ = ex.probe(k, "ratio", None, pos, neg)
        sup = Support(probe, pos, ex.X(f"rl__{task}"), alpha)
        d_rl, d_ev = sup.dist(ex.X(f"rl__{task}")), sup.dist(ex.X(f"eval__{task}"))
        Z = probe.z(ex.X(f"rl__{task}"))
        rows = []
        for n in [n for n in sizes if n <= len(d_rl) - 50]:
            cov_r, cov_h, cov_s = [], [], []
            for _ in range(draws):
                idx = rng.choice(len(d_rl), n, replace=False)
                tau = float(np.quantile(d_rl[idx], 1 - alpha))
                rest = np.ones(len(d_rl), bool)
                rest[idx] = False
                cov_r.append(float((d_ev <= tau).mean()))
                cov_h.append(float((d_rl[rest] <= tau).mean()))
                v = rng.normal(size=Z.shape[1])
                proj = Z @ v
                half = np.where(proj >= np.median(proj))[0]
                if len(half) >= n:
                    sidx = rng.choice(half, n, replace=False)
                    cov_s.append(float((d_ev <= float(np.quantile(d_rl[sidx], 1 - alpha))).mean()))
            rows.append({"n": n, "eval_mean": float(np.mean(cov_r)), "eval_q05": float(np.quantile(cov_r, 0.05)),
                         "heldrl_mean": float(np.mean(cov_h)),
                         "skew_mean": float(np.mean(cov_s)) if cov_s else None,
                         "skew_q05": float(np.quantile(cov_s, 0.05)) if cov_s else None})
        out.append({"stage": k, "task": task, "nominal": 1 - alpha, "n_rl": len(d_rl),
                    "full": float((d_ev <= sup.tau).mean()), "rows": rows})
    return out


def part_ci(ex, fams, tasks, gate_fire, B, rng):
    """paired bootstrap over prompts (per task) and seeds"""
    arr = {}
    for fam, fa in fams.items():
        arr[fam] = {sd: [e["d"][t] for t in tasks] for sd, e in fa["seeds"].items()}
    if "seq" in fams and gate_fire is not None:
        arr["seq@sw"] = {sd: [e["d"][t] * gate_fire[t] for t in tasks] for sd, e in fams["seq"]["seeds"].items()}
    IDX = [rng.integers(0, n, size=(B, n)) for n in (len(ex.X(f"eval__{t}")) for t in tasks)]

    def boot(fa_, fb_=None):
        seeds = sorted(set(arr[fa_]) & (set(arr[fb_]) if fb_ else set(arr[fa_])))
        if not seeds:
            return None
        S = rng.integers(0, len(seeds), size=(B, len(seeds)))

        def per_seed(fam):   # seeds x B: mean over tasks of the resampled prompt means
            return np.stack([np.mean([arr[fam][sd][i][IDX[i]].mean(1) for i in range(len(tasks))], axis=0)
                             for sd in seeds])
        A = per_seed(fa_)
        D = A - per_seed(fb_) if fb_ else A
        draws = D[S, np.arange(B)[:, None]].mean(1)
        point = float(np.mean([np.mean([arr[fa_][sd][i].mean() for i in range(len(tasks))]) for sd in seeds])
                      - (np.mean([np.mean([arr[fb_][sd][i].mean() for i in range(len(tasks))]) for sd in seeds])
                         if fb_ else 0.0))
        return {"point": point, "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
                "p_le0": float((draws <= 0).mean()), "seeds": len(seeds)}
    out = {"gain": {fam: boot(fam) for fam in arr}, "diff": {}}
    for a, b in (("tripwire_prompt", "seq"), ("router", "tripwire_prompt"), ("seq@sw", "tripwire_prompt"),
                 ("tripwire_prompt", "refkl"), ("tripwire_prompt_seq", "tripwire_prompt"), ("seq@sw", "seq"),
                 ("refkl", "seq"), ("tripwire", "seq")):
        if a in arr and b in arr:
            out["diff"][f"{a}|{b}"] = boot(a, b)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--out", default="", help="the switch report's folder (default paper/v2_results/<exp>/switch)")
    ap.add_argument("--name", default="", help="letters only: macro prefix \\rv<name>...")
    ap.add_argument("--parts", default="zoo,middle,redraws,regioncov,ci")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--eps", default="0.03,0.1")
    ap.add_argument("--dims", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--redraws", type=int, default=20)
    ap.add_argument("--sizes", default="25,50,100,200,400,800")
    ap.add_argument("--cov-draws", type=int, default=100)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.name and not args.name.isalpha():
        raise SystemExit("--name must be letters only")
    cfg = load_config(args.config, args.set)
    runs = args.runs or os.path.join(cfg["out_dir"], cfg["exp"])
    feat = args.features or os.path.join(runs, "analysis", "gate_features.npz")
    sw_dir = args.out or os.path.join("paper", "v2_results", cfg["exp"], "switch")
    sp = os.path.join(sw_dir, "switch_summary.json")
    if not (os.path.exists(feat) and os.path.exists(sp)):
        raise SystemExit(f"needs {feat} and {sp} (scripts/switch_report.py first)")
    out_dir = os.path.join(sw_dir, "review")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summ = json.load(open(sp))
    ex = Experiment(cfg, feat, feat[:-4] + ".json")
    tasks, single = ex.tasks, ex.K == 1
    delta = float(ex.tc["delta"])
    eps_list = [float(e) for e in args.eps.split(",") if e.strip()]
    main_eps = float(summ["main_eps"])
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    lens = {p: len(ex.X(f"eval__{p}")) for p in ex.eval_keys()}
    fams = run_arrays(finished_runs(runs, ex.K), tasks, lens)
    label = readable_label(cfg).replace("_", r"\_")
    res = {"exp": cfg["exp"], "label": readable_label(cfg)}
    nums, pre = {}, "rv" + args.name

    if "zoo" in parts and single and ex.has("eval__dolly"):
        zoo = part_zoo(ex, fams, eps_list, args.alpha, delta)
        res["zoo"] = zoo
        names = {"probe": "probe (logistic)", "lda": "Gaussian (LDA)", "expertgate": "Expert Gate (PCA autoencoders)",
                 "knn": "15 nearest neighbours", "oneclass": "one-class (new-task region)"}
        tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
              r"\caption{Gates built from the same unlabeled prompts and trunk features (" + label + r"), at their own "
              r"threshold (natural; as a router uses them) and PAC-calibrated. Fire: firing on held-out pools (\%); CP: "
              r"95\% bound on Dolly; Seq$^{\mathrm{sw}}$: Seq-GRPO behind the gate, new-task gain (points) and Dolly "
              r"responses changed (\%). Only the calibrated rows carry a certificate.}",
              r"\label{tab:zoo_" + (args.name.lower() or "x") + "}", r"\resizebox{\linewidth}{!}{%",
              r"\begin{tabular}{l r l r r r r r r r}", r"\toprule",
              r"Gate & AUROC & Threshold & Power & Dolly & CP & MMLU & AdvB. & Gain$^{\mathrm{sw}}$ & Chg.$^{\mathrm{sw}}$ \\",
              r"\midrule"]
        for g, z in zoo.items():
            first = True
            for th, st in z["thresholds"].items():
                sq = st["families"].get("seq") or {}
                tl.append(" & ".join([names[g] if first else "", f"{z['auroc']:.3f}" if first and z["auroc"] else "",
                                      "natural" if th == "natural" else r"$\varepsilon=" + th + "$", f2(st["power"]),
                                      f2(st["fire"].get("dolly")), f2(st["cp_dolly"]), f2(st["fire"].get("mmlu")),
                                      f2(st["fire"].get("advbench")), f2(sq.get("gain"), signed=True), f2(sq.get("chg"))])
                          + r" \\")
                first = False
                G = g.capitalize()
                T = "Nat" if th == "natural" else ("A" if th == f"{eps_list[0]:g}" else "B")
                nums[f"{pre}Zoo{G}{T}Dolly"] = f2(st["fire"].get("dolly"))
                nums[f"{pre}Zoo{G}{T}Power"] = f2(st["power"])
                nums[f"{pre}Zoo{G}{T}Gain"] = f2(sq.get("gain"), signed=True)
                nums[f"{pre}Zoo{G}{T}Mmlu"] = f2(st["fire"].get("mmlu"))
            if z["auroc"] is not None:
                nums[f"{pre}Zoo{g.capitalize()}Auroc"] = f"{z['auroc']:.3f}"
            tl.append(r"\midrule" if g != list(zoo)[-1] else "")
        nat = [z["thresholds"]["natural"]["fire"].get("dolly") for z in zoo.values()]
        nat = [x for x in nat if x is not None]
        if nat:
            nums[pre + "ZooNatMin"], nums[pre + "ZooNatMax"] = f2(min(nat)), f2(max(nat))
        tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        with open(os.path.join(out_dir, "zoo_table.tex"), "w") as fh:
            fh.write("\n".join(x for x in tl if x != "") + "\n")

    if "middle" in parts and single and ex.has("eval__dolly"):
        mid = part_middle(ex, fams, [int(x) for x in args.dims.split(",")], args.draws, eps_list, delta, rng)
        res["middle"] = mid
        tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
              r"\caption{The middle of the separability scale, simulated (" + label + r"): the probe fitted on $k$ random "
              r"projections of the trunk features (mean of " + str(args.draws) + r" draws) and PAC-calibrated. "
              r"$\beta$: firing on unlabeled RL prompts; Pred.: $\beta G$; Meas.: Seq-GRPO's gain behind the gate "
              r"(points); Dolly: firing on held-out Dolly prompts (\%).}",
              r"\label{tab:middle_" + (args.name.lower() or "x") + "}", r"\begin{tabular}{r r r r r r r}", r"\toprule",
              r"$k$ & AUROC & $\varepsilon$ & $\beta$ & Pred. & Meas. & Dolly \\", r"\midrule"]
        for g in mid["agg"]:
            tl.append(" & ".join([str(g["k"]), f"{g['auroc']:.3f}" if g["auroc"] is not None else "--", f"{g['eps']:g}",
                                  f2(g["beta_rl"]), f2(g["pred"], signed=True), f2(g["meas"], signed=True),
                                  f2(g["dolly_fire"])]) + r" \\")
        tl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(os.path.join(out_dir, "middle_table.tex"), "w") as fh:
            fh.write("\n".join(tl) + "\n")
        with open(os.path.join(out_dir, "middle.dat"), "w") as fh:
            fh.write("k eps auroc beta pred meas dolly\n")
            for g in mid["agg"]:
                vals = [g["auroc"], g["beta_rl"], g["pred"], g["meas"], g["dolly_fire"]]
                fh.write(f"{g['k']} {g['eps']:g} " + " ".join("nan" if v is None else f"{v:.4f}" for v in vals) + "\n")
        band = [g for g in mid["agg"] if g["auroc"] is not None and 0.7 <= g["auroc"] <= 0.9 and g["meas"] is not None]
        if band:
            errs = [abs(g["meas"] - g["pred"]) for g in band]
            nums[pre + "MidN"] = str(len(band))
            nums[pre + "MidMae"] = f2(sum(errs) / len(errs))
            at_main = [g["meas"] for g in band if g["eps"] == main_eps]
            if at_main:
                nums[pre + "MidMeasMax"] = f2(max(at_main), signed=True)

    if "redraws" in parts and single and args.redraws > 0:
        rd = part_redraws(ex, fams, args.redraws, eps_list, delta, rng)
        res["redraws"] = rd
        s = rd["summary"].get(f"{main_eps:g}")
        if s:
            if s["dolly"]:
                nums[pre + "RdDollyMean"], nums[pre + "RdDollyMax"] = f2(s["dolly"]["mean"]), f2(s["dolly"]["max"])
                nums[pre + "RdDollyAbove"] = f2(s["dolly_above_eps"])
            nums[pre + "RdPowerMean"], nums[pre + "RdPowerSd"] = f2(s["power"]["mean"]), f2(s["power"]["sd"])
            if s["seq_gain"]:
                nums[pre + "RdGainMean"] = f2(s["seq_gain"]["mean"], signed=True)
                nums[pre + "RdGainSd"] = f2(s["seq_gain"]["sd"])
            nums[pre + "RdN"] = str(s["R"])

    if "regioncov" in parts:
        rc = part_regioncov(ex, [int(x) for x in args.sizes.split(",")], args.cov_draws, args.alpha, rng)
        res["regioncov"] = rc
        tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
              r"\caption{Held-out coverage of the new task's region (" + label + r", nominal " + f2(1 - args.alpha)
              + r"\%) when its threshold is set on $n$ unlabeled new-task prompts: mean and 5th percentile over "
              + str(args.cov_draws) + r" random subsets, and over skewed subsets (one half of a random projection).}",
              r"\label{tab:regioncov_" + (args.name.lower() or "x") + "}", r"\begin{tabular}{l r r r r r}", r"\toprule",
              r"Task & $n$ & Mean & q05 & Skewed mean & Skewed q05 \\", r"\midrule"]
        for st in rc:
            for i, r in enumerate(st["rows"]):
                tl.append(" & ".join([st["task"] if i == 0 else "", str(r["n"]), f2(r["eval_mean"]), f2(r["eval_q05"]),
                                      f2(r["skew_mean"]), f2(r["skew_q05"])]) + r" \\")
            if st["stage"] == 1 and st["rows"]:
                small = st["rows"][0]
                nums[pre + "RcSmallN"] = str(small["n"])
                nums[pre + "RcSmallQ"] = f2(small["eval_q05"])
                nums[pre + "RcSmallSkew"] = f2(small["skew_q05"])
                nums[pre + "RcFull"] = f2(st["full"])
        tl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(os.path.join(out_dir, "regioncov_table.tex"), "w") as fh:
            fh.write("\n".join(tl) + "\n")

    if "ci" in parts and fams:
        g = build_variant(ex, "ratio", main_eps, args.alpha)
        fire = {t: g["any"][t] for t in tasks if t in g["any"]}
        ci = part_ci(ex, fams, tasks, fire if len(fire) == len(tasks) else None, args.boot, rng)
        res["ci"] = ci
        for fam, c in ci["gain"].items():
            if c and fam in FAMN:
                nums[f"{pre}Ci{FAMN[fam]}"] = f"[{f2(c['lo'], signed=True)}, {f2(c['hi'], signed=True)}]"
        for key, c in ci["diff"].items():
            a, b = key.split("|")
            if c and a in FAMN and b in FAMN:
                m = f"{pre}Diff{FAMN[a]}Vs{FAMN[b]}"
                nums[m] = f2(c["point"], signed=True)
                nums[m + "Ci"] = f"[{f2(c['lo'], signed=True)}, {f2(c['hi'], signed=True)}]"
                nums[m + "P"] = pv(c["p_le0"])

    with open(os.path.join(out_dir, "review.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "review_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/review_analyses.py\n")
        for k in sorted(nums):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    print(f"wrote {out_dir}: parts {sorted(p for p in parts if p in res)}, {len(nums)} macros")


if __name__ == "__main__":
    main()
