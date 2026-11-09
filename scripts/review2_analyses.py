#!/usr/bin/env python
"""Review add-ons, round 2 (Oct 29), on stored features and outputs (CPU only: numpy + scipy; no GPU, no new runs).

    python scripts/review2_analyses.py --config configs/v2/overlap_q15.yaml --name Nq [--out paper/v2_results/<exp>/switch]
        [--parts sens,cov,budget]
    python scripts/review2_analyses.py --aggregate paper/v2_results     # pooled tables and numbers, every experiment

Needs <runs>/analysis/gate_features.npz and the switch report of the experiment (switch_summary.json in --out).
Writes <out>/review2/review2.json and review2_numbers.tex (macros \\rvb<name>...); --aggregate reads every
<results>/*/switch/review2/review2.json and writes <results>/overview/review2/ (tables *.tex, macros \\rvbAll...).
  sens    the two-sided tripwire's settings, one at a time around the paper's (alpha 0.02, shrinkage 0.1, every
          unlabeled RL prompt of the task): alpha; the shrinkage of the region's covariance (1 = no covariance:
          Euclidean distance); the number n of unlabeled new-task prompts its threshold is set on (random subsets,
          --sens-draws draws). Each setting is PAC-calibrated at the paper's eps. Per setting: power on held-out
          new-task prompts, the region's held-out coverage, firing on Dolly (with its 95% bound), MMLU, XSTest,
          AdvBench, AdvBench in jailbreak templates, harmful requests in the task's own format, Dolly / AdvBench
          after a system prompt or an earlier turn, and Seq-GRPO behind the gate. The probe without a region is
          the first row.
  cov     the covariance term of Proposition 4.5 per task: the gain behind a gate is beta G + Cov(gain, firing).
          (a) Per task and ungated method, behind the deployed gate: measured gain, beta G (beta on unlabeled RL
          prompts, as predicted before training), the error, and the gain difference between fired and non-fired
          held-out prompts (Cov = beta (1 - beta) times it), with 95% paired-bootstrap intervals over prompts and
          seeds. (b) Seq-GRPO behind constructed gates that fire on as many held-out prompts as the deployed gate:
          the worst and the best possible (oracle on the per-prompt gain, so outside the label-free setting), one
          that fires first on the prompts the base model already solves, a feature adversary (ridge regression on
          the trunk features, cross-fitted, predicting each prompt's gain; it fires on the lowest predictions: the
          most a gate on these features could exploit) and random gates (mean and 95% range).
  budget  a stream of K stages that shares one budget: each real stage's gate calibrated at eps/K (K known) and at
          6 eps / (pi^2 K^2) (the K-th stage when K is not known in advance), K = 1 ... 50. Per K: whether the
          calibration sets can certify that level, the level they do certify, the calibration size each reference
          component would need, power on held-out new-task prompts, firing on held-out Dolly.
"""
from __future__ import annotations

import argparse
import glob
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
from switch_analyses import f2, finished_runs  # noqa: E402
from switch_report import GATED, Experiment, Support, build_variant, readable_label  # noqa: E402
from review_analyses import run_arrays, switched  # noqa: E402

JB = ("jb_prefix", "jb_roleplay", "jb_hypo")
TURN = ("dolly_sys", "dolly_turn", "advbench_sys", "advbench_turn")
KS = (1, 2, 3, 5, 10, 20, 50)


# ----------------------------------------------------------------------------- shared
def calib_kappa(cs, refs_cal, eps_ref, eps_old, delta):
    """PAC threshold as build_variant sets it (no union): reference components at eps_ref, earlier tasks at eps_old,
    delta split over components. Returns kappa and, per reference component, (n, feasible, eps_cert)."""
    n_all = len(cs)
    refs = {c: v for c, v in cs.items() if c in refs_cal}
    olds = {c: v for c, v in cs.items() if c not in refs_cal}
    kappa, info = -np.inf, {}
    for part, e in ((refs, eps_ref), (olds, eps_old)):
        if not part:
            continue
        r = calibrate({c: v.tolist() for c, v in part.items()}, e, delta * len(part) / n_all)
        kappa = max(kappa, float(r["kappa"]))
        if part is refs:
            info = {c: (v["n"], v["feasible"], v["eps_cert"], r["delta_per_component"]) for c, v in r["components"].items()}
    return float(kappa), info


def pool_stats(fire, tasks):
    """firing summary of a gate (fire: pool -> bool[] of the stream's union gate)"""
    g = lambda p: float(fire[p].mean()) if p in fire else None  # noqa: E731
    jb = [g(p) for p in JB if p in fire]
    wr = [g(p) for p in fire if p.startswith("advwrap_")]
    out = {p: g(p) for p in ("dolly", "mmlu", "xstest", "advbench") + TURN if p in fire}
    out["power"] = float(np.mean([g(t) for t in tasks if t in fire])) if any(t in fire for t in tasks) else None
    out["jb"] = float(np.mean(jb)) if jb else None
    out["jb_max"] = float(np.max(jb)) if jb else None
    out["wrap"] = float(np.mean(wr)) if wr else None
    out["cp_dolly"] = clopper_pearson_upper(int(fire["dolly"].sum()), len(fire["dolly"])) if "dolly" in fire else None
    return out


# ----------------------------------------------------------------------------- sens
def stage_cache(ex, k, gamma, cache):
    """probe scores and region distances of stage k's gate (the region's shape depends only on the shrinkage; alpha
    and n only move its threshold), computed once"""
    key = (k, gamma)
    if key not in cache:
        task, pos, neg, cal, refs_cal = ex.stage_data(k, "twosided", None)
        probe, _ = ex.probe(k, "twosided", None, pos, neg)
        sup = Support(probe, pos, ex.X(f"rl__{task}"), 0.02, gamma)
        sd = lambda X: (probe.score(X), sup.dist(X))  # noqa: E731
        cache[key] = {"task": task, "refs_cal": refs_cal, "rl": sup.dist(ex.X(f"rl__{task}")),
                      "cal": {c: sd(X) for c, X in cal.items()},
                      "eval": {p: sd(ex.X(f"eval__{p}")) for p in ex.eval_keys()}}
    return cache[key]


def twosided_eval(ex, fams, eps, delta, alpha, gamma, cache, rng=None, n=None):
    fire, cover = {}, []
    for k in range(1, ex.K + 1):
        c = stage_cache(ex, k, gamma, cache)
        d_rl = c["rl"]
        if n is not None:
            if n > len(d_rl) - 10:
                return None
            d_rl = d_rl[rng.choice(len(d_rl), n, replace=False)]
        tau = float(np.quantile(d_rl, 1.0 - alpha))
        kappa, _ = calib_kappa({cc: np.where(dd <= tau, ss, -np.inf) for cc, (ss, dd) in c["cal"].items()},
                               c["refs_cal"], eps, eps, delta)
        for p, (ss, dd) in c["eval"].items():
            f = (dd <= tau) & (ss > kappa)
            fire[p] = f if p not in fire else (fire[p] | f)
        if c["task"] in c["eval"]:
            cover.append(float((c["eval"][c["task"]][1] <= tau).mean()))
    st = pool_stats(fire, ex.tasks)
    st["coverage"] = float(np.mean(cover)) if cover else None
    if "seq" in fams:
        sw = switched(fams["seq"], fire, ex.tasks)
        st.update({"seq_gain": sw["gain"], "seq_chg": sw["chg"], "seq_own": sw["own_gain"]})
    return st


def part_sens(ex, fams, eps, delta, alphas, gammas, sizes, draws, rng):
    rows = []
    g = build_variant(ex, "ratio", eps, 0.02)
    st = pool_stats(g["any"], ex.tasks)
    st["coverage"] = None
    if "seq" in fams:
        sw = switched(fams["seq"], g["any"], ex.tasks)
        st.update({"seq_gain": sw["gain"], "seq_chg": sw["chg"], "seq_own": sw["own_gain"]})
    rows.append({"setting": "probe", "what": "probe only (no region)", **st})
    grid = [("paper", "paper: alpha 0.02, shrinkage 0.1, all n", 0.02, 0.1, None)]
    grid += [(f"alpha{a:g}", f"alpha {a:g}", a, 0.1, None) for a in alphas if a != 0.02]
    grid += [(f"shrink{s:g}", f"shrinkage {s:g}", 0.02, s, None) for s in gammas if s != 0.1]
    grid += [(f"n{n}", f"n = {n}", 0.02, 0.1, n) for n in sizes]
    cache = {}
    for key, what, a, s, n in grid:
        if n is None:
            st = twosided_eval(ex, fams, eps, delta, a, s, cache)
            if st is not None:
                rows.append({"setting": key, "what": what, "draws": 1, **st})
        else:
            sts = [twosided_eval(ex, fams, eps, delta, a, s, cache, rng, n) for _ in range(draws)]
            sts = [x for x in sts if x is not None]
            if not sts:
                continue
            agg = {}
            for f in sts[0]:
                v = [x[f] for x in sts if x.get(f) is not None]
                agg[f] = float(np.mean(v)) if v else None
            agg["power_min"] = float(np.min([x["power"] for x in sts]))
            agg["dolly_max"] = float(np.max([x["dolly"] for x in sts])) if sts[0].get("dolly") is not None else None
            agg["jb_max"] = float(np.max([x["jb_max"] for x in sts])) if sts[0].get("jb_max") is not None else None
            rows.append({"setting": key, "what": what, "draws": len(sts), **agg})
        r = rows[-1]
        print(f"  sens {key}: power {f2(r['power'])}, Dolly {f2(r.get('dolly'))}, jailbreak max {f2(r.get('jb_max'))}")
    return rows


# ----------------------------------------------------------------------------- cov
def ridge_oof(X, y, folds, rng, lams=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)):
    """out-of-fold predictions of kernel (dual) ridge on standardized features; lambda by cross-validated R^2"""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Kf = Z @ Z.T / Z.shape[1]
    n = len(y)
    fold = rng.permutation(n) % folds
    best = None
    for lam in lams:
        p = np.zeros(n)
        for f in range(folds):
            tr, te = fold != f, fold == f
            mu = y[tr].mean()
            a = np.linalg.solve(Kf[np.ix_(tr, tr)] + lam / Z.shape[1] * np.eye(tr.sum()), y[tr] - mu)
            p[te] = Kf[np.ix_(te, tr)] @ a + mu
        r2 = 1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if best is None or r2 > best[0]:
            best = (r2, lam, p)
    return best


def pick_lowest(v, m, rng):
    """indices of the m smallest values of v, ties broken at random"""
    return np.lexsort((rng.random(len(v)), v))[:m]


def boot_task(D, f, b_rl, B, rng):
    """D: seeds x prompts (per-prompt gain), f: bool firing on the held-out prompts"""
    S, n = D.shape
    Ff = f.astype(np.float64)

    def stats(Dm, Fm):   # Dm: ... x n, Fm: ... x n -> meas, G, beta, delta(fired - not)
        Fm = np.broadcast_to(Fm, Dm.shape)
        meas = (Dm * Fm).mean(-1)
        G = Dm.mean(-1)
        beta = Fm.mean(-1)
        nf = Fm.sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dlt = (Dm * Fm).sum(-1) / nf - (Dm * (1 - Fm)).sum(-1) / (n - nf)
        return meas, G, beta, dlt
    meas, G, beta, dlt = stats(D, Ff[None])
    pt = {"meas": float(meas.mean()), "G": float(G.mean()), "beta_eval": float(Ff.mean()), "beta_rl": float(b_rl),
          "seeds": S, "n": n}
    pt["pred"] = pt["beta_rl"] * pt["G"]
    pt["err"] = pt["meas"] - pt["pred"]
    pt["cov"] = pt["meas"] - pt["beta_eval"] * pt["G"]
    pt["dfire"] = float(np.nanmean(dlt)) if 0 < Ff.sum() < n else None
    idx = rng.integers(0, n, size=(B, n))
    sidx = rng.integers(0, S, size=(B, S))
    Fb = Ff[idx]                                   # B x n
    Db = D[:, idx]                                 # S x B x n
    mb, Gb, bb, db = stats(Db, Fb[None])           # S x B each
    pick = lambda A: A[sidx, np.arange(B)[:, None]].mean(1)  # noqa: E731   (resampled seeds)
    mb, Gb, bb, db = pick(mb), pick(Gb), pick(bb), pick(db)
    for key, v in (("err", mb - b_rl * Gb), ("cov", mb - bb * Gb), ("dfire", db), ("meas", mb), ("G", Gb)):
        v = v[np.isfinite(v)]
        if key == "dfire" and pt["dfire"] is None:
            continue
        pt[key + "_lo"], pt[key + "_hi"] = (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (None, None)
    return pt


def part_cov(ex, fams, eps, alpha, B, n_random, folds, rng):
    g = build_variant(ex, "ratio", eps, alpha)
    rows, cons = [], []
    for t in ex.tasks:
        if t not in g["any"]:
            continue
        f = g["any"][t]
        b_rl = float(g["rl_any"][t].mean())
        for fam, fa in fams.items():
            if fa["method"] in GATED or not fa["seeds"]:
                continue
            D = np.stack([e["d"][t] for e in fa["seeds"].values()])
            r = boot_task(D, f, b_rl, B, rng)
            rows.append({"task": t, "family": fam, **r})
        fa = fams.get("seq")
        m, n = int(f.sum()), len(f)
        if fa is None or not 0 < m < n:
            continue
        D = np.stack([e["d"][t] for e in fa["seeds"].values()])
        dbar = D.mean(0)
        base = np.mean([e["base"][t] for e in fa["seeds"].values()], axis=0)
        G = float(D.mean())

        def kept(sel):
            mask = np.zeros(n, bool)
            mask[sel] = True
            return float((D * mask[None]).mean())
        r2, lam, p = ridge_oof(ex.X(f"eval__{t}").astype(np.float64), dbar, folds, rng)
        rand = np.array([kept(rng.choice(n, m, replace=False)) for _ in range(n_random)])
        c = {"task": t, "m": m, "n": n, "beta_eval": m / n, "beta_rl": b_rl, "G": G, "pred_eval": m / n * G,
             "pred_rl": b_rl * G, "deployed": kept(np.where(f)[0]),
             "worst": kept(pick_lowest(dbar, m, rng)), "best": kept(pick_lowest(-dbar, m, rng)),
             "solved_first": kept(pick_lowest(-base, m, rng)),
             "feature_adv": kept(pick_lowest(p, m, rng)), "feature_r2": float(r2), "feature_lambda": lam,
             "random_mean": float(rand.mean()), "random_lo": float(np.quantile(rand, 0.025)),
             "random_hi": float(np.quantile(rand, 0.975))}
        cons.append(c)
        print(f"  cov {t}: beta {f2(m / n)}, G {f2(G, signed=True)}; kept: deployed {f2(c['deployed'], signed=True)}, "
              f"worst {f2(c['worst'], signed=True)}, feature adversary {f2(c['feature_adv'], signed=True)} "
              f"(R2 {r2:.2f}), random {f2(c['random_mean'], signed=True)}")
    return {"rows": rows, "constructed": cons}


# ----------------------------------------------------------------------------- budget
def n_needed(e, d_c):
    """smallest calibration size at which level e can be certified (the largest score as threshold): (1-e)^n <= d_c"""
    return int(math.ceil(math.log(d_c) / math.log1p(-e))) if 0 < e < 1 else None


def part_budget(ex, eps, delta, ks):
    out = []
    for k in range(1, ex.K + 1):
        task, pos, neg, cal, refs_cal = ex.stage_data(k, "ratio", None)
        probe, _ = ex.probe(k, "ratio", None, pos, neg)
        cs = {c: probe.score(X) for c, X in cal.items()}
        s_task = probe.score(ex.X(f"eval__{task}")) if ex.has(f"eval__{task}") else None
        s_dol = probe.score(ex.X("eval__dolly")) if ex.has("eval__dolly") else None
        rows = []
        for K in ks:
            for sched, e in (("split", eps / K), ("anytime", 6 * eps / (math.pi ** 2 * K * K))):
                kap, info = calib_kappa(cs, refs_cal, e, eps, delta)
                feas = all(v[1] for v in info.values())
                cert = max(v[2] for v in info.values())
                need = max(n_needed(e, v[3]) for v in info.values())
                have = min(v[0] for v in info.values())
                rows.append({"K": K, "sched": sched, "eps_stage": e, "feasible": bool(feas), "eps_cert": float(cert),
                             "n_need": need, "n_have": have,
                             "power": float((s_task > kap).mean()) if s_task is not None else None,
                             "dolly": float((s_dol > kap).mean()) if s_dol is not None else None})
        out.append({"stage": k, "task": task, "rows": rows})
        r10 = [r for r in rows if r["K"] == 10 and r["sched"] == "split"]
        if r10:
            r = r10[0]
            print(f"  budget {task}, K=10 (eps/K {100 * r['eps_stage']:.2f}%): certifiable {r['feasible']} "
                  f"(n {r['n_have']}, needs {r['n_need']}), power {f2(r['power'])}")
    return out


# ----------------------------------------------------------------------------- aggregate
TASKN = {"nq": "NQ", "ifc": "IF", "concise": "Concise", "concisectx": "Concise-ctx", "gsm8k": "GSM8K",
         "countdown": "Countdown", "kk": "K\\&K", "math": "MATH", "sudoku": "Sudoku"}


def short_label(lbl):
    return lbl.replace("Qwen2.5-", "Q-").replace("-Instruct", "").replace("_", r"\_")


def ci(v, lo, hi):
    return f"{f2(v, signed=True)} [{f2(lo, signed=True)}, {f2(hi, signed=True)}]" if v is not None else "--"


def aggregate(results):
    files = sorted(glob.glob(os.path.join(results, "*", "switch", "review2", "review2.json")))
    if not files:
        raise SystemExit(f"no {results}/*/switch/review2/review2.json yet")
    R = [json.load(open(f)) for f in files]
    out = os.path.join(results, "overview", "review2")
    os.makedirs(out, exist_ok=True)
    nums = {}
    # ---- (a) per-task covariance, Seq-GRPO behind the deployed gate
    tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
          r"\caption{The covariance term of Proposition~\ref{prop:switchgain}, per task: Seq-GRPO behind the deployed "
          r"gate ($\varepsilon=0.03$). $\beta$: firing on unlabeled RL prompts (known before training); $G$: "
          r"Seq-GRPO's own gain; Kept: measured gain behind the gate; Err.: Kept $-\,\beta G$; $\Delta$: gain on fired "
          r"minus non-fired held-out prompts ($\mathrm{Cov}=\beta(1-\beta)\Delta$). Brackets: 95\% paired-bootstrap "
          r"intervals over prompts and seeds ($\beta$ held fixed, as it is known before training). Points.}",
          r"\label{tab:covtask}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{l l r r r r l l}", r"\toprule",
          r"Experiment & Task & $\beta$ & $G$ & Kept & $\beta G$ & Err. [95\% CI] & $\Delta$ [95\% CI] \\", r"\midrule"]
    rows_all, bl = [], []
    for r in R:
        for x in r.get("cov", {}).get("rows", []):
            if x["family"] != "seq":
                continue
            rows_all.append((r["label"], x))
            tl.append(" & ".join([short_label(r["label"]), TASKN.get(x["task"], x["task"]), f2(x["beta_rl"]),
                                  f2(x["G"], signed=True), f2(x["meas"], signed=True), f2(x["pred"], signed=True),
                                  ci(x["err"], x.get("err_lo"), x.get("err_hi")),
                                  ci(x["dfire"], x.get("dfire_lo"), x.get("dfire_hi")) if x["dfire"] is not None else "--"])
                      + r" \\")
            if x["beta_rl"] < 0.99:
                bl.append(x)
    tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(out, "covtask_table.tex"), "w") as fh:
        fh.write("\n".join(tl) + "\n")
    allfam = [x for r in R for x in r.get("cov", {}).get("rows", [])]
    if allfam:
        nums["rvbAllCovN"] = str(len(allfam))
        nums["rvbAllCovZero"] = str(sum(1 for x in allfam if x.get("err_lo") is not None and x["err_lo"] <= 0 <= x["err_hi"]))
        nums["rvbAllCovErrMax"] = f2(max(abs(x["err"]) for x in allfam))
    if bl:
        nums["rvbAllBlN"] = str(len(bl))
        nums["rvbAllBlErrMae"] = f2(float(np.mean([abs(x["err"]) for x in bl])))
        nums["rvbAllBlErrMax"] = f2(max(abs(x["err"]) for x in bl))
        nums["rvbAllBlZero"] = str(sum(1 for x in bl if x.get("err_lo") is not None and x["err_lo"] <= 0 <= x["err_hi"]))
        nums["rvbAllBlRel"] = f2(float(np.mean([abs(x["err"]) for x in bl])) / max(float(np.mean([abs(x["meas"]) for x in bl])), 1e-9))
    # ---- (b) constructed gates
    tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\caption{How far the covariance term can move the gain kept behind a gate. Seq-GRPO's gain (points) behind "
          r"gates that fire on as many held-out prompts as the deployed gate. Worst / best: oracle on each prompt's "
          r"gain (not label-free); solved: fires first on prompts the base model already solves; features: a ridge "
          r"regression on the trunk features, cross-fitted, that predicts each prompt's gain ($R^2$ in brackets) and "
          r"fires on the lowest predictions; random: mean and 95\% range of 1000 random gates ($=\beta G$ in "
          r"expectation). Tasks with $\beta<1$.}",
          r"\label{tab:covcons}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{l l r r r r r r l l}", r"\toprule",
          r"Experiment & Task & $\beta$ & $G$ & Deployed & Worst & Best & Solved & Features ($R^2$) & Random \\",
          r"\midrule"]
    cons = []
    for r in R:
        for c in r.get("cov", {}).get("constructed", []):
            cons.append((r, c))
            tl.append(" & ".join([short_label(r["label"]), TASKN.get(c["task"], c["task"]), f2(c["beta_eval"]),
                                  f2(c["G"], signed=True), f2(c["deployed"], signed=True), f2(c["worst"], signed=True),
                                  f2(c["best"], signed=True), f2(c["solved_first"], signed=True),
                                  f"{f2(c['feature_adv'], signed=True)} ({c['feature_r2']:.2f})",
                                  f"{f2(c['random_mean'], signed=True)} [{f2(c['random_lo'], signed=True)}, "
                                  f"{f2(c['random_hi'], signed=True)}]"]) + r" \\")
    tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(out, "covcons_table.tex"), "w") as fh:
        fh.write("\n".join(tl) + "\n")
    if cons:
        nums["rvbAllConsN"] = str(len(cons))
        nums["rvbAllConsRtwoMax"] = f"{max(c['feature_r2'] for _, c in cons):.2f}"
        nums["rvbAllConsFeatGapMax"] = f2(max(abs(c["feature_adv"] - c["pred_eval"]) for _, c in cons))
        nums["rvbAllConsDepGapMax"] = f2(max(abs(c["deployed"] - c["pred_eval"]) for _, c in cons))
        nums["rvbAllConsWorstGapMax"] = f2(max(c["pred_eval"] - c["worst"] for _, c in cons))
        for r, c in cons:   # the NQ 1.5B example in the text
            if r["name"] == "Nq" and c["task"] == "nq":
                for key in ("deployed", "worst", "best", "solved_first", "feature_adv", "random_mean", "pred_eval", "G"):
                    nums["rvbNq" + "".join(w.capitalize() for w in key.split("_"))] = f2(c[key], signed=True)
                nums["rvbNqBeta"] = f2(c["beta_eval"])
                nums["rvbNqRtwo"] = f"{c['feature_r2']:.2f}"
    # ---- sensitivity of the two-sided tripwire
    keys = []
    for r in R:
        for x in r.get("sens", []):
            if x["setting"] not in keys:
                keys.append(x["setting"])
    tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
          r"\caption{Sensitivity of the two-sided tripwire to its settings, one at a time around the paper's "
          r"($\alpha=0.02$, shrinkage $0.1$, every unlabeled RL prompt), each PAC-calibrated at $\varepsilon=0.03$. Mean "
          r"over " + str(len(R)) + r" experiments (streams: the union gate). Power: firing on held-out new-task prompts; "
          r"Cover: held-out new-task prompts inside the region; JB: AdvBench in jailbreak templates (mean / largest "
          r"template and experiment); Wrap: harmful requests in the task's format; Turn: Dolly and AdvBench after a "
          r"system prompt or an earlier turn (mean); Gain$^{\mathrm{sw}}$: Seq-GRPO behind the gate. Percent; gain in "
          r"points.}",
          r"\label{tab:sens}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{l r r r r r r r r}", r"\toprule",
          r"Setting & Power & Cover & Dolly & MMLU & JB & Wrap & Turn & Gain$^{\mathrm{sw}}$ \\", r"\midrule"]
    sens_rows = {}
    for key in keys:
        xs = [(r, x) for r in R for x in r.get("sens", []) if x["setting"] == key]

        def mean(f, xs=xs):
            v = [x.get(f) for _, x in xs if x.get(f) is not None]
            return float(np.mean(v)) if v else None

        def mx(f, xs=xs):
            v = [x.get(f) for _, x in xs if x.get(f) is not None]
            return float(np.max(v)) if v else None
        turn = [x.get(p) for _, x in xs for p in TURN if x.get(p) is not None]
        what = xs[0][1]["what"].replace("alpha", r"$\alpha$")
        if xs[0][1].get("draws", 1) > 1:
            what += f" ({xs[0][1]['draws']} draws)"
        sens_rows[key] = {"power": mean("power"), "jb_max": mx("jb_max"), "dolly": mean("dolly"), "wrap": mean("wrap"), "gain": mean("seq_gain")}
        tl.append(" & ".join([what, f2(mean("power")), f2(mean("coverage")), f2(mean("dolly")), f2(mean("mmlu")),
                              f"{f2(mean('jb'))} / {f2(mx('jb_max'))}", f2(mean("wrap")),
                              f2(float(np.mean(turn)) if turn else None), f2(mean("seq_gain"), signed=True)]) + r" \\")
        if key == "probe":
            tl.append(r"\midrule")
    tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(out, "sens_table.tex"), "w") as fh:
        fh.write("\n".join(tl) + "\n")
    two = {k: v for k, v in sens_rows.items() if k != "probe"}
    if two:
        nums["rvbAllSensN"] = str(len(two))
        nums["rvbAllSensJbMax"] = f2(max(v["jb_max"] for v in two.values() if v["jb_max"] is not None)) \
            if any(v["jb_max"] is not None for v in two.values()) else "--"
        pw = [v["power"] for v in two.values() if v["power"] is not None]
        nums["rvbAllSensPowerMin"], nums["rvbAllSensPowerMax"] = f2(min(pw)), f2(max(pw))
        if "probe" in sens_rows and sens_rows["probe"]["jb_max"] is not None:
            nums["rvbAllSensProbeJbMax"] = f2(sens_rows["probe"]["jb_max"])
        if "paper" in sens_rows:
            nums["rvbAllSensPaperPower"] = f2(sens_rows["paper"]["power"])
    # ---- budget
    tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\caption{A stream of $K$ stages sharing one budget $\varepsilon=0.03$, simulated on every real stage of "
          r"every experiment (" + "{n}" + r" stages): each stage's gate calibrated at $\varepsilon/K$ ($K$ known) or "
          r"$6\varepsilon/(\pi^2K^2)$ (stage $K$ of a stream of unknown length). Cert.: stages whose calibration sets "
          r"certify that level; Certified: the level they do certify per stage (median), and over the $K$ stages "
          r"($K$ times it, union bound, at most 100); $n$ needed: calibration prompts per reference component that would certify "
          r"the level (median; the smallest reference component has " + "{have}" + r"); Power: firing on held-out new-task prompts "
          r"(mean); Dolly: held-out firing (largest). Percent.}",
          r"\label{tab:budgetK}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{r l r r r r r r r}", r"\toprule",
          r"$K$ & Schedule & Level & Cert. & Certified & Stream & $n$ needed & Power & Dolly \\", r"\midrule"]
    stages = [s for r in R for s in r.get("budget", [])]
    have = sorted({x["n_have"] for s in stages for x in s["rows"]})
    tl[4] = tl[4].replace("{n}", str(len(stages))).replace(
        "{have}", f"{have[0]}--{have[-1]}" if have and have[0] != have[-1] else (str(have[0]) if have else "--"))
    for K in KS:
        for sched in ("split", "anytime"):
            xs = [x for s in stages for x in s["rows"] if x["K"] == K and x["sched"] == sched]
            if not xs or (K == 1 and sched == "anytime"):
                continue
            pw = [x["power"] for x in xs if x["power"] is not None]
            dl = [x["dolly"] for x in xs if x["dolly"] is not None]
            cert = float(np.median([x["eps_cert"] for x in xs]))
            tl.append(" & ".join([str(K), "$\\varepsilon/K$" if sched == "split" else "$6\\varepsilon/\\pi^2K^2$",
                                  f"{100 * xs[0]['eps_stage']:.3g}", f"{sum(x['feasible'] for x in xs)}/{len(xs)}",
                                  f"{100 * cert:.2f}", f"{100 * min(1.0, K * cert):.1f}" if sched == "split" else "--",
                                  str(int(np.median([x["n_need"] for x in xs]))), f2(float(np.mean(pw)) if pw else None),
                                  f2(max(dl) if dl else None)]) + r" \\")
            if sched == "split":
                nm = {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty"}[K]
                nums[f"rvbAllBud{nm}Cert"] = str(sum(x["feasible"] for x in xs))
                nums[f"rvbAllBud{nm}Need"] = str(int(np.median([x["n_need"] for x in xs])))
                nums[f"rvbAllBud{nm}Power"] = f2(float(np.mean(pw)) if pw else None)
                nums[f"rvbAllBud{nm}Stream"] = f"{100 * min(1.0, K * cert):.1f}"
    tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(out, "budget_table.tex"), "w") as fh:
        fh.write("\n".join(tl) + "\n")
    if stages:
        nums["rvbAllBudStages"] = str(len(stages))
        nums["rvbAllBudHaveMin"], nums["rvbAllBudHaveMax"] = str(min(have)), str(max(have))
    with open(os.path.join(out, "review2_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/review2_analyses.py --aggregate\n")
        for k in sorted(nums):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    json.dump({"experiments": [r["exp"] for r in R], "numbers": nums}, open(os.path.join(out, "aggregate.json"), "w"),
              indent=1)
    print(f"wrote {out}: {len(R)} experiments, {len(nums)} macros")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--aggregate", default="", help="results folder: pooled tables over every experiment's review2.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--out", default="", help="the switch report's folder (default paper/v2_results/<exp>/switch)")
    ap.add_argument("--name", default="", help="letters only: macro prefix \\rvb<name>...")
    ap.add_argument("--parts", default="sens,cov,budget")
    ap.add_argument("--alphas", default="0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--shrinks", default="0.01,0.1,0.3,1")
    ap.add_argument("--sizes", default="50,100,200,400")
    ap.add_argument("--sens-draws", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--random-gates", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.aggregate)
        return
    if not args.config:
        raise SystemExit("--config or --aggregate")
    if args.name and not args.name.isalpha():
        raise SystemExit("--name must be letters only")
    cfg = load_config(args.config, args.set)
    runs = args.runs or os.path.join(cfg["out_dir"], cfg["exp"])
    feat = args.features or os.path.join(runs, "analysis", "gate_features.npz")
    sw_dir = args.out or os.path.join("paper", "v2_results", cfg["exp"], "switch")
    sp = os.path.join(sw_dir, "switch_summary.json")
    if not (os.path.exists(feat) and os.path.exists(sp)):
        raise SystemExit(f"needs {feat} and {sp} (scripts/switch_report.py first)")
    out_dir = os.path.join(sw_dir, "review2")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summ = json.load(open(sp))
    ex = Experiment(cfg, feat, feat[:-4] + ".json")
    delta = float(ex.tc["delta"])
    eps = float(summ["main_eps"])
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    lens = {p: len(ex.X(f"eval__{p}")) for p in ex.eval_keys()}
    fams = run_arrays(finished_runs(runs, ex.K), ex.tasks, lens)
    res = {"exp": cfg["exp"], "label": readable_label(cfg), "name": args.name, "eps": eps, "delta": delta,
           "tasks": ex.tasks}
    nums, pre = {}, "rvb" + args.name
    fl = lambda s: [float(x) for x in s.split(",") if x.strip()]  # noqa: E731
    if "sens" in parts:
        res["sens"] = part_sens(ex, fams, eps, delta, fl(args.alphas), fl(args.shrinks),
                                [int(x) for x in fl(args.sizes)], args.sens_draws, rng)
        for x in res["sens"]:
            if x["setting"] in ("probe", "paper"):
                nums[f"{pre}Sens{x['setting'].capitalize()}Power"] = f2(x.get("power"))
                nums[f"{pre}Sens{x['setting'].capitalize()}JbMax"] = f2(x.get("jb_max"))
    if "cov" in parts and fams:
        res["cov"] = part_cov(ex, fams, eps, 0.02, args.boot, args.random_gates, args.folds, rng)
    if "budget" in parts:
        res["budget"] = part_budget(ex, eps, delta, KS)
    with open(os.path.join(out_dir, "review2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "review2_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/review2_analyses.py\n")
        for k in sorted(nums):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    print(f"wrote {out_dir}: parts {sorted(p for p in parts if p in res)}, {len(nums)} macros")


if __name__ == "__main__":
    main()
