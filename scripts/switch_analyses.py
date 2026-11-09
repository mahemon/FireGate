#!/usr/bin/env python
"""Analyses on stored features and outputs for the review points (CPU only: numpy + scipy; no GPU, no new runs).

    python scripts/switch_analyses.py --config configs/v2/overlap_q15.yaml --name Nq [--out paper/v2_results/<exp>/switch]

Needs <runs>/analysis/gate_features.npz and the switch report of the same experiment (scripts/switch_report.py
writes switch_summary.json into --out first). Writes into --out/analyses/:
  judged_points.dat     every method as trained: Dolly responses changed (exact string) and judged substantively
                        different, new-task gain (a judged version of the frontier figure; nan: not judged)
  sweep_<family>.dat    (single-task experiments) the deployed probe's threshold swept so that a share beta of the
                        unlabeled RL prompts fires (beta 0.1 ... 0.9, and the deployed threshold): predicted gain
                        beta * G against the measured gain behind the gate, with a paired bootstrap over evaluation and
                        RL prompts (95% intervals), the covariance term Cov(G, dr) = measured - beta_eval * G, and the
                        Dolly firing / change at that threshold. Keep-all is beta = 1 (gain G); a random gate at rate
                        beta keeps beta * G in expectation, so the prediction is the random-gate baseline.
  calib_check.tex       calibration against held-out firing, per stage and calibrated component that has an
                        evaluation pool: in-sample firing at the threshold, held-out firing, Fisher's exact test, and a
                        re-split test (calibration and held-out prompts pooled and split again R times, threshold
                        recomputed each time): the share of re-splits with held-out firing at least the observed one is
                        the p-value of exchangeability; also the 95th percentile of held-out firing over re-splits
  numerics.json         the numerics floor: the base model's greedy responses on two GPU models (stage 0 of runs that
                        ran on different GPU models), share that differ per pool, and the judged share (when
                        scripts/judge_changes.py labelled the _base_eval pair)
  analyses.json, analyses_numbers.tex   all numbers; macros \\an<name>...
Forfeited transfer: for every ungated model, its own change minus its change behind the gate (MMLU, XSTest, AdvBench,
new task), i.e. what the prompts on which the gate does not fire would have gained.
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
from scipy.stats import fisher_exact  # noqa: E402

from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402
from switch_report import (GATED, SUBSTANTIVE, Experiment, StageGate, fam_key,  # noqa: E402
                           fam_label, load_run, probe_kind, readable_label)

BETAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SWEEP_LETTERS = "ABCDEFGHI"   # macro suffix of beta 0.1 ... 0.9
LOW_EPS = (0.01, 0.02, 0.05)  # v6: calibrated sweep levels "eps0.01" ...


def beta_two_sided(x, n, conf=0.95):
    """two-sided Clopper-Pearson interval of a firing rate"""
    from scipy.stats import beta as _b
    a = (1 - conf) / 2
    return (float(_b.ppf(a, x, n - x + 1)) if x > 0 else 0.0, float(_b.ppf(1 - a, x + 1, n - x)) if x < n else 1.0)


def m0(a):
    return None if a is None else a[0]


def f2(v, scale=100.0, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{scale * v:+.1f}" if signed else f"{scale * v:.1f}"
    return s.replace("-", "$-$", 1) if signed else s


def pv(p):
    return "--" if p is None else (f"{p:.2f}" if p >= 0.01 else "$<$0.01")


# ----------------------------------------------------------------------------- runs
def finished_runs(runs, K):
    """[(seed dir, family, method, B rows, R rows, judge labels)] of every finished run"""
    out = []
    seeds = sorted((d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
    for sd in seeds:
        for fam in sorted(os.listdir(os.path.join(runs, sd))):
            md = os.path.join(runs, sd, fam)
            cj = os.path.join(md, "config.json")
            if not (os.path.isdir(md) and os.path.exists(os.path.join(md, "DONE")) and os.path.exists(cj)):
                continue
            loaded = load_run(md, K)
            if loaded is None:
                continue
            B, R = loaded
            jp = os.path.join(runs, "analysis", "judge", f"{sd}__{fam}.json")
            labels = json.load(open(jp)).get("labels") if os.path.exists(jp) else None
            out.append((sd, fam, json.load(open(cj)).get("method"), B, R, labels))
    return out


# ----------------------------------------------------------------------------- judged points
def judged_points(summ, gain_key):
    rows = []
    for fam in sorted(summ["families"], key=fam_key):
        F = summ["families"][fam]
        o = F["own"]
        c, s, g = o.get("dolly:chg"), o.get("dolly:subst"), o.get(gain_key)
        if c is None or g is None:
            continue
        rows.append({"family": fam, "label": fam_label(fam, F["method"]), "n": F["n"],
                     "chg": c[0], "chg_sd": c[1], "subst": None if s is None else s[0],
                     "subst_sd": None if s is None else s[1], "gain": g[0], "gain_sd": g[1],
                     "judged": m0(o.get("dolly:judged"))})
    return rows


# ----------------------------------------------------------------------------- coverage sweep
def coverage_sweep(ex, runs_list, task, n_boot, rng):
    """single-task experiments: the deployed probe's threshold at RL-prompt firing beta; per ungated family the
    predicted (beta * G) and measured gain behind the gate, with bootstrap intervals"""
    _, pos, neg, cal, _ = ex.stage_data(1, "ratio", None)
    probe, _ = ex.probe(1, "ratio", None, pos, neg)
    s_rl = probe.score(ex.X(f"rl__{task}"))
    s_ev = probe.score(ex.X(f"eval__{task}"))
    s_do = probe.score(ex.X("eval__dolly")) if ex.has("eval__dolly") else None
    dep = ex.deployed(1)
    levels = [(f"{b:g}", float(np.quantile(s_rl, 1.0 - b))) for b in BETAS]
    if dep is not None:
        levels.append(("deployed", dep[1]))
    # v6: PAC-calibrated thresholds below the paper's eps grid (the certified frontier at eps <= 0.05)
    cs = {c: probe.score(X).tolist() for c, X in cal.items()}
    for e in LOW_EPS:
        levels.append((f"eps{e:g}", float(calibrate(cs, e, float(ex.tc["delta"]))["kappa"])))
    by_fam = {}
    for sd, fam, method, B, R, labels in runs_list:
        if method in GATED or task not in R or task not in B or R[task] is None or B[task] is None:
            continue
        if len(R[task]) != len(s_ev):
            continue
        d = np.array([float(r["m"]["correct"]) - float(b["m"]["correct"]) for r, b in zip(R[task], B[task])])
        ch = sub = None
        if s_do is not None and R.get("dolly") and B.get("dolly") and len(R["dolly"]) == len(s_do):
            ch = np.array([r["out"] != b["out"] for r, b in zip(R["dolly"], B["dolly"])])
            lab = (labels or {}).get("dolly")
            if lab is not None:
                sub = np.array([(lab.get(str(r["i"])) in SUBSTANTIVE) if c else False
                                for r, c in zip(R["dolly"], ch)])
                known = [lab.get(str(r["i"])) is not None for r, c in zip(R["dolly"], ch) if c]
                if known and not all(known):
                    sub = None   # partly judged: no judged drift for this seed
        base = np.array([float(b["m"]["correct"]) for b in B[task]])   # the base model's own correctness
        by_fam.setdefault(fam, []).append((d, ch, sub, base))
    n_ev, n_rl = len(s_ev), len(s_rl)
    I_ev = rng.integers(0, n_ev, size=(n_boot, n_ev))
    I_rl = rng.integers(0, n_rl, size=(n_boot, n_rl))
    out = {}
    for fam, lst in by_fam.items():
        D = np.stack([x[0] for x in lst])                       # seeds x prompts
        G = D.mean(1)                                           # own gain per seed
        DB = D[:, I_ev]                                         # seeds x boot x prompts
        GB = DB.mean(2)                                         # seeds x boot
        rows = []
        for name, kap in levels:
            f_ev = s_ev > kap
            f_rl = s_rl > kap
            b_pre, b_ev = float(f_rl.mean()), float(f_ev.mean())
            meas = float((D * f_ev).mean(1).mean())
            pred = float((b_pre * G).mean())
            cov = float(((D * f_ev).mean(1) - b_ev * G).mean())
            mb = (DB * f_ev[I_ev][None]).mean(2).mean(0)        # boot
            pb = (f_rl[I_rl].mean(1)[None] * GB).mean(0)
            eb = mb - pb
            # pre-training diagnostic of the assumption behind beta * G: does the gate pick prompts the base model
            # already gets right (or wrong) more often? Cov(fire, base correctness) on evaluation prompts, no training
            cov_base = float(np.mean([np.mean(f_ev * x[3]) - f_ev.mean() * x[3].mean() for x in lst]))
            row = {"level": name, "kappa": kap, "beta_pre": b_pre, "beta_eval": b_ev, "pred": pred, "meas": meas,
                   "meas_lo": float(np.quantile(mb, 0.025)), "meas_hi": float(np.quantile(mb, 0.975)),
                   "err": meas - pred, "err_lo": float(np.quantile(eb, 0.025)), "err_hi": float(np.quantile(eb, 0.975)),
                   "cov": cov, "keep_all": float(G.mean()), "n_seeds": len(lst), "cov_base": cov_base,
                   "beta_lo": beta_two_sided(int(f_rl.sum()), len(f_rl))[0],
                   "beta_hi": beta_two_sided(int(f_rl.sum()), len(f_rl))[1]}
            if s_do is not None:
                f_do = s_do > kap
                row["dolly_fire"] = float(f_do.mean())
                chs = [x[1] for x in lst if x[1] is not None]
                row["dolly_chg"] = float(np.mean([(c & f_do).mean() for c in chs])) if chs else None
                subs = [x[2] for x in lst if x[2] is not None]
                row["dolly_subst"] = float(np.mean([(s & f_do).mean() for s in subs])) if len(subs) == len(lst) else None
            rows.append(row)
        out[fam] = rows
    return out


# ----------------------------------------------------------------------------- calibration vs held-out
def stage_calibration(ex, base, eps, k, protect, alpha):
    """per-component thresholds of stage k of a non-union gate, as build_variant computes them"""
    delta = float(ex.tc["delta"])
    task, pos, neg, cal, refs_cal = ex.stage_data(k, base, protect)
    probe, _ = ex.probe(k, base, protect, pos, neg)
    sup = ex.support(k, probe_kind(base), probe, pos, task, alpha) if base in ("twosided", "twosided_remedy") else None
    gate = StageGate(probe, sup, 0.0)
    cs = {c: gate.score(X) for c, X in cal.items()}
    n_all = len(cs)
    comp = {}
    for part in ({c: v for c, v in cs.items() if c in refs_cal}, {c: v for c, v in cs.items() if c not in refs_cal}):
        if not part:
            continue
        res = calibrate({c: v.tolist() for c, v in part.items()}, eps, delta * len(part) / n_all)
        for c, v in res["components"].items():
            comp[c] = {"kappa_c": float(v["kappa_c"]), "k": int(v["k"]), "n": int(v["n"]),
                       "d": float(res["delta_per_component"]), "eps_cert": float(v["eps_cert"])}
    gate.kappa = max(v["kappa_c"] for v in comp.values())
    return task, gate, cs, comp, sup


def calib_check(ex, base, eps, protect, alpha, n_resplit, rng, kappa_ref):
    rows, region = [], []
    for k in range(1, ex.K + 1):
        task, gate, cs, comp, sup = stage_calibration(ex, base, eps, k, protect, alpha)
        if kappa_ref is not None and abs(gate.kappa - kappa_ref[k]) > 1e-6 * max(1.0, abs(kappa_ref[k])):
            print(f"  warning: stage {k} {base}: threshold {gate.kappa} differs from the switch report's {kappa_ref[k]}")
        for c, sc in cs.items():
            pool = c[len("old_"):] if c.startswith("old_") else c
            if not ex.has(f"eval__{pool}"):
                continue
            se = gate.score(ex.X(f"eval__{pool}"))
            n_c, n_e = len(sc), len(se)
            x_c, x_e = int((sc > gate.kappa).sum()), int((se > gate.kappa).sum())
            _, p_f = fisher_exact([[x_c, n_c - x_c], [x_e, n_e - x_e]])
            other = max([v["kappa_c"] for cc, v in comp.items() if cc != c], default=-np.inf)
            kidx = comp[c]["k"]   # the order statistic depends on n_cal, eps and delta only: the same in a re-split
            pooled = np.concatenate([sc, se])
            P = np.stack([rng.permutation(len(pooled)) for _ in range(n_resplit)])
            calp, test = pooled[P[:, :n_c]], pooled[P[:, n_c:]]
            kc = np.partition(calp, kidx - 1, axis=1)[:, kidx - 1]
            kap = np.maximum(kc, other)
            tf = (test > kap[:, None]).mean(1)
            rows.append({"stage": k, "component": c, "pool": pool, "n_cal": n_c, "fire_cal": x_c / n_c, "n_eval": n_e,
                         "fire_eval": x_e / n_e, "cp_cal": clopper_pearson_upper(x_c, n_c), "eps": comp[c]["eps_cert"],
                         "p_fisher": float(p_f), "p_exch": float((tf >= x_e / n_e - 1e-12).mean()),
                         "resplit_mean": float(tf.mean()), "resplit_q95": float(np.quantile(tf, 0.95)),
                         "resplit_above_eps": float((tf > comp[c]["eps_cert"]).mean())})
        if sup is not None:
            region.append({"stage": k, "task": task, "rl_in": float(sup.inside(ex.X(f"rl__{task}")).mean()),
                           "fit_in": float(sup.inside(ex.X(f"fit__new__{task}")).mean()),
                           "eval_in": float(sup.inside(ex.X(f"eval__{task}")).mean())
                           if ex.has(f"eval__{task}") else None})
    return rows, region


# ----------------------------------------------------------------------------- numerics floor
def numerics_floor(runs, K):
    """stage-0 (base model) responses of runs on different GPU models: share that differ per pool"""
    by_dev = {}
    for sd in sorted(d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()):
        for fam in sorted(os.listdir(os.path.join(runs, sd))):
            edir = os.path.join(runs, sd, fam, "stage0", "eval")
            if not os.path.exists(os.path.join(edir, "DONE")):
                continue
            pools = {fn[:-6]: os.path.join(edir, fn) for fn in os.listdir(edir) if fn.endswith(".jsonl")}
            if not pools:
                continue
            with open(next(iter(pools.values()))) as fh:
                first = fh.readline()
            dev = json.loads(first).get("dev", "?") if first.strip() else "?"
            by_dev.setdefault(dev, (f"{sd}/{fam}", pools))
    devs = sorted(by_dev)
    pairs = []
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            (ra, pa), (rb, pb) = by_dev[devs[i]], by_dev[devs[j]]
            per = {}
            for p in sorted(set(pa) & set(pb)):
                a = [json.loads(x) for x in open(pa[p]) if x.strip()]
                b = [json.loads(x) for x in open(pb[p]) if x.strip()]
                a.sort(key=lambda r: r["i"])
                b.sort(key=lambda r: r["i"])
                if len(a) != len(b) or not a:
                    continue
                per[p] = {"n": len(a), "chg": float(np.mean([x["out"] != y["out"] for x, y in zip(a, b)]))}
            pairs.append({"devices": [devs[i], devs[j]], "runs": [ra, rb], "pools": per})
    out = {"devices": devs, "pairs": pairs}
    jd = os.path.join(runs, "analysis", "judge")
    if os.path.exists(os.path.join(jd, "_numerics.json")):
        lab = json.load(open(os.path.join(jd, "_numerics.json"))).get("labels", {})
        dv = json.load(open(os.path.join(jd, "_numerics_devices.json"))) \
            if os.path.exists(os.path.join(jd, "_numerics_devices.json")) else {}
        out["judged"] = {"devices": dv, "pools": {p: {"changed": len(v), "different": sum(x in SUBSTANTIVE for x in v.values())}
                                                  for p, v in lab.items()}}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--out", default="", help="the switch report's folder (default paper/v2_results/<exp>/switch)")
    ap.add_argument("--name", default="", help="letters only: macro prefix \\an<name>...")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--protect", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--resplits", type=int, default=2000)
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
    out_dir = os.path.join(sw_dir, "analyses")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summ = json.load(open(sp))
    ex = Experiment(cfg, feat, feat[:-4] + ".json")
    tasks, stream = ex.tasks, ex.K > 1
    gain_key = "task:gain" if stream else f"{tasks[0]}:correct"
    eps = float(summ["main_eps"])
    protect = [c.strip() for c in args.protect.split(",") if c.strip()]
    res = {"exp": cfg["exp"], "label": readable_label(cfg), "eps": eps}
    nums, pre = {}, "an" + args.name

    # judged points
    pts = judged_points(summ, gain_key)
    res["judged_points"] = pts
    with open(os.path.join(out_dir, "judged_points.dat"), "w") as fh:
        fh.write("chg subst gain chg_sd subst_sd gain_sd family\n")
        for r in pts:
            fh.write(f"{100 * r['chg']:.2f} " + (f"{100 * r['subst']:.2f}" if r["subst"] is not None else "nan")
                     + f" {100 * r['gain']:.2f} {100 * r['chg_sd']:.2f} "
                     + (f"{100 * r['subst_sd']:.2f}" if r["subst_sd"] is not None else "nan")
                     + f" {100 * r['gain_sd']:.2f} {r['family']}\n")
    seq = next((r for r in pts if r["family"] == "seq"), None)
    if seq is not None and seq["subst"] is not None:
        nums[pre + "SeqChg"], nums[pre + "SeqSubst"] = f2(seq["chg"]), f2(seq["subst"])
        nums[pre + "SeqSubstShare"] = f2(seq["subst"] / seq["chg"] if seq["chg"] else None)

    runs_list = finished_runs(runs, ex.K)

    # coverage sweep (single-task experiments)
    if not stream and ex.has(f"eval__{tasks[0]}"):
        sw = coverage_sweep(ex, runs_list, tasks[0], args.boot, rng)
        res["sweep"] = sw
        for fam, rows in sw.items():
            with open(os.path.join(out_dir, f"sweep_{fam}.dat"), "w") as fh:
                fh.write("beta_pre beta_eval pred meas meas_lo meas_hi err err_lo err_hi cov dolly_fire dolly_chg "
                         "dolly_subst cov_base beta_lo beta_hi level\n")
                for r in rows:
                    vals = [r["beta_pre"], r["beta_eval"], r["pred"], r["meas"], r["meas_lo"], r["meas_hi"], r["err"],
                            r["err_lo"], r["err_hi"], r["cov"], r.get("dolly_fire"), r.get("dolly_chg"),
                            r.get("dolly_subst"), r["cov_base"], r["beta_lo"], r["beta_hi"]]
                    fh.write(" ".join("nan" if v is None else f"{100 * v:.2f}" for v in vals) + f" {r['level']}\n")
        if "seq" in sw:
            grid = [r for r in sw["seq"] if r["level"] != "deployed" and not r["level"].startswith("eps")]
            errs = [abs(r["err"]) for r in grid]
            nums[pre + "SweepMae"] = f2(sum(errs) / len(errs))
            nums[pre + "SweepMax"] = f2(max(errs))
            nums[pre + "SweepCover"] = str(sum(r["err_lo"] <= 0 <= r["err_hi"] for r in grid))
            nums[pre + "SweepLevels"] = str(len(grid))
            nums[pre + "SweepCovMax"] = f2(max(abs(r["cov"]) for r in grid))
            nums[pre + "SweepKeepAll"] = f2(grid[0]["keep_all"], signed=True)
            nums[pre + "SweepCovBaseMax"] = f2(max(abs(r["cov_base"]) for r in grid))
            if len(grid) > 2:   # does the pre-training diagnostic track the realized covariance term?
                cb, cv = np.array([r["cov_base"] for r in grid]), np.array([r["cov"] for r in grid])
                if cb.std() > 0 and cv.std() > 0:
                    nums[pre + "SweepCovCorr"] = f"{float(np.corrcoef(cb, cv)[0, 1]):.2f}"
            for r in sw["seq"]:
                if r["level"].startswith("eps"):
                    L = {"eps0.01": "A", "eps0.02": "B", "eps0.05": "C"}.get(r["level"])
                    if L:
                        nums[pre + "LowGain" + L] = f2(r["meas"], signed=True)
                        nums[pre + "LowPower" + L] = f2(r["beta_pre"])
                        nums[pre + "LowDolly" + L] = f2(r.get("dolly_fire"))
            for i, r in enumerate(grid):
                L = SWEEP_LETTERS[i] if i < len(SWEEP_LETTERS) else None
                if L is None:
                    break
                nums[pre + "SweepPred" + L] = f2(r["pred"], signed=True)
                nums[pre + "SweepMeas" + L] = f2(r["meas"], signed=True)
                nums[pre + "SweepDolly" + L] = f2(r.get("dolly_fire"))
            lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
                     r"\caption{Coverage sweep (" + res["label"].replace("_", r"\_") + r"): the deployed probe's "
                     r"threshold set so that a share $\beta$ of unlabeled RL prompts fires. Pred.: $\beta G$ (the "
                     r"gain kept by a random gate at the same rate); Meas.: Seq-GRPO's gain behind the gate with a "
                     r"95\% paired bootstrap interval over evaluation and RL prompts; Cov.: $\mathrm{Cov}(G,\Delta r)$ "
                     r"on evaluation prompts; Dolly: firing on held-out Dolly prompts (the drift bound at that "
                     r"threshold, not calibrated). Keep-all ($\beta=1$): " + f2(grid[0]["keep_all"], signed=True)
                     + r" points; mean over " + str(grid[0]["n_seeds"]) + r" seeds.}",
                     r"\label{tab:sweep_" + (args.name.lower() or "x") + "}",
                     r"\begin{tabular}{l rr r r r r}", r"\toprule",
                     r"$\beta$ & $\beta_{\mathrm{RL}}$ & $\beta_{\mathrm{eval}}$ & Pred. & Meas. & Cov. & Dolly \\",
                     r"\midrule"]
            for r in sw["seq"]:
                lv = (r"$\varepsilon=" + r["level"][3:] + "$" if r["level"].startswith("eps") else r["level"])
                lines.append(" & ".join([lv, f2(r["beta_pre"]), f2(r["beta_eval"]), f2(r["pred"], signed=True),
                                         f2(r["meas"], signed=True) + r" {\scriptsize[" + f2(r["meas_lo"], signed=True)
                                         + ", " + f2(r["meas_hi"], signed=True) + "]}", f2(r["cov"], signed=True),
                                         f2(r.get("dolly_fire"))]) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            with open(os.path.join(out_dir, "sweep_table.tex"), "w") as fh:
                fh.write("\n".join(lines) + "\n")

    # calibration vs held-out firing (+ region coverage on held-out new-task prompts)
    res["calib"] = {}
    tl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
          r"\caption{Calibration against held-out firing (" + res["label"].replace("_", r"\_") + r", $\varepsilon="
          + f"{eps:g}" + r"$). Cal.: in-sample firing on the calibration prompts at the threshold; Held-out: firing "
          r"on the evaluation prompts of the same pool; $p_F$: Fisher's exact test; re-split: calibration and "
          r"held-out prompts pooled and split again " + str(args.resplits) + r" times with the threshold recomputed, "
          r"95th percentile of held-out firing and $p$-value of the observed one (exchangeability).}",
          r"\label{tab:calib_" + (args.name.lower() or "x") + "}", r"\begin{tabular}{l l r r r r r r r}", r"\toprule",
          r"Gate & Pool & $k$ & Cal. & CP & Held-out & $p_F$ & q95 & $p$ \\", r"\midrule"]
    pmin = []
    variants = ["ratio", "twosided"] + (["twosided_remedy"] if f"twosided_remedy|{eps}" in summ["gates"] else [])
    for v in variants:
        key = f"{v}|{eps}"
        kref = {int(k): x for k, x in summ["gates"][key]["kappa"].items()} if key in summ["gates"] else None
        rows, region = calib_check(ex, v, eps, protect, args.alpha, args.resplits, rng, kref)
        res["calib"][v] = {"rows": rows, "region": region}
        for r in rows:
            tl.append(" & ".join([v.replace("_", r"\_"), r["pool"].replace("_", r"\_"), str(r["stage"]),
                                  f"{r['fire_cal'] * r['n_cal']:.0f}/{r['n_cal']}", f2(r["cp_cal"]),
                                  f"{r['fire_eval'] * r['n_eval']:.0f}/{r['n_eval']}", pv(r["p_fisher"]),
                                  f2(r["resplit_q95"]), pv(r["p_exch"])]) + r" \\")
            pmin.append(r["p_exch"])
        Vn = "".join(w.capitalize() for w in v.split("_"))
        for r in rows:
            if r["stage"] == 1 and r["pool"] in ("dolly", "advbench"):
                P = r["pool"].capitalize()
                nums[f"{pre}{Vn}Cal{P}"] = f2(r["fire_cal"])
                nums[f"{pre}{Vn}Held{P}"] = f2(r["fire_eval"])
                nums[f"{pre}{Vn}CalN{P}"] = f"{r['fire_cal'] * r['n_cal']:.0f}/{r['n_cal']}"
                nums[f"{pre}{Vn}HeldN{P}"] = f"{r['fire_eval'] * r['n_eval']:.0f}/{r['n_eval']}"
                nums[f"{pre}{Vn}Pexch{P}"] = pv(r["p_exch"])
                nums[f"{pre}{Vn}Pfisher{P}"] = pv(r["p_fisher"])
                nums[f"{pre}{Vn}Qnf{P}"] = f2(r["resplit_q95"])
        for g in region:
            if g["stage"] == 1:
                nums[f"{pre}{Vn}RegionRl"] = f2(g["rl_in"])
                nums[f"{pre}{Vn}RegionFit"] = f2(g["fit_in"])
                if g["eval_in"] is not None:
                    nums[f"{pre}{Vn}RegionEval"] = f2(g["eval_in"])
    tl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, "calib_check.tex"), "w") as fh:
        fh.write("\n".join(tl) + "\n")
    if pmin:
        nums[pre + "ExchMin"] = pv(min(pmin))
        nums[pre + "ExchN"] = str(len(pmin))
        nums[pre + "ExchBelow"] = str(sum(p < 0.05 for p in pmin))

    # numerics floor
    nf = numerics_floor(runs, ex.K)
    res["numerics"] = nf
    with open(os.path.join(out_dir, "numerics.json"), "w") as fh:
        json.dump(nf, fh, indent=1)
    dch = [pr["pools"]["dolly"]["chg"] for pr in nf["pairs"] if "dolly" in pr["pools"]]
    if dch:
        nums[pre + "NumChgDolly"] = f2(max(dch))
        nums[pre + "NumPairs"] = str(len(dch))
    if nf.get("judged", {}).get("pools", {}).get("dolly"):
        jd = nf["judged"]["pools"]["dolly"]
        nums[pre + "NumDiffDolly"] = str(jd["different"])
        nums[pre + "NumChgdDolly"] = str(jd["changed"])

    # forfeited transfer: own change minus the change behind the deployed gate
    ff = {}
    for fam, F in summ["families"].items():
        if F["method"] in GATED:
            continue
        s = F["switch"].get(f"ratio|{eps}", {})
        ff[fam] = {k: (m0(F["own"].get(k)) - m0(s.get(k))) if F["own"].get(k) and s.get(k) else None
                   for k in (gain_key, "mmlu:correct", "xstest:refused_safe", "advbench:refused")}
    res["forfeit"] = ff
    if "seq" in ff:
        for k, mn in ((gain_key, "Gain"), ("mmlu:correct", "Mmlu"), ("xstest:refused_safe", "Xs"),
                      ("advbench:refused", "Adv")):
            if ff["seq"].get(k) is not None:
                nums[f"{pre}SeqForfeit{mn}"] = f2(ff["seq"][k], signed=True)

    with open(os.path.join(out_dir, "analyses.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "analyses_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/switch_analyses.py\n")
        for k in sorted(nums):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    print(f"wrote {out_dir}: {len(pts)} judged points, sweep of {len(res.get('sweep', {}))} families, "
          f"{sum(len(v['rows']) for v in res['calib'].values())} calibration checks, "
          f"{len(nf['pairs'])} GPU-model pairs, {len(nums)} macros")


if __name__ == "__main__":
    main()
