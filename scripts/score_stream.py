#!/usr/bin/env python
"""Step p3.18 (Nov 4, PLAN_PAPER3.md): the four score classes of scripts/score_classes.py on the task streams (CPU,
stored features, no training).

    python scripts/score_stream.py --config configs/v2/stream_q15.yaml --config configs/v2/stream5_q15.yaml \
        --out paper/v3_results/p3_18

For every stage k of a stream, each score (linear probe, Gaussian ratio, 15-NN similarity, MLP; the settings of
score_classes.py, fixed before p3.10) is fitted on the stage's fitting prompts (the new task against the reference
components and the earlier tasks) and PAC-calibrated as the deployed stage gate is (scripts/switch_report.py,
build_variant "ratio": every reference component and every earlier task at eps, delta shared over the components),
with delta divided by four so that the four certificates hold together. Reported per stage and score: the coverage
beta_k of the stage's unlabeled RL prompts (each score's beta_k is a lower bound on the limit beta*_k of Theorem 4.1
for that stage), and the firing of the union of the stage gates of that score on every held-out pool.
Outputs in --out: score_stream.json, score_stream_table.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import score_classes as sc  # noqa: E402
import switch_report as sr  # noqa: E402
from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402

SCORES = ("probe", "lda", "knn", "mlp")
NAMES = {"probe": "linear classifier", "lda": "Gaussian ratio", "knn": "15-NN similarity", "mlp": "MLP"}
POOLS = ("dolly", "advbench", "mmlu", "xstest", "wildchat")


def stage_kappa(cs, refs_cal, eps, delta):
    """the threshold of build_variant('ratio') for one stage: references and earlier tasks at eps, delta shared"""
    n_all = len(cs)
    refs = {c: v for c, v in cs.items() if c in refs_cal}
    olds = {c: v for c, v in cs.items() if c not in refs_cal}
    kappa = -np.inf
    for part in (refs, olds):
        if part:
            r = calibrate({c: v.tolist() for c, v in part.items()}, eps, delta * len(part) / n_all)
            kappa = max(kappa, float(r["kappa"]))
    return kappa


def run(cfg_path, eps, runs_root, want):
    cfg = load_config(cfg_path, [])
    runs = os.path.join(runs_root or cfg["out_dir"], cfg["exp"])
    feat = os.path.join(runs, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        print(f"skip {cfg['exp']}: {feat} is missing")
        return None
    ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
    if ex.K < 2:
        print(f"skip {cfg['exp']}: not a stream")
        return None
    delta = float(ex.tc["delta"]) / len(want)
    fitters = {"probe": lambda p, n: sr.fit_probe(p, n, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"])).score,
               "lda": sc.fit_lda, "knn": sc.fit_knn, "mlp": sc.fit_mlp}
    pools = [p for p in POOLS if ex.has(f"eval__{p}")]
    res = {"exp": cfg["exp"], "label": sr.readable_label(cfg), "tasks": ex.tasks, "eps": eps, "delta_each": delta,
           "scores": {}}
    for name in want:
        stages, fire_any = [], {p: None for p in pools}
        for k in range(1, ex.K + 1):
            task, pos, neg, cal, refs_cal = ex.stage_data(k, "ratio", [])
            score = fitters[name](pos, neg)
            cs = {c: np.asarray(score(X), np.float64) for c, X in cal.items()}
            kappa = stage_kappa(cs, refs_cal, eps, delta)
            beta = float((np.asarray(score(ex.X(f"rl__{task}")), np.float64) > kappa).mean())
            fires = {p: np.asarray(score(ex.X(f"eval__{p}")), np.float64) > kappa for p in pools}
            for p, f in fires.items():
                fire_any[p] = f if fire_any[p] is None else (fire_any[p] | f)
            stages.append({"task": task, "kappa": kappa, "beta": beta,
                           "fire": {p: float(f.mean()) for p, f in fires.items()}})
            print(f"{cfg['exp']} {name} stage {k} ({task}): beta {100 * beta:.1f}%  "
                  + "  ".join(f"{p} {100 * stages[-1]['fire'][p]:.1f}%" for p in pools), flush=True)
        res["scores"][name] = {"stages": stages,
                               "union": {p: float(f.mean()) for p, f in fire_any.items()},
                               "union_cp": {p: clopper_pearson_upper(int(f.sum()), len(f)) for p, f in fire_any.items()}}
    return res


def write_table(res, out):
    tasks_max = max(len(r["tasks"]) for r in res)
    cols = ["dolly", "advbench", "mmlu"]
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\caption{Score classes on the task streams, each stage calibrated as the deployed stage "
             r"gate at $\varepsilon=0.03$ and $\delta/4$, from stored features. $\beta_k$: coverage of stage $k$'s "
             r"unlabeled RL prompts (\%), a lower bound on that stage's limit; union: firing of the stage gates' union "
             r"on held-out prompts (\%).}",
             r"\label{tab:scorestream}",
             r"\begin{tabular}{l l " + "r" * tasks_max + " " + "r" * len(cols) + "}", r"\toprule",
             r" & & \multicolumn{" + str(tasks_max) + r"}{c}{$\beta_k$} & \multicolumn{" + str(len(cols))
             + r"}{c}{union fires} \\",
             "Stream & Score & " + " & ".join(str(k) for k in range(1, tasks_max + 1)) + " & Dolly & AdvBench & MMLU \\\\",
             r"\midrule"]
    for r in res:
        first = r["label"].replace("_", r"\_")
        for name, s in r["scores"].items():
            b = [f"{100 * st['beta']:.1f}" for st in s["stages"]] + ["--"] * (tasks_max - len(s["stages"]))
            u = [("--" if c not in s["union"] else f"{100 * s['union'][c]:.1f}") for c in cols]
            lines.append(f"{first} & {NAMES[name]} & " + " & ".join(b + u) + r" \\")
            first = ""
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "score_stream_table.tex"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--scores", default=",".join(SCORES))
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_18"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    want = [s.strip() for s in args.scores.split(",") if s.strip()]
    res = [r for r in (run(c, args.eps, args.runs_root, want) for c in args.config) if r is not None]
    json.dump(res, open(os.path.join(args.out, "score_stream.json"), "w"), indent=1)
    if res:
        write_table(res, args.out)
    print(f"wrote {args.out}/score_stream.json" + (", score_stream_table.tex" if res else ""))


if __name__ == "__main__":
    main()
