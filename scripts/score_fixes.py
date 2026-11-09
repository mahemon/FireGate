#!/usr/bin/env python
"""Step p3.11 (Nov 2, PLAN_PAPER3.md): the fixes for the weak cells of p3.10 (CPU, stored features, no training).

    python scripts/score_fixes.py --exp configs/v2/overlap_q15.yaml:Nq: --out paper/v3_results/p3_11
    python scripts/score_fixes.py --exp configs/v2/ladder_concisectx_q15.yaml:Concisectx:advbench,general \
        --out paper/v3_results/p3_11
    python scripts/score_fixes.py --merge --out paper/v3_results/p3_11          # tables from every <exp>/fixes.json

One experiment per process (memory); --merge writes the tables. For one single-task experiment, every gate below is
built from the four scores of scripts/score_classes.py (linear probe, Gaussian ratio, 15-NN similarity, MLP; same
settings, fixed before any result), on each reference, with and without the new task's region:
  references  deployed (every reference component); remedy (deployed + the harmful-request component of the task's
              forms or passages, step p3.8; where stored); protect_remedy (the protected components + that component;
              with --exp ...:<protected components>)
  region      off, or on: the score counts only inside the two-sided region of scripts/switch_report.py (shrunk
              Mahalanobis distance to the task's fitting prompts, in the linear probe's standardized space, below the
              (1 - alpha) quantile of unlabeled RL prompts), calibrated on the conjunction
Fixes it answers:
  1  choose the score from coverage alone: every score is calibrated at delta/4 (Bonferroni over the four), so all four
     certificates hold together with probability 1 - delta and the score with the largest coverage of unlabeled RL
     prompts (no behaviour labels) can be chosen after seeing it. 'chosen' marks it per reference and region.
  2  the two-sided gate with the remedy on Concise-ctx (and Concise, SQuAD 2.0): the passage component of p3.8 added
     to the two-sided gate, read on the eight second-round forms (step p3.10b).
  3  scores that extrapolate off the reference (the MLP on MMLU, NQ) combined with the region.
Reported per gate: coverage beta, firing and 95% Clopper-Pearson bound on every held-out pool, the largest firing over
the first-round (st_*) and second-round (st2_*) harmful forms, and Seq-GRPO's gain and refusal behind the gate (every
seed, stored responses). Needs <runs>/analysis/gate_features.npz (with eval__st2_*, step p3.10).
Outputs: <out>/<exp>/fixes.json; with --merge <out>/fixes.json, fixes_table.tex, fixes_summary.txt.
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
NAMES = {"probe": "linear probe", "lda": "Gaussian ratio (LDA)", "knn": "15-NN similarity", "mlp": "MLP"}
REF_NAMES = {"deployed": "deployed", "remedy": "deployed + remedy", "protect_remedy": "protect + passages"}
POOLS = ("advbench", "st_ctx", "st_ctx_question", "mmlu", "dolly", "xstest", "wildchat")


def seq_runs_of(runs):
    out = []
    if not os.path.isdir(runs):
        return out
    for sd in sorted((d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:])):
        md = os.path.join(runs, sd, "seq")
        if os.path.exists(os.path.join(md, "DONE")):
            loaded = sr.load_run(md, 1)
            if loaded is not None:
                out.append((sd, loaded))
    return out


def run_exp(spec, eps, alpha, runs_root, out):
    cfg_path, name, prot = (spec.split(":") + ["", ""])[:3]
    cfg = load_config(cfg_path, [])
    runs = os.path.join(runs_root or cfg["out_dir"], cfg["exp"])
    feat = os.path.join(runs, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        print(f"skip {cfg['exp']}: {feat} is missing")
        return
    ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
    if ex.K != 1:
        print(f"skip {cfg['exp']}: single-task experiments only")
        return
    protect = [c.strip() for c in prot.split(",") if c.strip()]
    wraps = any(ex.has(f"wrap__{t}__fit") for t in ex.tasks)
    refs = ["deployed"] + (["remedy"] if wraps else []) + (["protect_remedy"] if protect and wraps else [])
    delta = float(ex.tc["delta"])
    d_each = delta / len(SCORES)
    keys = ex.eval_keys()
    first = [p for p in keys if p.startswith("st_")]
    second = [p for p in keys if p.startswith("st2_")]
    seq = seq_runs_of(runs)
    fitters = {"probe": lambda p, n: sr.fit_probe(p, n, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"])).score,
               "lda": sc.fit_lda, "knn": sc.fit_knn, "mlp": sc.fit_mlp}
    res = {"exp": cfg["exp"], "task": ex.tasks[0], "name": name, "label": sr.readable_label(cfg), "protect": protect, "eps": eps,
           "alpha": alpha, "delta": delta, "delta_each": d_each, "seq_seeds": [s for s, _ in seq], "gates": []}
    task = ex.tasks[0]
    own = [sr.metrics(R, B, ex.tasks) for _, (B, R) in seq]
    res["seq_own"] = {k: sr.agg([o.get(k) for o in own]) for k in (f"{task}:correct", "advbench:refused")}
    for ref in refs:
        base = {"deployed": "ratio", "remedy": "remedy", "protect_remedy": "protect_remedy"}[ref]
        task, pos, neg, cal, _ = ex.stage_data(1, base, protect)
        lin = sr.fit_probe(pos, neg, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"]))
        sup = sr.Support(lin, pos, ex.X(f"rl__{task}"), alpha)
        inside = {k: sup.inside(ex.X(f"eval__{k}")) for k in keys}
        inside_rl = sup.inside(ex.X(f"rl__{task}"))
        inside_cal = {c: sup.inside(X) for c, X in cal.items()}
        for sname in SCORES:
            score = lin.score if sname == "probe" else fitters[sname](pos, neg)
            raw = {k: np.asarray(score(ex.X(f"eval__{k}")), np.float64) for k in keys}
            raw_rl = np.asarray(score(ex.X(f"rl__{task}")), np.float64)
            raw_cal = {c: np.asarray(score(X), np.float64) for c, X in cal.items()}
            for region in (False, True):
                mask = (lambda s, m: np.where(m, s, -np.inf)) if region else (lambda s, m: s)
                cs = {c: mask(v, inside_cal[c]) for c, v in raw_cal.items()}
                cr = calibrate({c: v.tolist() for c, v in cs.items()}, eps, d_each)
                kappa = float(cr["kappa"])
                fire = {k: mask(raw[k], inside[k]) > kappa for k in keys}
                beta = float((mask(raw_rl, inside_rl) > kappa).mean())
                sw = [sr.metrics(R, B, ex.tasks, None, fire) for _, (B, R) in seq]
                g = {"ref": ref, "region": region, "score": sname, "kappa": kappa, "beta": beta,
                     "eps_cert": {c: v["eps_cert"] for c, v in cr["components"].items()},
                     "fire": {k: float(f.mean()) for k, f in fire.items()},
                     "cp": {k: clopper_pearson_upper(int(f.sum()), len(f)) for k, f in fire.items()},
                     "max_st": max([float(fire[k].mean()) for k in first], default=None),
                     "max_st2": max([float(fire[k].mean()) for k in second], default=None),
                     "argmax_st2": max(second, key=lambda k: fire[k].mean()) if second else None,
                     "seq_behind": {k: sr.agg([m.get(k) for m in sw])
                                    for k in (f"{task}:correct", "advbench:refused", "st_ctx:refused",
                                              "st_ctx_question:refused")}}
                res["gates"].append(g)
                print(f"{cfg['exp']} {ref:15s} region={int(region)} {sname:5s} beta {100 * beta:5.1f}%  "
                      + "  ".join(f"{p} {100 * g['fire'][p]:.1f}%" for p in POOLS if p in g["fire"])
                      + f"  max st2 {pct(g['max_st2'])}  gain {gain(g, task)}", flush=True)
        for region in (False, True):
            grp = [g for g in res["gates"] if g["ref"] == ref and g["region"] == region]
            best = max(grp, key=lambda g: g["beta"])
            for g in grp:
                g["chosen"] = g is best
    os.makedirs(os.path.join(out, cfg["exp"]), exist_ok=True)
    json.dump(res, open(os.path.join(out, cfg["exp"], "fixes.json"), "w"), indent=1)
    print(f"wrote {out}/{cfg['exp']}/fixes.json")


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def gain(g, task):
    a = g["seq_behind"].get(f"{task}:correct")
    return "--" if a is None or a[0] is None else f"{100 * a[0]:+.1f}"


LABELS = {"v2_ladder_concise_q15": "Concise", "v2_ladder_concisectx_q15": "Concise-ctx", "v3_squad2_q15": "SQuAD~2.0",
          "v2_overlap_q15": "NQ"}


def merge(out):
    res = []
    for d in sorted(os.listdir(out)):
        f = os.path.join(out, d, "fixes.json")
        if os.path.exists(f):
            res.append(json.load(open(f)))
    json.dump(res, open(os.path.join(out, "fixes.json"), "w"), indent=1)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{Every score with and without the new task's region, each calibrated at $\varepsilon=0.03$ and "
             r"$\delta/4$, so that the score with the largest coverage of unlabeled RL prompts ($\ast$) can be chosen "
             r"after seeing it (Qwen2.5-1.5B, from stored features, no training). Firing (\%) on held-out prompts; "
             r"Harm.+pass.: the larger of the two passage forms; St.~1, St.~2: the largest firing over the first- and "
             r"second-round harmful forms; Gain: points of Seq-GRPO's gain that the switch keeps behind the gate.}",
             r"\label{tab:fixes}", r"\begin{tabular}{l l c l r rrrrrrr r}", r"\toprule",
             r"Experiment & Reference & Region & Score & $\beta\uparrow$ & AdvBench & Harm.+pass. & MMLU & Dolly & "
             r"XSTest & St.~1 & St.~2 & Gain$\uparrow$ \\", r"\midrule"]
    summ = []
    for r in res:
        task = r["task"]
        first = True
        for ref in ("deployed", "remedy", "protect_remedy"):
            for region in (False, True):
                grp = [g for g in r["gates"] if g["ref"] == ref and g["region"] == region]
                for g in grp:
                    f = g["fire"]
                    pas = max([f[k] for k in ("st_ctx", "st_ctx_question") if k in f], default=None)
                    star = r"$^\ast$" if g.get("chosen") else ""
                    lab = LABELS.get(r["exp"], r["exp"].replace("_", r"\_")) if first else ""
                    lines.append(f"{lab} & {REF_NAMES[ref]} & {'yes' if region else 'no'} & "
                                 f"{NAMES[g['score']]}{star} & {pct(g['beta'])} & "
                                 f"{pct(f.get('advbench'))} & {pct(pas)} & {pct(f.get('mmlu'))} & {pct(f.get('dolly'))} & "
                                 f"{pct(f.get('xstest'))} & {pct(g['max_st'])} & {pct(g['max_st2'])} & {gain(g, task)} \\\\")
                    first = False
                    if g.get("chosen"):
                        summ.append(f"{r['exp']:26s} {REF_NAMES[ref]:18s} region={'yes' if region else 'no '} chosen "
                                    f"{g['score']:5s} beta {pct(g['beta']):>5s}  advbench {pct(f.get('advbench'))}  "
                                    f"harm+pass {pct(pas)}  mmlu {pct(f.get('mmlu'))}  dolly {pct(f.get('dolly'))}  "
                                    f"xstest {pct(f.get('xstest'))}  max st {pct(g['max_st'])}  max st2 "
                                    f"{pct(g['max_st2'])} ({g['argmax_st2']})  gain {gain(g, task)}")
                if grp:
                    lines.append(r"\cmidrule(lr){2-13}")
        if lines[-1].startswith(r"\cmidrule"):
            lines[-1] = r"\midrule"
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "fixes_table.tex"), "w").write("\n".join(lines) + "\n")
    # the three cells the fixes answer
    sel = []
    for r in res:
        for g in r["gates"]:
            if (r["exp"] == "v2_overlap_q15" and g["score"] == "mlp" and g["ref"] == "deployed") or \
               (r["exp"] in ("v2_ladder_concisectx_q15", "v2_ladder_concise_q15", "v3_squad2_q15") and g["score"] == "probe"
                and g["ref"] in ("deployed", "remedy") and g["region"]):
                sel.append(f"{r['exp']:26s} {REF_NAMES[g['ref']]:18s} region={'yes' if g['region'] else 'no '} "
                           f"{g['score']:5s} beta {pct(g['beta']):>5s}  mmlu {pct(g['fire'].get('mmlu'))}  "
                           f"3 pass. {pct(g['fire'].get('st2_long_ctx'))}  max st2 {pct(g['max_st2'])}  "
                           f"gain {gain(g, r['task'])}")
    txt = "chosen score per reference and region (fix 1)\n" + "\n".join(summ) + \
          "\n\nfix 2 (probe, two-sided, with and without the remedy) and fix 3 (MLP on NQ, with and without the region)\n" + \
          "\n".join(sel) + "\n"
    open(os.path.join(out, "fixes_summary.txt"), "w").write(txt)
    print(txt + f"wrote {out}/fixes.json, fixes_table.tex, fixes_summary.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", default=[], help="config:name:protected components")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_11"))
    ap.add_argument("--runs-root", default="", help="folder holding <exp>/analysis/ (default: the config's out_dir)")
    ap.add_argument("--merge", action="store_true", help="write the tables from every <out>/<exp>/fixes.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for spec in args.exp:
        run_exp(spec, args.eps, args.alpha, args.runs_root, args.out)
    if args.merge:
        merge(args.out)


if __name__ == "__main__":
    main()
