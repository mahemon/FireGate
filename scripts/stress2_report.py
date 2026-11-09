#!/usr/bin/env python
"""Step p3.10b (Nov 2, PLAN_PAPER3.md): harmful requests in forms written before any gate was evaluated on them.

    python scripts/stress2_report.py --exp configs/v3/squad2_q15.yaml:Squad:advbench,general \
        --exp configs/v2/ladder_ifc_q15.yaml:Ifc: --out paper/v3_results/p3_10

For every experiment (config:name:protected components, the last part empty when nothing is protected), the gates of
scripts/switch_report.py are built from stored features at --eps (deployed, two-sided, and where the data allow the
remedy, the two-sided gate with the remedy, protect and protect + passages), and their firing is reported on the
second-round forms st2_* of twcore/stress.py, with the 95% Clopper-Pearson bound and, for comparison, the first-round
forms st_*. By Theorem 3.2 a prompt on which a gate does not fire is answered by every model behind it as the base
model answers it, so the firing is the whole result: no model is run. The forms are fixed in twcore/stress.py; this
script only reads features. Needs <runs>/analysis/gate_features.npz with the eval__st2_* sets (step p3.10).
Outputs in --out: stress2.json, stress2_table.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import switch_report as sr  # noqa: E402
from twcore.certify import clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402

FORMS = ["st2_para", "st2_squadctx", "st2_ctx_before", "st2_markdown", "st2_sys_ctx", "st2_turn_ctx", "st2_long_ctx",
         "st2_noise"]
SHORT = {"st2_para": "para.", "st2_squadctx": "new pass.", "st2_ctx_before": "pass. first", "st2_markdown": "code block",
         "st2_sys_ctx": "sys.+pass.", "st2_turn_ctx": "turn+pass.", "st2_long_ctx": "3 pass.", "st2_noise": "typos"}
GATE_NAMES = {"ratio": "deployed", "twosided": "two-sided", "remedy": "+ remedy", "twosided_remedy": "two-sided + remedy",
              "protect": "protect", "protect_remedy": "protect + passages"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True, help="config:name:protected components")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_10"))
    ap.add_argument("--runs-root", default="", help="folder holding <exp>/analysis/ (default: the config's out_dir)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = []
    for spec in args.exp:
        cfg_path, name, prot = (spec.split(":") + ["", ""])[:3]
        cfg = load_config(cfg_path, [])
        feat = os.path.join(args.runs_root or cfg["out_dir"], cfg["exp"], "analysis", "gate_features.npz")
        if not os.path.exists(feat):
            print(f"skip {cfg['exp']}: {feat} is missing")
            continue
        ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
        forms = [f for f in FORMS if ex.has(f"eval__{f}")]
        if not forms:
            print(f"skip {cfg['exp']}: no eval__st2_* features yet (step p3.10)")
            continue
        protect = [c.strip() for c in prot.split(",") if c.strip()]
        wraps = any(ex.has(f"wrap__{t}__fit") for t in ex.tasks)
        variants = ["ratio", "twosided"] + (["remedy", "twosided_remedy"] if wraps and not protect else [])
        if protect:
            variants += ["protect"] + (["protect_remedy"] if wraps else [])
        first = [p for p in ex.eval_keys() if p.startswith("st_")]
        row = {"exp": cfg["exp"], "name": name, "label": sr.readable_label(cfg), "forms": forms, "gates": {}}
        for v in variants:
            g = sr.build_variant(ex, v, args.eps, args.alpha, protect)
            st = sr.gate_stats(ex, g)
            row["gates"][v] = {"beta": st["power_pre"], "fire": {p: st["fire"][p] for p in forms + first},
                               "cp": {p: st["cp"][p] for p in forms + first},
                               "n": {p: int(len(g["any"][p])) for p in forms}}
            print(f"{cfg['exp']} {v}: " + "  ".join(f"{SHORT[p]} {100 * st['fire'][p]:.1f}%" for p in forms), flush=True)
        res.append(row)
    json.dump(res, open(os.path.join(args.out, "stress2.json"), "w"), indent=1)
    write_table(res, args.out)
    print(f"wrote {args.out}/stress2.json and stress2_table.tex")


def write_table(res, out):
    forms = [f for f in FORMS if any(f in r["forms"] for r in res)]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{Harmful requests in forms written before any gate was evaluated on them (step p3.10b): firing "
             r"(\%) of each gate at $\varepsilon=0.03$ on the 250 held-out AdvBench requests in every form, computed "
             r"before training. A gate that does not fire leaves the base model's answer in place for every model behind "
             r"it. $\beta$: coverage of the unlabeled RL prompts (mean over tasks for streams).}",
             r"\label{tab:stress2}", r"\begin{tabular}{l l r " + "r" * len(forms) + "}", r"\toprule",
             "Experiment & Gate & $\\beta$ & " + " & ".join(SHORT[f] for f in forms) + r" \\", r"\midrule"]
    for r in res:
        first = True
        for v, g in r["gates"].items():
            b = [x for x in g["beta"].values() if x is not None]
            beta = f"{100 * sum(b) / len(b):.1f}" if b else "--"
            cells = [("--" if f not in g["fire"] or g["fire"][f] is None else f"{100 * g['fire'][f]:.1f}") for f in forms]
            lab = r["label"].replace("_", r"\_") if first else ""
            lines.append(f"{lab} & {GATE_NAMES.get(v, v)} & {beta} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "stress2_table.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
