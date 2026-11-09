#!/usr/bin/env python
"""Step p3.22 (Nov 4, PLAN_PAPER3.md): a language model as one more score class of the gate, merged with the linear
classifier and the Gaussian ratio fitted and calibrated on the same components.

    python scripts/llm_gate_grid.py --root paper/v3_results/p3_22 --p320 paper/v3_results/p3_20 \
        --fixes paper/v3_results

Reads <root>/<exp>/<variant>/scores.json (score "llm", scripts/score_classes.py on llm_scores.npz) and, for the same
experiment and reference, the linear classifier and Gaussian ratio: p3.20 (<p320>/<exp>/scores.json) for the deployed
reference, p3.10a (<fixes>/<exp>/scores_<variant>/scores.json) for the protect variants. Reported per row: coverage
beta of unlabeled RL prompts, held-out Dolly firing with its 95% bound, the largest firing over the harmful forms
(st_*, st2_*, ext_*, held-out AdvBench), and Seq-GRPO's gain behind the gate.
Outputs in --root: llm_gate_grid.json, llm_gate_table.tex.
"""
from __future__ import annotations

import argparse
import json
import os

HARM = ("advbench", "st_", "st2_", "ext_", "pr_mj_", "pr_mt_harm", "pr_sys_harm")


def harm_max(r):
    vals = [v for p, v in r["fire"].items() if p == "advbench" or any(p.startswith(h) for h in HARM[1:])]
    return max(vals) if vals else None


def row(name, r, task):
    g = r["seq_behind"].get(f"{task}:correct")
    return {"score": name, "beta": r["beta"], "dolly": r["fire"].get("dolly"), "cp": r["cp"].get("dolly"),
            "harm_max": harm_max(r), "gain": None if g is None else g[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("paper", "v3_results", "p3_22"))
    ap.add_argument("--p320", default=os.path.join("paper", "v3_results", "p3_20"))
    ap.add_argument("--fixes", default=os.path.join("paper", "v3_results"))
    args = ap.parse_args()
    out = []
    for exp in sorted(os.listdir(args.root)) if os.path.isdir(args.root) else []:
        for variant in ("deployed", "protect_remedy"):
            f = os.path.join(args.root, exp, variant, "scores.json")
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            rows = []
            other = os.path.join(args.p320, exp, "scores.json") if variant == "deployed" else \
                os.path.join(args.fixes, exp, f"scores_{variant}", "scores.json")
            if os.path.exists(other):
                o = json.load(open(other))
                rows += [row(n, o["scores"][n], o["task"]) for n in ("probe", "lda") if n in o["scores"]]
            rows.append(row("llm", r["scores"]["llm"], r["task"]))
            own = (r.get("seq_own", {}).get(f"{r['task']}:correct") or [None])[0]
            out.append({"exp": exp, "label": r["label"], "task": r["task"], "variant": variant, "own_gain": own,
                        "rows": rows})
    os.makedirs(args.root, exist_ok=True)
    json.dump(out, open(os.path.join(args.root, "llm_gate_grid.json"), "w"), indent=1)
    pc = lambda v: "--" if v is None else f"{100 * v:.1f}"  # noqa: E731
    pg = lambda v: "--" if v is None else f"{100 * v:+.1f}"  # noqa: E731
    names = {"probe": "linear classifier", "lda": "Gaussian ratio", "llm": "language model"}
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Three scores for the same gate: the linear classifier and the Gaussian ratio, fitted on the task's and "
         r"the reference's prompts, and a language model (Qwen2.5-7B-Instruct) asked once, with a question fixed before "
         r"any score, whether a prompt is a request of the task, score $\log p(\text{Yes})-\log p(\text{No})$; each "
         r"calibrated at $\varepsilon=0.03$ on the same components. $\beta$: coverage of unlabeled RL "
         r"prompts (\%); Dolly: held-out firing [95\% bound]; Harm.: largest firing over held-out harmful forms; Gain: "
         r"Seq-GRPO's gain behind the gate (points), against its own $\bar G$.}",
         r"\label{tab:llmgate}", r"\begin{tabular}{l l l r r r r}", r"\toprule",
         r"Experiment & Reference & Score & $\beta$ & Dolly & Harm. & Gain \\", r"\midrule"]
    for e in out:
        first = True
        for rr in e["rows"]:
            lab = (e["label"].replace("_", r"\_") + f" ($\\bar G$ {pg(e['own_gain'])})") if first else ""
            ref = e["variant"].replace("_", " ") if first else ""
            L.append(f"{lab} & {ref} & {names.get(rr['score'], rr['score'])} & {pc(rr['beta'])} & "
                     f"{pc(rr['dolly'])} [{pc(rr['cp'])}] & {pc(rr['harm_max'])} & {pg(rr['gain'])} \\\\")
            first = False
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(args.root, "llm_gate_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"wrote {args.root}/llm_gate_grid.json and llm_gate_table.tex ({len(out)} rows)")


if __name__ == "__main__":
    main()
