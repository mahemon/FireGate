#!/usr/bin/env python
"""Step p3.20 (Nov 4, PLAN_PAPER3.md): the linear classifier against the Gaussian ratio on every model-task pair of
Table "The switch on seven models" (CPU, stored features; the per-pair runs are scripts/score_classes.py).

    python scripts/score_classes.py --config <cfg> --variant deployed --scores probe,lda --out <root>/<exp>   # per pair
    python scripts/score_grid.py --root paper/v3_results/p3_20                                               # merge

Both scores are fitted and calibrated as in p3.10a (deployed reference, eps 0.03, settings fixed before any result).
Reported per pair: each score's coverage beta of unlabeled RL prompts (a lower bound on beta* of Theorem 4.1), its
held-out Dolly firing with the 95% Clopper-Pearson bound, its firing on MMLU (a pool no component samples), and
Seq-GRPO's gain behind it (mean over seeds).
Outputs in --root: score_grid.json, score_grid_table.tex.
"""
from __future__ import annotations

import argparse
import json
import os

ORDER = ("Qwen2.5-1.5B", "Qwen2.5-7B", "OLMo-2", "Llama-3.2-3B", "Llama-3.1-8B", "DeepSeek", "Mistral")
# the names of the paper's tables (as scripts/recommended_table.py)
MODEL = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "OLMo-2-1124-7B-Instruct": "OLMo-2-7B", "OLMo-2-1124-7B": "OLMo-2-7B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "deepseek-llm-7b-chat": "DeepSeek-LLM-7B",
         "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Mistral-7B-v0.3": "Mistral-7B"}
TASK = {"ifc": "IF", "nq": "NQ"}


def key(r):
    lab = r["label"]
    m = next((i for i, o in enumerate(ORDER) if o.lower() in lab.lower()), len(ORDER))
    return m, r["task"]


def model_name(label):
    m = label.split(",")[-1].strip()
    return MODEL.get(m, MODEL.get(m + "-Instruct", m.replace("-Instruct", "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("paper", "v3_results", "p3_20"))
    args = ap.parse_args()
    rows = []
    for d in sorted(os.listdir(args.root)) if os.path.isdir(args.root) else []:
        f = os.path.join(args.root, d, "scores.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        s = r["scores"]
        if not {"probe", "lda"} <= set(s):
            continue
        g = lambda n: s[n]["seq_behind"].get(f"{r['task']}:correct")  # noqa: E731
        rows.append({"exp": r["exp"], "label": r["label"], "task": r["task"],
                     "own_gain": (r.get("seq_own", {}).get(f"{r['task']}:correct") or [None])[0],
                     **{n: {"beta": s[n]["beta"], "dolly": s[n]["fire"].get("dolly"), "cp": s[n]["cp"].get("dolly"),
                            "mmlu": s[n]["fire"].get("mmlu"), "gain": None if g(n) is None else g(n)[0]} for n in ("probe", "lda")}})
    rows.sort(key=key)
    json.dump(rows, open(os.path.join(args.root, "score_grid.json"), "w"), indent=1)
    pc = lambda v: "--" if v is None else f"{100 * v:.1f}"  # noqa: E731
    pg = lambda v: "--" if v is None else f"{100 * v:+.1f}"  # noqa: E731
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{The linear classifier against the Gaussian ratio on every model--task pair, both fitted and "
         r"calibrated at $\varepsilon=0.03$ on the deployed reference from stored features, settings fixed before any "
         r"result. $\beta$: coverage of unlabeled RL prompts (\%), a lower bound on $\beta^\ast$; Dolly: held-out "
         r"firing (\%) [95\% bound]; MMLU: firing on MMLU, which no component samples (\%); Gain: Seq-GRPO's gain "
         r"behind the gate (points), against its own gain $\bar G$.}",
         r"\label{tab:scoregrid}", r"\begin{tabular}{l l r rrrr rrrr}", r"\toprule",
         r" & & & \multicolumn{4}{c}{linear classifier} & \multicolumn{4}{c}{Gaussian ratio} \\",
         r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
         r"Model & Task & $\bar G$ & $\beta$ & Dolly & MMLU & Gain & $\beta$ & Dolly & MMLU & Gain \\", r"\midrule"]
    for r in rows:
        model = model_name(r["label"]).replace("_", r"\_")
        cells = []
        for n in ("probe", "lda"):
            c = r[n]
            cells += [pc(c["beta"]), f"{pc(c['dolly'])} [{pc(c['cp'])}]", pc(c.get("mmlu")), pg(c["gain"])]
        L.append(f"{model} & {TASK.get(r['task'], r['task'])} & {pg(r['own_gain'])} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(args.root, "score_grid_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"wrote {args.root}/score_grid.json and score_grid_table.tex ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
