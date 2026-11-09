#!/usr/bin/env python
"""Step p3.9 (Nov 2 plan, implemented Nov 3; PLAN_PAPER3.md): Table 2 of the paper (the switch on seven models)
behind the gate the paper recommends, from stored results (CPU, seconds).

    python scripts/recommended_table.py --out paper/v3_results/p3_9 [--glob 'paper/fam_results/*/switch/switch_summary.json']

Table 2 puts Seq-GRPO behind the deployed one-sided prompt-time gate. The paper recommends the two-sided gate, with
the remedy where the task has harmful-format components (IF): twosided_remedy|0.03 where scripts/switch_report.py
built it, otherwise twosided|0.03. For every experiment with a switch_summary.json this reports, for both gates at
eps = 0.03: coverage beta on unlabeled RL prompts, the gain Seq-GRPO keeps behind it (mean over seeds), the Dolly
responses it changes, the 95% Clopper-Pearson bound of its firing on held-out Dolly, and its firing on MMLU, the
jailbreak templates and WildChat (where stored). The predictions of the Nov 2 plan (written before the step) are
checked: coverage at most 2.0 points below the deployed gate's; gain kept within 1 point of the deployed gate's; no
firing on MMLU or the jailbreak templates; WildChat at most 1.2%.
Outputs in --out: recommended.json, recommended_table.tex, recommended_summary.txt.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MODEL = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "OLMo-2-1124-7B-Instruct": "OLMo-2-7B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "deepseek-llm-7b-chat": "DeepSeek-LLM-7B",
         "Mistral-7B-Instruct-v0.3": "Mistral-7B"}
ORDER = ["Qwen2.5-1.5B", "Qwen2.5-7B", "OLMo-2-7B", "Llama-3.2-3B", "Llama-3.1-8B", "DeepSeek-LLM-7B", "Mistral-7B"]
TASK = {"ifc": "IF", "nq": "NQ"}


def first(v):
    return None if v is None else v[0]


def row_of(path):
    d = json.load(open(path))
    tasks = d["tasks"]
    if len(tasks) != 1 or tasks[0] not in TASK or "seq" not in d.get("families", {}):
        return None
    t = tasks[0]
    dep = "ratio|0.03"
    rec = "twosided_remedy|0.03" if "twosided_remedy|0.03" in d["gates"] else "twosided|0.03"
    if dep not in d["gates"] or rec not in d["gates"]:
        return None
    seq = d["families"]["seq"]
    model = d["label"].split(", ", 1)[1] if ", " in d["label"] else d["label"]
    r = {"exp": d["exp"], "model": MODEL.get(model, model), "task": TASK[t], "n": seq.get("n"),
         "gain": first(seq["own"].get(f"{t}:correct")), "chg": first(seq["own"].get("dolly:chg")), "gates": {}}
    for name, g in (("deployed", dep), ("recommended", rec)):
        G, sw = d["gates"][g], seq.get("switch", {}).get(g, {})
        fire = G["fire"]
        r["gates"][name] = {"key": g, "beta": G["power_pre"].get(t), "kept": first(sw.get(f"{t}:correct")),
                            "chg": first(sw.get("dolly:chg")), "cp_dolly": G["cp"].get("dolly"),
                            "mmlu": fire.get("mmlu"), "jb": max((v for k, v in fire.items() if k.startswith("jb_")),
                                                               default=None),
                            "wildchat": fire.get("wildchat")}
    return r


def pct(v, signed=False):
    if v is None:
        return "--"
    s = f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return s.replace("-", "$-$") if signed else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join("paper", "fam_results", "*", "switch", "switch_summary.json"))
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_9"))
    args = ap.parse_args()
    rows = [r for r in (row_of(p) for p in sorted(glob.glob(args.glob))) if r is not None]
    rows.sort(key=lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99, r["task"]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, "recommended.json"), "w"), indent=1)
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\caption{\Cref{tab:fam} behind the recommended gate (two-sided; with the remedy on IF), beside the deployed "
         r"one-sided gate, $\varepsilon=0.03$, three seeds. $\beta$: firing on unlabeled RL prompts, before training; "
         r"Kept: Seq-GRPO's gain behind the gate (points); Chg.: Dolly responses changed (\%); CP: 95\% Clopper--Pearson "
         r"bound of the firing on held-out Dolly (\%); MMLU, JB: firing on MMLU and on the jailbreak templates (\%).}",
         r"\label{tab:famrec}",
         r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tabular}{l l r rrr rrrrrr}", r"\toprule",
         r" & & & \multicolumn{3}{c}{deployed gate} & \multicolumn{6}{c}{recommended gate} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-12}",
         r"Model & Task & $\bar G$ & $\beta$ & Kept & CP & $\beta$ & Kept & Chg. & CP & MMLU & JB \\", r"\midrule"]
    for r in rows:
        a, b = r["gates"]["deployed"], r["gates"]["recommended"]
        L.append(f"{r['model']} & {r['task']} & {pct(r['gain'], True)} & {pct(a['beta'])} & {pct(a['kept'], True)} & "
                 f"{pct(a['cp_dolly'])} & {pct(b['beta'])} & {pct(b['kept'], True)} & {pct(b['chg'])} & "
                 f"{pct(b['cp_dolly'])} & {pct(b['mmlu'])} & {pct(b['jb'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(args.out, "recommended_table.tex"), "w").write("\n".join(L) + "\n")
    S = [f"{len(rows)} experiments (deployed -> recommended gate, eps 0.03)"]
    for r in rows:
        a, b = r["gates"]["deployed"], r["gates"]["recommended"]
        S.append(f"  {r['model']:16s} {r['task']:3s} gain {pct(r['gain'], True):>6s}  beta {pct(a['beta'])} -> "
                 f"{pct(b['beta'])}  kept {pct(a['kept'], True)} -> {pct(b['kept'], True)}  Dolly chg "
                 f"{pct(a['chg'])} -> {pct(b['chg'])}  CP {pct(a['cp_dolly'])} -> {pct(b['cp_dolly'])}  "
                 f"MMLU {pct(a['mmlu'])} -> {pct(b['mmlu'])}  JB {pct(a['jb'])} -> {pct(b['jb'])}  ({b['key']})")
    ok = lambda c, have: "n/a" if not have else ("PASS" if c else "FAIL")  # noqa: E731
    db = [(r, r["gates"]["deployed"]["beta"] - r["gates"]["recommended"]["beta"]) for r in rows
          if r["gates"]["deployed"]["beta"] is not None and r["gates"]["recommended"]["beta"] is not None]
    dk = [(r, r["gates"]["deployed"]["kept"] - r["gates"]["recommended"]["kept"]) for r in rows
          if r["gates"]["deployed"]["kept"] is not None and r["gates"]["recommended"]["kept"] is not None]
    mm = [r["gates"]["recommended"][k] for r in rows for k in ("mmlu", "jb") if r["gates"]["recommended"][k] is not None]
    wc = [r["gates"]["recommended"]["wildchat"] for r in rows if r["gates"]["recommended"]["wildchat"] is not None]
    S.append("predictions of the Nov 2 plan (step p3.9, written before the step)")
    S.append(f"coverage at most 2.0 points below the deployed gate: {ok(all(v <= 0.02 for _, v in db), db)} "
             f"(largest drop {max((100 * v for _, v in db), default=0):.1f} points)")
    S.append(f"gain kept within 1 point of the deployed gate's: {ok(all(abs(v) <= 0.01 for _, v in dk), dk)} "
             f"(largest difference {max((100 * abs(v) for _, v in dk), default=0):.1f} points)")
    S.append(f"no firing on MMLU or the jailbreak templates: {ok(all(v == 0 for v in mm), mm)} "
             f"(largest {max((100 * v for v in mm), default=0):.1f}%)")
    S.append(f"WildChat at most 1.2%: {ok(all(v <= 0.012 for v in wc), wc)}"
             + ("" if wc else " (not in these summaries; scripts/wildchat_report.py reports it)"))
    txt = "\n".join(S) + "\n"
    open(os.path.join(args.out, "recommended_summary.txt"), "w").write(txt)
    print(txt + f"wrote {args.out}/recommended.json, recommended_table.tex, recommended_summary.txt")


if __name__ == "__main__":
    main()
