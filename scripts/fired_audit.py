#!/usr/bin/env python
"""Step p3.17 (Nov 4, PLAN_PAPER3.md): what changes where the gate fires on harmful requests (CPU, stored responses).

    python scripts/fired_audit.py --exp configs/v2/overlap_q15.yaml:Nq: ... --out paper/v3_results/p3_17

The certificate bounds how often the gated model departs from the old one, and lets it change arbitrarily where the
gate fires. For every experiment (config:name:protected components, as scripts/external_report.py) the deployed and
the chosen gate are built from stored features exactly as in p3.12, and Seq-GRPO is put behind each (Corollary 3.4: a
prompt on which the gate is silent gets the base model's response). On every harmful pool with stored responses of
both the base and the trained model (held-out AdvBench, XSTest's unsafe prompts, the stress forms st_* / st2_*, the
task-format requests advwrap_*, the format shifts advbench_sys / advbench_turn, the external sets ext_*), counts are
pooled over seeds:
  n            prompts x seeds;  fired: prompts on which the gate fired
  base_ref     refusals of the base model on the fired prompts;  gated_ref: refusals behind the gate there
  newly        prompts the base model refuses and the gated model answers (all of them lie where the gate fired)
  newly_share  newly / n, with its 95% Wilson interval: the share of all harmful prompts on which the gated system
               stops refusing, the severe failure that a small firing rate could hide
The same counts are reported for the runs trained under the gate (tripwire_prompt), from their own responses.
Refusal is the string check of twcore/tasks.is_refusal, as everywhere in the paper; no judge is used.
Outputs in --out: fired_audit.json, fired_table.tex, fired_summary.txt.
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

import switch_report as sr  # noqa: E402
from external_report import GATE_NAMES, chosen_variant  # noqa: E402
from twcore.config import load_config  # noqa: E402

HARM_PREFIX = ("st_", "st2_", "ext_", "advwrap_")
HARM_NAMES = ("advbench", "advbench_sys", "advbench_turn", "xstest")


def harmful(pool):
    return pool in HARM_NAMES or pool.startswith(HARM_PREFIX)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_pair(md, K):
    """{pool: (base rows, final rows)} of one run: stage evaluations, extra evaluations and the external sets"""
    loaded = sr.load_run(md, K)
    if loaded is None:
        return None
    s0, sK = loaded
    out = {p: (s0[p], sK[p]) for p in sK if p in s0 and s0[p] is not None and sK[p] is not None}
    xd = os.path.join(md, "extra", "external")
    if os.path.isdir(xd):
        for fn in os.listdir(xd):
            if fn.startswith("final_") and fn.endswith(".jsonl"):
                name = fn[len("final_"):-6]
                b, f = sr.read_rows(os.path.join(xd, f"base_{name}.jsonl")), sr.read_rows(os.path.join(xd, fn))
                if b is not None and f is not None:
                    out[name] = (b, f)
    return out


def rows_of(pool, base, fin):
    """aligned (base, final) refusal pairs of a harmful pool; XSTest: its unsafe prompts only. Returns the index list
    too, so the gate's firing can be read at the same positions."""
    if len(base) != len(fin) or not base or "refused" not in base[0].get("m", {}):
        return None
    idx = [i for i, b in enumerate(base) if pool != "xstest" or b.get("label") == "unsafe"]
    return idx, [int(base[i]["m"]["refused"]) for i in idx], [int(fin[i]["m"]["refused"]) for i in idx]


def empty():
    return {"n": 0, "fired": 0, "base_ref": 0, "gated_ref": 0, "newly": 0, "newly_refused": 0}


def add(acc, fire, b, f):
    for on, rb, rf in zip(fire, b, f):
        acc["n"] += 1
        if on:
            acc["fired"] += 1
            acc["base_ref"] += rb
            acc["gated_ref"] += rf
            acc["newly"] += int(rb == 1 and rf == 0)
            acc["newly_refused"] += int(rb == 0 and rf == 1)


def finish(acc):
    n, k = acc["n"], acc["fired"]
    acc["fire"] = k / n if n else None
    acc["base_ref_fired"] = acc["base_ref"] / k if k else None
    acc["gated_ref_fired"] = acc["gated_ref"] / k if k else None
    acc["newly_share"] = acc["newly"] / n if n else None
    acc["newly_ci"] = wilson(acc["newly"], n)
    acc["gated_ref_fired_ci"] = wilson(acc["gated_ref"], k) if k else (None, None)
    return acc


def runs_of(runs, method):
    out = []
    if not os.path.isdir(runs):
        return out
    for sd in sorted((d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:])):
        md = os.path.join(runs, sd, method)
        if os.path.exists(os.path.join(md, "DONE")):
            out.append((sd, md))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True, help="config:name:protected components")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_17"))
    ap.add_argument("--runs-root", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = []
    for spec in args.exp:
        cfg_path, name, prot = (spec.split(":") + ["", ""])[:3]
        cfg = load_config(cfg_path, [])
        runs = os.path.join(args.runs_root or cfg["out_dir"], cfg["exp"])
        feat = os.path.join(runs, "analysis", "gate_features.npz")
        if not os.path.exists(feat):
            print(f"skip {cfg['exp']}: {feat} is missing")
            continue
        ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
        protect = [c.strip() for c in prot.split(",") if c.strip()]
        wraps = any(ex.has(f"wrap__{t}__fit") for t in ex.tasks)
        chosen = chosen_variant(protect, wraps)
        gates = {v: sr.build_variant(ex, v, args.eps, args.alpha, protect) for v in dict.fromkeys(["ratio", chosen])}
        row = {"exp": cfg["exp"], "name": name, "label": sr.readable_label(cfg), "chosen": chosen, "gates": {},
               "trained": {}, "seeds": []}
        for v, g in gates.items():
            pools, total = {}, empty()
            for sd, md in runs_of(runs, "seq"):
                pairs = load_pair(md, ex.K)
                if pairs is None:
                    continue
                if sd not in row["seeds"]:
                    row["seeds"].append(sd)
                for p, (b, f) in pairs.items():
                    if not harmful(p) or p not in g["any"]:
                        continue
                    al = rows_of(p, b, f)
                    if al is None or len(g["any"][p]) != len(b):
                        continue
                    idx, rb, rf = al
                    fire = [bool(g["any"][p][i]) for i in idx]
                    add(pools.setdefault(p, empty()), fire, rb, rf)
                    add(total, fire, rb, rf)
            row["gates"][v] = {"pools": {p: finish(a) for p, a in sorted(pools.items())}, "total": finish(total)}
        pools, total = {}, empty()
        for sd, md in runs_of(runs, "tripwire_prompt"):
            pairs = load_pair(md, ex.K)
            if pairs is None:
                continue
            for p, (b, f) in pairs.items():
                if not harmful(p):
                    continue
                al = rows_of(p, b, f)
                if al is None:
                    continue
                idx, rb, rf = al
                changed = [b[i]["out"] != f[i]["out"] for i in idx]   # its own gate fired where the response changed
                add(pools.setdefault(p, empty()), changed, rb, rf)
                add(total, changed, rb, rf)
        row["trained"] = {"pools": {p: finish(a) for p, a in sorted(pools.items())}, "total": finish(total)}
        t = row["gates"][chosen]["total"]
        print(f"{cfg['exp']} {chosen}: {t['n']} harmful prompts x seeds, fired {t['fired']}, newly answered "
              f"{t['newly']} ({pc(t['newly_share'], 2)}%)", flush=True)
        res.append(row)
    json.dump(res, open(os.path.join(args.out, "fired_audit.json"), "w"), indent=1)
    write_table(res, args.out)
    txt = summary(res)
    open(os.path.join(args.out, "fired_summary.txt"), "w").write(txt)
    print(txt + f"wrote {args.out}/fired_audit.json, fired_table.tex, fired_summary.txt")


def worst(pools):
    """the pool with the largest share of newly answered prompts"""
    c = [(a["newly_share"], p) for p, a in pools.items() if a["newly_share"] is not None and a["n"]]
    return max(c) if c else (None, None)


def pc(v, d=1):
    return "--" if v is None else f"{100 * v:.{d}f}"


def summary(res):
    L = ["p3.17: harmful prompts where the gate fires (Seq-GRPO behind the gate; refusal string check)"]
    for r in res:
        for v, g in r["gates"].items():
            t = g["total"]
            ws, wp = worst(g["pools"])
            L.append(f"{r['exp']} {GATE_NAMES[v]}{' (chosen)' if v == r['chosen'] else ''}: n {t['n']}, fired "
                     f"{pc(t['fire'], 2)}%, refusal on fired {pc(t['base_ref_fired'])}% -> {pc(t['gated_ref_fired'])}%, "
                     f"newly answered {t['newly']} = {pc(t['newly_share'], 2)}% [{pc(t['newly_ci'][0], 2)}, "
                     f"{pc(t['newly_ci'][1], 2)}]; worst pool {wp} {pc(ws, 2)}%")
        t = r["trained"]["total"]
        if t["n"]:
            L.append(f"{r['exp']} trained under the gate: n {t['n']}, changed {pc(t['fire'], 2)}%, newly answered "
                     f"{t['newly']} = {pc(t['newly_share'], 2)}%")
    return "\n".join(L) + "\n"


def write_table(res, out):
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Harmful prompts on which the gate fires, where the certificate allows any change (seed-pooled "
             r"counts over every harmful pool with stored responses: held-out AdvBench, XSTest's unsafe prompts, the "
             r"stress forms, the task-format requests, the format shifts and the external sets). Seq-GRPO behind the "
             r"chosen and the deployed gate; trained: \methodp{}'s own runs, where a response can change only if its "
             r"gate fired. Fired: share of prompts on which the gate fired; refusal on fired: the base model's and the "
             r"gated model's refusal rate there; newly answered: prompts the base model refuses and the gated model "
             r"answers, as a share of all, with its 95\% Wilson interval; worst pool: the largest such share.}",
             r"\label{tab:fired}",
             r"\begin{tabular}{l l r r rr r r l}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{refusal on fired (\%)} & & & \\", r"\cmidrule(lr){5-6}",
             r"Experiment & Gate & $n$ & Fired (\%) & base & gated & Newly answered (\%) & [95\% CI] & worst pool (\%) \\",
             r"\midrule"]
    for r in res:
        first = r["label"].replace("_", r"\_")
        for v in dict.fromkeys([r["chosen"], "ratio"]):
            t = r["gates"][v]["total"]
            if not t["n"]:
                continue
            ws, wp = worst(r["gates"][v]["pools"])
            star = r"$^\ast$" if v == r["chosen"] else ""
            lines.append(f"{first} & {GATE_NAMES[v]}{star} & {t['n']} & {pc(t['fire'])} & {pc(t['base_ref_fired'])} & "
                         f"{pc(t['gated_ref_fired'])} & {pc(t['newly_share'], 2)} & [{pc(t['newly_ci'][0], 2)}, "
                         f"{pc(t['newly_ci'][1], 2)}] & {(wp or '--').replace('_', chr(92) + '_')} {pc(ws, 2)} \\\\")
            first = ""
        t = r["trained"]["total"]
        if t["n"]:
            ws, wp = worst(r["trained"]["pools"])
            lines.append(f"{first} & trained & {t['n']} & {pc(t['fire'])} & {pc(t['base_ref_fired'])} & "
                         f"{pc(t['gated_ref_fired'])} & {pc(t['newly_share'], 2)} & [{pc(t['newly_ci'][0], 2)}, "
                         f"{pc(t['newly_ci'][1], 2)}] & {(wp or '--').replace('_', chr(92) + '_')} {pc(ws, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "fired_table.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
