#!/usr/bin/env python
"""Does the judge agree with a judge of another model family, and with a person? (CPU; reads stored labels.)

    python scripts/judge_agreement.py --exps Nq:runs/v2_overlap_q15,Ifc:runs/v2_ladder_ifc_q15 --tag olmo
        [--manual paper/v2_results/v2_overlap_q15/switch/sample_for_manual.tsv ...] [--out paper/v2_results/overview]

For every run judged by both judges (runs/<exp>/analysis/judge/<seed>__<family>.json and
runs/<exp>/analysis/judge_<tag>/<seed>__<family>.json) and every pool: the pairs both judges labelled, raw
agreement, Cohen's kappa over SAME / STYLE / DIFFERENT, agreement on DIFFERENT against the rest, and the share of all
responses of the pool rated substantively different (DIFFERENT) by each judge, per family. UNPARSED labels are left out.
--manual: copies of judge/sample_for_manual.tsv with the column your_label filled in (SAME, STYLE or DIFFERENT, or
S / T / D); each judge's labels are compared with yours.
Writes <out>/judge_agreement.json, judge_agreement.tex and judge_agreement_macros.tex (\\ja<Exp>...).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

LABELS = ("SAME", "STYLE", "DIFFERENT")
SHORT = {"S": "SAME", "T": "STYLE", "D": "DIFFERENT", "SAME": "SAME", "STYLE": "STYLE", "DIFFERENT": "DIFFERENT",
         "DIFF": "DIFFERENT"}


def kappa(a, b, cats):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def compare(a, b):
    """a, b: lists of labels on the same pairs (UNPARSED already removed)"""
    if not a:
        return None
    ba, bb = ["D" if x == "DIFFERENT" else "O" for x in a], ["D" if x == "DIFFERENT" else "O" for x in b]
    conf = {x: {y: sum(1 for p, q in zip(a, b) if p == x and q == y) for y in LABELS} for x in LABELS}
    return {"n": len(a), "agree": sum(x == y for x, y in zip(a, b)) / len(a), "kappa": kappa(a, b, LABELS),
            "agree_diff": sum(x == y for x, y in zip(ba, bb)) / len(a), "kappa_diff": kappa(ba, bb, ("D", "O")),
            "confusion": conf}


def pool_size(runs, sd, fam, pool):
    p = os.path.join(runs, sd, fam, "stage0", "eval", f"{pool}.jsonl")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return sum(1 for line in fh if line.strip())


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{100 * v:.1f}" if pct else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", required=True, help="Name:runs/<exp>,... (names: letters only)")
    ap.add_argument("--tag", default="olmo")
    ap.add_argument("--manual", nargs="*", default=[])
    ap.add_argument("--out", default="paper/v2_results/overview")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res, macros, lines = {}, {}, []
    for item in args.exps.split(","):
        if not item.strip():
            continue
        name, runs = item.split(":", 1)
        da, db = os.path.join(runs, "analysis", "judge"), os.path.join(runs, "analysis", f"judge_{args.tag}")
        if not (os.path.isdir(da) and os.path.isdir(db)):
            print(f"{name}: no labels of both judges ({da}, {db})")
            continue
        per_pool, per_fam = {}, {}
        for fn in sorted(os.listdir(db)):
            if not fn.endswith(".json") or fn.startswith("_") or not os.path.exists(os.path.join(da, fn)):
                continue
            sd, fam = fn[:-5].split("__", 1)
            la = json.load(open(os.path.join(da, fn))).get("labels", {})
            lb = json.load(open(os.path.join(db, fn))).get("labels", {})
            for pool in sorted(set(la) & set(lb)):
                common = [i for i in lb[pool] if i in la[pool] and la[pool][i] in LABELS and lb[pool][i] in LABELS]
                pp = per_pool.setdefault(pool, ([], []))
                pp[0].extend(la[pool][i] for i in common)
                pp[1].extend(lb[pool][i] for i in common)
                n = pool_size(runs, sd, fam, pool)
                if n:
                    f = per_fam.setdefault(fam, {}).setdefault(pool, {"a": [], "b": []})
                    f["a"].append(sum(la[pool][i] == "DIFFERENT" for i in common) / n)
                    f["b"].append(sum(lb[pool][i] == "DIFFERENT" for i in common) / n)
        r = {"pools": {p: compare(a, b) for p, (a, b) in per_pool.items()},
             "families": {fam: {p: {"subst_a": sum(v["a"]) / len(v["a"]), "subst_b": sum(v["b"]) / len(v["b"]),
                                    "n_runs": len(v["a"])} for p, v in d.items()} for fam, d in per_fam.items()}}
        res[name] = r
        first = True
        for p, c in r["pools"].items():
            if c is None:
                continue
            sa = r["families"].get("seq", {}).get(p, {})
            lines.append(" & ".join([name if first else "", p.replace("_", r"\_"),
                                     str(c["n"]), fmt(c["agree"]), fmt(c["kappa"], False), fmt(c["agree_diff"]),
                                     fmt(c["kappa_diff"], False), fmt(sa.get("subst_a")), fmt(sa.get("subst_b"))])
                         + r" \\")
            first = False
            P = p.capitalize().replace("_", "")
            if P.isalpha():
                macros[f"ja{name}{P}N"] = str(c["n"])
                macros[f"ja{name}{P}Agree"] = fmt(c["agree"])
                macros[f"ja{name}{P}Kappa"] = fmt(c["kappa"], False)
                macros[f"ja{name}{P}KappaDiff"] = fmt(c["kappa_diff"], False)
                if sa:
                    macros[f"ja{name}{P}SeqSubstA"] = fmt(sa["subst_a"])
                    macros[f"ja{name}{P}SeqSubstB"] = fmt(sa["subst_b"])
        print(f"{name}: " + ", ".join(f"{p} n={c['n']} agree {100 * c['agree']:.1f}% kappa {c['kappa'] if c['kappa'] is None else round(c['kappa'], 2)}"
                                      for p, c in r["pools"].items() if c))

    # manual labels
    man = {}
    for path in args.manual:
        if not os.path.exists(path):
            print("no such file:", path)
            continue
        runs_dir = None
        rows = list(csv.DictReader(open(path), delimiter="\t"))
        human = [(r["run"], r["pool"], r["i"], SHORT.get(r.get("your_label", "").strip().upper()), r["judge"])
                 for r in rows]
        human = [h for h in human if h[3] is not None]
        if not human:
            print(f"{path}: no filled your_label column")
            continue
        exp = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        for item in args.exps.split(","):
            nm, rd = item.split(":", 1)
            if os.path.basename(rd.rstrip("/")) == exp:
                runs_dir, name = rd, nm
                break
        a = [h[4] for h in human if h[4] in LABELS]
        hh = [h[3] for h in human if h[4] in LABELS]
        entry = {"file": path, "n": len(human), "judge": compare(a, hh)}
        if runs_dir is not None:   # the second judge's labels of the same pairs
            pairs = []
            for run, pool, i, lab, _ in human:
                fp = os.path.join(runs_dir, "analysis", f"judge_{args.tag}", f"{run}.json")
                if os.path.exists(fp):
                    b = json.load(open(fp)).get("labels", {}).get(pool, {}).get(i)
                    if b in LABELS:
                        pairs.append((b, lab))
            entry["second"] = compare([p[0] for p in pairs], [p[1] for p in pairs])
            if entry["judge"]:
                macros[f"ja{name}HumanN"] = str(entry["judge"]["n"])
                macros[f"ja{name}HumanAgree"] = fmt(entry["judge"]["agree"])
                macros[f"ja{name}HumanKappa"] = fmt(entry["judge"]["kappa"], False)
                macros[f"ja{name}HumanAgreeDiff"] = fmt(entry["judge"]["agree_diff"])
            if entry["second"]:
                macros[f"ja{name}HumanAgreeB"] = fmt(entry["second"]["agree"])
        man[path] = entry
        print(f"{path}: {len(human)} labelled pairs; judge agrees on {fmt(entry['judge']['agree']) if entry['judge'] else '--'}%")
    res["_manual"] = man

    with open(os.path.join(args.out, "judge_agreement.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Agreement of the judge (Qwen2.5-7B-Instruct) with a judge of another model family "
           r"(OLMo-2-7B-Instruct) on the changed responses both labelled. Agree: same label (\%); $\kappa$: Cohen's "
           r"kappa over SAME, STYLE, DIFFERENT; D: the same for DIFFERENT against the rest; Subst.: share of all "
           r"responses of the pool that each judge rates as substantively different, Seq-GRPO, mean over seeds.}",
           r"\label{tab:judge_agreement}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l r rr rr rr}",
           r"\toprule", r" & & & \multicolumn{2}{c}{all labels} & \multicolumn{2}{c}{D vs rest} & "
           r"\multicolumn{2}{c}{Subst.} \\", r"Experiment & Pool & $n$ & Agree & $\kappa$ & Agree & $\kappa$ & Qwen & OLMo \\",
           r"\midrule"] + lines + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(args.out, "judge_agreement.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    with open(os.path.join(args.out, "judge_agreement_macros.tex"), "w") as fh:
        fh.write("% generated by scripts/judge_agreement.py\n")
        for k in sorted(macros):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
    print(f"wrote {args.out}/judge_agreement.* ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
