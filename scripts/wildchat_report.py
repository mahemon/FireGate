#!/usr/bin/env python
"""Real-user requests (WildChat; Nov 1, PLAN_NOV.md): what the certificate means on traffic nobody calibrated on (CPU).

    python scripts/wildchat_report.py --results paper/nov_results --runs runs --out paper/nov_results/overview

Reads <results>/<exp>/switch/switch_summary.json (scripts/switch_report.py; its gates are evaluated on the stored
WildChat features, its runs on extra/wildchat/ or on the WildChat pool of the stage evaluations) and, for the
non-fired check, the runs' own files. One row per experiment that has WildChat features:
  deployed tripwire (eps of the config): firing on WildChat and its 95% Clopper-Pearson bound (the certificate
  there), next to its firing on held-out Dolly; the two-sided tripwire (+ new-task region) the same way;
  Seq-GRPO: share of WildChat responses changed, ungated and behind each of the two tripwires (the switch);
  Tripwire-prompt: share changed, and non-fired responses that differ from the base model's (0 by Theorem 3.2).
Writes <out>/wildchat.{md,tex,json}.
"""
from __future__ import annotations

import argparse
import json
import os

ORDER = ["v2_ladder_ifc_q15", "v2_overlap_q15", "v2_stream_q15", "v2_ladder_ifc_q7", "v2_overlap_q7", "v2_overlap_olmo7",
         "fam_ifc_llama3b", "fam_nq_llama3b", "fam_ifc_llama8", "fam_nq_llama8", "fam_ifc_ds7", "fam_nq_ds7",
         "fam_ifc_mistral7", "fam_nq_mistral7", "nov_ifc_llama13", "nov_nq_llama13", "nov_ifc_qwen14", "nov_nq_qwen14",
         "nov_code_q15"]


def rows_of(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["i"])


def fired(r):
    return any(v is not None for v in (r.get("fire") or {}).values())


def nonfired(runs, exp):
    """(non-fired WildChat responses that differ from the base model's, non-fired responses) over the
    Tripwire-prompt runs: from extra/wildchat/ (eval_extra.py) or from the WildChat pool of the stage evaluations"""
    root = os.path.join(runs, exp)
    if not os.path.isdir(root):
        return None
    diff = n = 0
    found = False
    for sd in sorted(d for d in os.listdir(root) if d.startswith("s") and d[1:].isdigit()):
        md = os.path.join(root, sd, "tripwire_prompt")
        if not os.path.exists(os.path.join(md, "DONE")):
            continue
        fin, base = rows_of(os.path.join(md, "extra", "wildchat", "final_wildchat.jsonl")), \
            rows_of(os.path.join(md, "extra", "wildchat", "base_wildchat.jsonl"))
        if fin is None:
            stages = sorted(int(d[5:]) for d in os.listdir(md) if d.startswith("stage") and d[5:].isdigit())
            if stages:
                fin = rows_of(os.path.join(md, f"stage{stages[-1]}", "eval", "wildchat.jsonl"))
                base = rows_of(os.path.join(md, "stage0", "eval", "wildchat.jsonl"))
        if not fin or not base or len(fin) != len(base):
            continue
        found = True
        for a, b in zip(fin, base):
            if not fired(a):
                n += 1
                diff += int(a["out"] != b["out"])
    return (diff, n) if found else None


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def first(x):
    return None if x is None else x[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.results, "overview")
    os.makedirs(out_dir, exist_ok=True)
    exps = [e for e in ORDER if os.path.exists(os.path.join(args.results, e, "switch", "switch_summary.json"))]
    data, md, tex = [], [], []
    md += ["| experiment | tripwire on WildChat (CP95) | on Dolly | two-sided on WildChat (CP95) | Seq-GRPO changed: "
           "ungated (seeds) | behind tripwire | behind two-sided | Tripwire-prompt changed | non-fired changed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for exp in exps:
        s = json.load(open(os.path.join(args.results, exp, "switch", "switch_summary.json")))
        e = s["main_eps"]
        g = s["gates"].get(f"ratio|{e:g}") or s["gates"].get(f"ratio|{e}") or {}
        g2 = s["gates"].get(f"twosided|{e:g}") or s["gates"].get(f"twosided|{e}") or {}
        if (g.get("fire") or {}).get("wildchat") is None:
            continue
        fam = s["families"]
        seq = fam.get("seq", {})
        sw = (seq.get("switch") or {})
        sw1 = sw.get(f"ratio|{e:g}") or sw.get(f"ratio|{e}") or {}
        sw2 = sw.get(f"twosided|{e:g}") or sw.get(f"twosided|{e}") or {}
        own = seq.get("own") or {}
        twp = (fam.get("tripwire_prompt", {}).get("own") or {})
        nfc = nonfired(args.runs, exp)
        row = {"exp": exp, "label": s.get("label", exp), "eps": e,
               "fire": g["fire"].get("wildchat"), "cp": g["cp"].get("wildchat"), "fire_dolly": g["fire"].get("dolly"),
               "fire2": (g2.get("fire") or {}).get("wildchat"), "cp2": (g2.get("cp") or {}).get("wildchat"),
               "seq_chg": own.get("wildchat:chg"), "seq_sw_chg": sw1.get("wildchat:chg"),
               "seq_sw2_chg": sw2.get("wildchat:chg"), "seq_refused": own.get("wildchat:refused"),
               "twp_chg": twp.get("wildchat:chg"), "nonfired": nfc}
        data.append(row)
        nseq = row["seq_chg"][2] if row["seq_chg"] else 0
        cells = [row["label"], f"{pct(row['fire'])} ({pct(row['cp'])})", pct(row["fire_dolly"]),
                 f"{pct(row['fire2'])} ({pct(row['cp2'])})", f"{pct(first(row['seq_chg']))} ({nseq})",
                 pct(first(row["seq_sw_chg"])), pct(first(row["seq_sw2_chg"])), pct(first(row["twp_chg"])),
                 "--" if nfc is None else f"{nfc[0]} / {nfc[1]}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(c.replace("_", r"\_") for c in cells) + r" \\")
    head = [r"% WildChat: 500 first user turns of English conversations, never used to fit or calibrate (PLAN_NOV.md).",
            r"% Firing and changed responses in %, CP95: 95% Clopper-Pearson bound; Seq-GRPO changed: seed-0 runs.",
            r"\begin{tabular}{l rr r r rrr r}", r"\toprule",
            r"experiment & tripwire (CP) & on Dolly & two-sided (CP) & Seq ungated & behind tripwire & behind two-sided & "
            r"TW-prompt & non-fired chg. \\", r"\midrule"]
    with open(os.path.join(out_dir, "wildchat.json"), "w") as f:
        json.dump(data, f, indent=1)
    with open(os.path.join(out_dir, "wildchat.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "wildchat.tex"), "w") as f:
        f.write("\n".join(head + tex + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    print("\n".join(md))
    print(f"wrote {out_dir}/wildchat.md / .tex / .json ({len(data)} experiments)")


if __name__ == "__main__":
    main()
