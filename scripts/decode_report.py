"""Table of drift under other decoding rules (scripts/eval_extra.py --what decode), for the appendix.

    python scripts/decode_report.py --out paper/v2_results/decode \
        --run "Stream, \\methodp{}=runs/v2_stream_q15/s0/tripwire_prompt" --run "Stream, Seq-GRPO=runs/v2_stream_q15/s0/seq"

Each --run is "<label>=<run folder>"; runs without a decode summary are left out. Writes decode_table.tex,
decode_numbers.tex (\\dec... macros) and decode_summary.json.
Columns, per pool: firing and its 95% Clopper--Pearson bound (gated runs), the share of changed responses under shared
randomness (an upper bound on the TV distance), changed non-fired responses (Theorem 3.2: 0), the 95% lower bound on
the TV distance from response lengths, and the score change (points).
"""
import argparse
import json
import os


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(v, signed=False):
    if v is None:
        return "--"
    s = f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return s.replace("-", "$-$", 1) if signed else s


def setting_name(key):
    """t1_p0.9 -> T=1, top-p 0.9"""
    t, _, p = key[1:].partition("_p")
    return f"$T={t}$" + (f", top-$p$ {p}" if p else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="<label>=<run folder>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for spec in a.run:
        label, _, d = spec.partition("=")
        f = os.path.join(d, "extra", "summary.json")
        if not os.path.isfile(f):
            print(f"  {d}: no extra/summary.json, left out")
            continue
        s = json.load(open(f))
        if "decode" not in s:
            print(f"  {d}: no decode results yet, left out")
            continue
        gated = os.path.isfile(os.path.join(d, "stage1", "tripwire", "tripwire.pt"))
        runs.append((label, d, gated, s))
    if not runs:
        raise SystemExit("no decode results")

    rows, out, nonfired, checked, gated_chg, seq_ks, serving = [], {}, 0, 0, [], [], []
    ungated_lb = {}     # slug of an ungated run's label -> largest TV lower bound over the pools, per rule
    for label, d, gated, s in runs:
        out[label] = {"run": d, "gated": gated, "decode": s["decode"], "serving": s.get("decode_serving", {})}
        for key in s["decode"]:
            cell = {}
            for pool in ("dolly", "advbench", "xstest", "mmlu"):   # XSTest: only in the non-fired check
                st = s["decode"][key].get(pool)
                if not st:
                    continue
                seeds = st["seeds"]
                metric = {"dolly": "leak", "advbench": "refused", "xstest": "refused", "mmlu": "correct"}[pool]
                if not all(x.get("same_device", True) for x in seeds):
                    print(f"  WARNING {label} {key} {pool}: final and base decoded on different GPU models")
                cell[pool] = {"fire": mean(x["fire_rate"] for x in seeds), "cp": max(x["cert_cp95"] for x in seeds),
                              "chg": mean(x["changed"] for x in seeds),
                              "nf": sum(x["nonfired_changed"] for x in seeds),
                              "nnf": sum(x["n"] - x["fired"] for x in seeds),
                              "ks": mean(x.get("len_ks_lb95") for x in seeds),
                              "delta": mean(x.get(f"{metric}_delta") for x in seeds)}
                if gated:
                    nonfired += cell[pool]["nf"]
                    checked += cell[pool]["nnf"]
                    if pool == "dolly":
                        gated_chg.append(cell[pool]["chg"])
                elif pool == "dolly" and cell[pool]["ks"] is not None:
                    seq_ks.append(cell[pool]["ks"])
            dl, ad, mm = cell.get("dolly", {}), cell.get("advbench", {}), cell.get("mmlu", {})
            lbs = [c["ks"] for c in cell.values() if c.get("ks") is not None]
            lb = max(lbs) if lbs else None
            if not gated and lb is not None:
                slug = "".join(w[:1].upper() + w[1:].lower() for w in label.replace(",", " ").replace("-", " ").split()
                               if w.isalpha() and w.lower() != "grpo")
                ungated_lb.setdefault(slug, []).append(lb)
            fire = f"{pct(dl.get('fire'))} ({pct(dl.get('cp'))})" if gated and dl else "--"
            xs = cell.get("xstest", {})
            nf = (f"{dl['nf'] + ad.get('nf', 0) + xs.get('nf', 0) + mm.get('nf', 0)}" if gated and dl else "--")
            rows.append(f"{label} & {setting_name(key)} & {fire} & {pct(dl.get('chg'))} & "
                        f"{pct(ad.get('chg'))} & {pct(ad.get('delta'), True)} & {pct(mm.get('chg'))} & "
                        f"{pct(mm.get('delta'), True)} & {pct(lb)} & {nf} \\\\")
        for pool, v in s.get("decode_serving", {}).items():
            serving.append(v["changed"])
            if not v.get("same_device", True):
                print(f"  WARNING {label} serving control {pool}: decoded on different GPU models")
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    nums = {"decNonfired": str(nonfired), "decNonfiredChecked": f"{checked:,}".replace(",", "{,}"),
            "decGatedChgMax": pct(max(gated_chg)) if gated_chg else "--",
            "decSeqKsMin": pct(min(seq_ks)) if seq_ks else "--",
            "decServingMax": pct(max(serving)) if serving else "--",
            "decServingMin": pct(min(serving)) if serving else "--"}
    for slug, v in ungated_lb.items():
        nums[f"dec{slug}LbMin"], nums[f"dec{slug}LbMax"] = pct(min(v)), pct(max(v))
    tab = [r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Drift under other decoding rules (Qwen2.5-1.5B, seed 0; the final and the base model share one "
           r"decoding seed and are decoded in the same batches). Fire: firing on held-out Dolly prompts under that "
           r"rule and its 95\% Clopper--Pearson bound, which bounds the TV distance on Dolly under that rule "
           r"(\cref{thm:coupling}); a prompt-time gate's firing is the same under every rule, an anytime gate's is "
           r"not. Chg.: share of changed responses, an estimate of an upper bound on the TV "
           r"distance (coupling inequality). $\Delta$: change of refusal (AdvBench) and accuracy (MMLU), points. "
           r"TV$\ge$: largest 95\% lower confidence bound on the TV distance over the four pools (Dolly, AdvBench, "
           r"XSTest, MMLU), from the Kolmogorov--Smirnov distance of response lengths; for ungated runs it, not Chg., "
           r"is the evidence of drift. Non-fired: changed responses on prompts where no tripwire fired, over the four "
           r"pools (\cref{thm:coupling}: 0). Computed by \texttt{scripts/decode\_report.py}.}",
           r"\label{tab:decode}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{l l r r r r r r r r}", r"\toprule",
           r" & & \multicolumn{2}{c}{Dolly (\%)} & \multicolumn{2}{c}{AdvBench} & \multicolumn{2}{c}{MMLU} & & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Run & Decoding & Fire (CP) & Chg. & Chg. & $\Delta$ & Chg. & $\Delta$ & TV$\ge$ & Non-fired \\",
           r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(a.out, "decode_table.tex"), "w").write("\n".join(tab) + "\n")
    open(os.path.join(a.out, "decode_numbers.tex"), "w").write(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(nums.items())))
    json.dump(out, open(os.path.join(a.out, "decode_summary.json"), "w"), indent=1)
    for k, v in sorted(nums.items()):
        print(k, v)
    print("tables in", a.out)


if __name__ == "__main__":
    main()
