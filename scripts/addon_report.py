#!/usr/bin/env python
"""Overview tables of the add-on plans, kept apart from the main paper tables (CPU only).

    python scripts/addon_report.py --plan sft --results paper/sft_results --runs runs
    python scripts/addon_report.py --plan fam --results paper/fam_results --runs runs
    python scripts/addon_report.py --plan nov --results paper/nov_results --runs runs

Reads <results>/<exp>/switch/switch_summary.json (scripts/switch_report.py, run by "bash slurm/plan.sh sft-tables" /
"fam-tables" for the add-on experiments and their reference experiments) and the runs' evaluation files, and writes
<results>/overview/addon_<plan>.{md,tex,json}. For --plan sft also <results>/overview/pilot_tabular.tex (Nov 4): the
tabular of the paper's Table "Full-rank updates and SFT" (IF and NQ, Qwen2.5-1.5B) with the number of seeds and the
standard deviation of the gain over seeds, for main.tex to read once the general-claim runs of slurm/p3_general.sh
are in.

sft (PLAN_SFT.md): per task and model, the LoRA GRPO reference runs, the full-rank GRPO and LoRA SFT runs of the v7
  pilot (IF), and the full-rank SFT runs: task gain, Dolly changed, Dolly changed in substance (LLM judge), MMLU
  change, the same model behind the deployed tripwire (eps = the config's), and for gated runs the number of
  non-fired evaluation responses that differ from the base model (0 by Theorem 3.2).
fam (PLAN_SCALE.md): one row per model and task (the existing Qwen2.5-1.5B / 7B and OLMo-2-7B rows first): Seq-GRPO
  gain and Dolly changed, Seq-GRPO behind the deployed tripwire, the tripwire's firing beta on held-out task prompts
  and the prediction beta * G, Tripwire-prompt gain, Dolly changed and non-fired changes.
nov (PLAN_NOV.md): the fam table's columns for every model and task of the switch table, the two larger models
  (Llama-2-13B-Chat, Qwen2.5-14B) and the code task, plus Ref-KL's gain and Dolly changed where it ran.
Missing experiments or runs are left out (or shown as --).
"""
from __future__ import annotations

import argparse
import json
import os

FAM_MODELS = [("llama3b", "Llama-3.2-3B"), ("ds7", "DeepSeek-LLM-7B"), ("llama8", "Llama-3.1-8B"),
              ("mistral7", "Mistral-7B-v0.3")]
FAM_EXISTING = [("Qwen2.5-1.5B", {"ifc": "v2_ladder_ifc_q15", "nq": "v2_overlap_q15"}),
                ("Qwen2.5-7B", {"ifc": "v2_ladder_ifc_q7", "nq": "v2_overlap_q7"}),
                ("OLMo-2-7B", {"nq": "v2_overlap_olmo7"})]
SFT_GROUPS = [
    ("NQ, Qwen2.5-1.5B", [
        ("v2_overlap_q15", "seq", "LoRA GRPO: Seq"),
        ("v2_overlap_q15", "tripwire_prompt", "LoRA GRPO: Tripwire-prompt"),
        ("sft_nq_q15", "seq", "full-rank SFT, gold answers: Seq"),
        ("sft_nq_q15", "tripwire_prompt", "full-rank SFT, gold answers: Tripwire-prompt"),
        ("sft_nq_q15", "seq_full", "full-rank SFT, gold answers: Seq, blocks 1-27"),
        ("sft_nq_q15", "seq_teacher", "full-rank SFT, 7B teacher: Seq"),
        ("sft_nq_q15", "tripwire_prompt_teacher", "full-rank SFT, 7B teacher: Tripwire-prompt"),
    ]),
    ("IF, Qwen2.5-1.5B", [
        ("v2_ladder_ifc_q15", "seq", "LoRA GRPO: Seq"),
        ("v2_ladder_ifc_q15", "tripwire_prompt", "LoRA GRPO: Tripwire-prompt"),
        ("v2_ladder_ifc_q15", "seq_dense5e6", "full-rank GRPO (v7): Seq"),
        ("v2_ladder_ifc_q15", "seq_dense2e6", "full-rank GRPO (v7), lr 2e-6: Seq"),
        ("v2_ladder_ifc_q15", "tripwire_prompt_dense5e6", "full-rank GRPO (v7): Tripwire-prompt"),
        ("v2_ladder_ifc_q15", "seq_full5e6", "full-rank GRPO (v7): Seq, blocks 1-27"),
        ("v2_ladder_ifc_q15", "seq_sft", "LoRA SFT, own samples (v7): Seq"),
        ("v2_ladder_ifc_q15", "tripwire_prompt_sft", "LoRA SFT, own samples (v7): Tripwire-prompt"),
        ("sft_ifc_q15", "seq", "full-rank SFT, 7B teacher: Seq"),
        ("sft_ifc_q15", "tripwire_prompt", "full-rank SFT, 7B teacher: Tripwire-prompt"),
        ("sft_ifc_q15", "seq_full", "full-rank SFT, 7B teacher: Seq, blocks 1-27"),
    ]),
    ("NQ, Qwen2.5-7B", [
        ("v2_overlap_q7", "seq", "LoRA GRPO: Seq"),
        ("v2_overlap_q7", "tripwire_prompt", "LoRA GRPO: Tripwire-prompt"),
        ("sft_nq_q7", "seq", "full-rank SFT, gold answers: Seq"),
        ("sft_nq_q7", "tripwire_prompt", "full-rank SFT, gold answers: Tripwire-prompt"),
    ]),
]
GATED = ("tripwire", "tripwire_prompt", "router")
# the paper's Table "Full-rank updates and SFT" (tab:pilot), in its row order: (experiment, run family, row name)
PILOT = [
    ("IF", [
        ("v2_ladder_ifc_q15", "seq", r"IF: LoRA GRPO, Seq-GRPO"),
        ("v2_ladder_ifc_q15", "tripwire_prompt", r"LoRA GRPO, \methodp{}"),
        ("v2_ladder_ifc_q15", "seq_dense5e6", r"full-rank GRPO, Seq-GRPO"),
        ("v2_ladder_ifc_q15", "seq_dense2e6", r"\quad learning rate $2\cdot10^{-6}$"),
        ("v2_ladder_ifc_q15", "seq_full5e6", r"\quad all blocks"),
        ("v2_ladder_ifc_q15", "tripwire_prompt_dense5e6", r"full-rank GRPO, \methodp{}"),
        ("v2_ladder_ifc_q15", "seq_sft", r"LoRA SFT, ungated"),
        ("v2_ladder_ifc_q15", "tripwire_prompt_sft", r"LoRA SFT, \methodp{}"),
        ("sft_ifc_q15", "seq", r"full-rank SFT, 7B teacher, ungated"),
        ("sft_ifc_q15", "seq_full", r"\quad all blocks"),
        ("sft_ifc_q15", "tripwire_prompt", r"full-rank SFT, 7B teacher, \methodp{}"),
    ]),
    ("NQ", [
        ("v2_overlap_q15", "seq", r"NQ: LoRA GRPO, Seq-GRPO"),
        ("sft_nq_q15", "seq", r"full-rank SFT, gold, ungated"),
        ("sft_nq_q15", "seq_full", r"\quad all blocks"),
        ("sft_nq_q15", "tripwire_prompt", r"full-rank SFT, gold, \methodp{}"),
        ("sft_nq_q15", "seq_teacher", r"full-rank SFT, 7B teacher, ungated"),
        ("sft_nq_q15", "tripwire_prompt_teacher", r"full-rank SFT, 7B teacher, \methodp{}"),
    ]),
]


def load_summary(results, exp):
    p = os.path.join(results, exp, "switch", "switch_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mean(a):
    return None if a is None else a[0]


def sd(a):
    return None if a is None or len(a) < 2 else a[1]


def rows_of(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["i"])


def nonfired_changed(runs, exp, fam):
    """(non-fired evaluation responses that differ from the base model, non-fired responses) over the seeds of a gated
    run, from its stage-0 and final evaluation files; None if the run is not gated or recorded no firing"""
    root = os.path.join(runs, exp)
    if not os.path.isdir(root):
        return None
    n_diff = n_all = 0
    found = False
    for sd in sorted(d for d in os.listdir(root) if d.startswith("s") and d[1:].isdigit()):
        md = os.path.join(root, sd, fam)
        cj = os.path.join(md, "config.json")
        if not (os.path.exists(os.path.join(md, "DONE")) and os.path.exists(cj)):
            continue
        info = json.load(open(cj))
        if info.get("method") not in GATED:
            return None
        K = len(info.get("config", {}).get("stream", [])) or 1
        e0, eK = os.path.join(md, "stage0", "eval"), os.path.join(md, f"stage{K}", "eval")
        if not os.path.isdir(eK):
            continue
        for fn in sorted(os.listdir(eK)):
            if not fn.endswith(".jsonl"):
                continue
            rk, r0 = rows_of(os.path.join(eK, fn)), rows_of(os.path.join(e0, fn))
            if not rk or not r0 or len(rk) != len(r0):
                continue
            for a, b in zip(rk, r0):
                fire = a.get("fire") or {}
                if not fire:
                    continue
                found = True
                if all(v is None for v in fire.values()):
                    n_all += 1
                    n_diff += int(a["out"] != b["out"])
    return (n_diff, n_all) if found else None


def pct(v, signed=False):
    if v is None:
        return "--"
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


def family_row(s, fam, runs):
    """metrics of one run family of a switch summary (means over seeds)"""
    if s is None or fam not in s["families"]:
        return None
    f = s["families"][fam]
    e = s["main_eps"]
    sw = f["switch"].get(f"ratio|{e:g}") or f["switch"].get(f"ratio|{e}") or {}
    own = f["own"]
    return {"n": f["n"], "method": f["method"], "gain": mean(own.get("task:gain")), "gain_sd": sd(own.get("task:gain")),
            "acc": mean(own.get("task:acc")),
            "dolly_chg": mean(own.get("dolly:chg")), "dolly_subst": mean(own.get("dolly:subst")),
            "mmlu": mean(own.get("mmlu:correct")),
            "sw_gain": mean(sw.get("task:gain")), "sw_dolly_chg": mean(sw.get("dolly:chg")),
            "nonfired": nonfired_changed(runs, s["exp"], fam)}


def gate_beta(s):
    if s is None:
        return None, None
    g = s["gates"].get(f"ratio|{s['main_eps']:g}") or s["gates"].get(f"ratio|{s['main_eps']}") or {}
    t = s["tasks"][0]
    return (g.get("fire") or {}).get(t), (g.get("fire") or {}).get("dolly")


def nf(v):
    return "--" if v is None else f"{v[0]} / {v[1]}"


# ----------------------------------------------------------------------------- sft
def sft(results, runs):
    out = {"groups": []}
    md = ["| task, model | run | seeds | gain | Dolly changed | Dolly subst. | MMLU | gated: gain | gated: Dolly changed "
          "| non-fired changed |", "|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for label, rows in SFT_GROUPS:
        g = {"label": label, "rows": []}
        for exp, fam, name in rows:
            r = family_row(load_summary(results, exp), fam, runs)
            if r is None:
                continue
            r.update({"exp": exp, "family": fam, "name": name})
            g["rows"].append(r)
            gated_own = r["method"] in GATED
            sw_gain = "(own)" if gated_own else pct(r["sw_gain"], True)
            sw_chg = "(own)" if gated_own else pct(r["sw_dolly_chg"])
            md.append(f"| {label} | {name} | {r['n']} | {pct(r['gain'], True)} | {pct(r['dolly_chg'])} | "
                      f"{pct(r['dolly_subst'])} | {pct(r['mmlu'], True)} | {sw_gain} | {sw_chg} | {nf(r['nonfired'])} |")
            tex.append(f"{label} & {name} & {r['n']} & {pct(r['gain'], True)} & {pct(r['dolly_chg'])} & "
                       f"{pct(r['dolly_subst'])} & {pct(r['mmlu'], True)} & "
                       f"{'--' if gated_own else pct(r['sw_gain'], True)} & {'--' if gated_own else pct(r['sw_dolly_chg'])} & "
                       f"{nf(r['nonfired'])} \\\\")
        if g["rows"]:
            out["groups"].append(g)
            tex.append("\\midrule")
    head = ("\\begin{tabular}{llrrrrrrrr}\n\\toprule\ntask, model & run & seeds & gain & Dolly chg. & Dolly subst. & MMLU & "
            "gated gain & gated Dolly chg. & non-fired chg. \\\\\n\\midrule")
    caption = ("% gain, MMLU: points vs the base model; Dolly changed: % of held-out Dolly responses that differ from the base "
               "model's; subst.: judged a change in substance; gated: the same model behind the deployed prompt-time "
               "tripwire (eps of the config); non-fired chg.: differing / all non-fired evaluation responses of gated runs")
    return out, md, [caption, head] + (tex[:-1] if tex and tex[-1] == "\\midrule" else tex) + ["\\bottomrule", "\\end{tabular}"]


def pilot_tabular(results, runs):
    """the tabular of the paper's tab:pilot: one row per run family, means over seeds (gain with its sd over seeds)"""
    def sx(v, signed=False):
        return pct(v, signed).replace("-", "$-$") if v is not None and v < 0 else pct(v, signed)
    lines = [r"\begin{tabular}{l r rrrr rr}", r"\toprule",
             r" & & & & & & \multicolumn{2}{c}{behind gate} \\", r"\cmidrule(lr){7-8}",
             r"Run & Seeds & Gain$\uparrow$ & Chg.$\downarrow$ & Subst.$\downarrow$ & MMLU & Gain$\uparrow$ & Chg.$\downarrow$ \\",
             r"\midrule"]
    for gi, (_, rows) in enumerate(PILOT):
        if gi:
            lines.append(r"\midrule")
        for exp, fam, name in rows:
            r = family_row(load_summary(results, exp), fam, runs)
            if r is None:
                continue
            g = sx(r["gain"], True)
            if r.get("gain_sd") is not None and r["n"] > 1:
                g += f"{{\\scriptsize$\\pm${100 * r['gain_sd']:.1f}}}"
            gated_own = r["method"] in GATED
            sw = ("--", "--") if gated_own else (sx(r["sw_gain"], True), pct(r["sw_dolly_chg"]))
            lines.append(f"{name} & {r['n']} & {g} & {pct(r['dolly_chg'])} & {pct(r['dolly_subst'])} & "
                         f"{sx(r['mmlu'], True)} & {sw[0]} & {sw[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


# ----------------------------------------------------------------------------- fam
def fam(results, runs):
    out = {"rows": []}
    md = ["| model | task | seeds | Seq gain | Seq Dolly changed | Seq gated gain | Seq gated Dolly changed | beta | "
          "beta*G | gate on Dolly | TW-prompt gain | TW-prompt Dolly changed | non-fired changed |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    models = list(FAM_EXISTING) + [(name, {"ifc": f"fam_ifc_{k}", "nq": f"fam_nq_{k}"}) for k, name in FAM_MODELS]
    for name, exps in models:
        for task in ("ifc", "nq"):
            exp = exps.get(task)
            s = load_summary(results, exp) if exp else None
            if s is None:
                continue
            seq, twp = family_row(s, "seq", runs), family_row(s, "tripwire_prompt", runs)
            beta, dolly_fire = gate_beta(s)
            G = seq["gain"] if seq else None
            row = {"model": name, "task": task, "exp": exp, "beta": beta, "gate_dolly": dolly_fire,
                   "pred": (beta * G) if (beta is not None and G is not None) else None, "seq": seq, "tripwire_prompt": twp}
            out["rows"].append(row)
            q = seq or {}
            t = twp or {}
            cells = [name, "IF" if task == "ifc" else "NQ", str(q.get("n", t.get("n", "--"))),
                     pct(q.get("gain"), True), pct(q.get("dolly_chg")), pct(q.get("sw_gain"), True),
                     pct(q.get("sw_dolly_chg")), pct(beta), pct(row["pred"], True), pct(dolly_fire),
                     pct(t.get("gain"), True), pct(t.get("dolly_chg")), nf(t.get("nonfired"))]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{llrrrrrrrrrrr}\n\\toprule\nmodel & task & seeds & Seq gain & Seq Dolly chg. & gated gain & "
            "gated Dolly chg. & $\\beta$ & $\\beta G$ & gate on Dolly & TW-p gain & TW-p Dolly chg. & non-fired chg. \\\\\n\\midrule")
    caption = ("% LoRA, seed means. Seq: Seq-GRPO; gated: Seq-GRPO behind the deployed prompt-time tripwire; beta: its "
               "firing on held-out task prompts; beta*G: the predicted gated gain; TW-p: Tripwire-prompt trained behind it")
    return out, md, [caption, head] + tex + ["\\bottomrule", "\\end{tabular}"]


# ----------------------------------------------------------------------------- nov
NOV_MODELS = [("llama13", "Llama-2-13B-Chat"), ("qwen14", "Qwen2.5-14B")]


def nov(results, runs):
    out = {"rows": []}
    md = ["| model | task | seeds | Seq gain | Seq Dolly changed | Seq gated gain | Seq gated Dolly changed | beta | "
          "beta*G | gate on Dolly | TW-prompt gain | TW-prompt Dolly changed | non-fired changed | Ref-KL gain | "
          "Ref-KL Dolly changed |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    models = (list(FAM_EXISTING) + [(name, {"ifc": f"fam_ifc_{k}", "nq": f"fam_nq_{k}"}) for k, name in FAM_MODELS]
              + [(name, {"ifc": f"nov_ifc_{k}", "nq": f"nov_nq_{k}"}) for k, name in NOV_MODELS]
              + [("Qwen2.5-1.5B", {"code": "nov_code_q15"})])
    for name, exps in models:
        for task in ("ifc", "nq", "code"):
            exp = exps.get(task)
            s = load_summary(results, exp) if exp else None
            if s is None:
                continue
            seq, twp, rkl = (family_row(s, f, runs) for f in ("seq", "tripwire_prompt", "refkl"))
            beta, dolly_fire = gate_beta(s)
            G = seq["gain"] if seq else None
            row = {"model": name, "task": task, "exp": exp, "beta": beta, "gate_dolly": dolly_fire,
                   "pred": (beta * G) if (beta is not None and G is not None) else None, "seq": seq,
                   "tripwire_prompt": twp, "refkl": rkl}
            out["rows"].append(row)
            q, t, r = seq or {}, twp or {}, rkl or {}
            cells = [name, {"ifc": "IF", "nq": "NQ", "code": "Code"}[task], str(q.get("n", t.get("n", "--"))),
                     pct(q.get("gain"), True), pct(q.get("dolly_chg")), pct(q.get("sw_gain"), True),
                     pct(q.get("sw_dolly_chg")), pct(beta), pct(row["pred"], True), pct(dolly_fire),
                     pct(t.get("gain"), True), pct(t.get("dolly_chg")), nf(t.get("nonfired")),
                     pct(r.get("gain"), True) + (f" ({r['n']})" if r else ""), pct(r.get("dolly_chg"))]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{llrrrrrrrrrrrrr}\n\\toprule\nmodel & task & seeds & Seq gain & Seq Dolly chg. & gated gain & "
            "gated Dolly chg. & $\\beta$ & $\\beta G$ & gate on Dolly & TW-p gain & TW-p Dolly chg. & non-fired chg. & "
            "Ref-KL gain (seeds) & Ref-KL Dolly chg. \\\\\n\\midrule")
    caption = ("% LoRA, seed means. Seq: Seq-GRPO; gated: Seq-GRPO behind the deployed prompt-time tripwire; beta: its "
               "firing on held-out task prompts; beta*G: the predicted gated gain; TW-p: Tripwire-prompt trained behind it")
    return out, md, [caption, head] + tex + ["\\bottomrule", "\\end{tabular}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["sft", "fam", "nov"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.results, "overview")
    os.makedirs(out_dir, exist_ok=True)
    data, md, tex = {"sft": sft, "fam": fam, "nov": nov}[args.plan](args.results, args.runs)
    base = os.path.join(out_dir, f"addon_{args.plan}")
    with open(base + ".json", "w") as f:
        json.dump(data, f, indent=1)
    with open(base + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(base + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    if args.plan == "sft":
        with open(os.path.join(out_dir, "pilot_tabular.tex"), "w") as f:
            f.write("\n".join(pilot_tabular(args.results, args.runs)) + "\n")
        print(f"wrote {os.path.join(out_dir, 'pilot_tabular.tex')} (the paper's Table 21)")
    print("\n".join(md))
    print(f"wrote {base}.md / .tex / .json")


if __name__ == "__main__":
    main()
