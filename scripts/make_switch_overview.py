#!/usr/bin/env python
"""Overview of the certified switches across experiments (CPU; reads the switch reports, writes paper inputs).

    python scripts/make_switch_overview.py [--results paper/v2_results]

Reads <results>/<exp>/switch/switch_summary.json (scripts/switch_report.py) and writes <results>/overview/:
  switch_overview.tex   Seq-GRPO as trained, behind the deployed gate (eps 0.03) and behind the two-sided gate, for
                        every experiment; the gain kept behind the gate is compared with the prediction beta * G
  gate_overview.tex     firing of the gate variants on held-out pools (known before training)
  switch_pred.dat       predicted vs actual switched gain of every ungated model at every eps (overlap tasks)
  overview_macros.tex   \\ovw<Exp><Family><Variant><Metric> and \\ovg<Exp><Variant><Pool> for the text
"""
from __future__ import annotations

import argparse
import json
import math
import os

EXPS = [  # folder, macro name, label, kind
    ("v2_overlap_q15", "Nq", "NQ, Qwen2.5-1.5B", "overlap"),
    ("v2_overlap_q7", "Nqseven", "NQ, Qwen2.5-7B", "overlap"),
    ("v2_overlap_olmo7", "Nqolmo", "NQ, OLMo-2-7B", "overlap"),
    ("v2_ladder_ifc_q15", "Ifc", "IF, Qwen2.5-1.5B", "overlap"),
    ("v2_ladder_ifc_q7", "Ifcseven", "IF, Qwen2.5-7B", "overlap"),   # v4.1; left out until its switch report exists
    ("v2_ladder_concise_q15", "Concise", "Concise, Qwen2.5-1.5B", "overlap"),
    ("v2_ladder_concisectx_q15", "Concisectx", "Concise on context requests, 1.5B", "overlap"),   # v6 middle rung
    ("v2_stream_q15", "Stream", "3-task stream, 1.5B", "stream"),
    ("v2_stream_q7", "Streamseven", "3-task stream, 7B", "stream"),
    ("v2_stream5_q15", "Streamfive", "5-task stream, 1.5B", "stream"),
]
SHORT = {"Nq": "NQ, 1.5B", "Nqseven": "NQ, 7B", "Nqolmo": "NQ, OLMo-2", "Ifc": "IF, 1.5B", "Ifcseven": "IF, 7B",
         "Concise": "Concise, 1.5B", "Concisectx": "Concise ctx, 1.5B", "Stream": "3 tasks, 1.5B", "Streamseven": "3 tasks, 7B",
         "Streamfive": "5 tasks, 1.5B"}   # experiment labels of the gate table
FAMS = {"seq": "Seq", "kl": "Kl", "refkl": "Refkl", "nullspace": "Null", "ewc": "Ewc", "agem": "Agem",
        "olora": "Olora", "replay": "Replay", "tripwire_prompt": "Twp", "tripwire_prompt_seq": "Tws",
        "tripwire_prompt_e0.10": "Twpb", "tripwire_prompt_seq_e0.10": "Twsb", "router": "Router", "tripwire": "Tw",
        "refkl_b0.1": "Refkla", "refkl_b10": "Refklc", "kl_b1": "Kld", "ewc_l0.1": "Ewca", "ewc_l10": "Ewcc",
        "refkl_b10s4": "Refkld", "refkl_b100s4": "Refkle", "nullspace_rf25": "Nullrf",   # v5 baselines
        "seq_dense5e6": "Seqdense", "seq_dense2e6": "Seqdensetwo", "seq_full5e6": "Seqfull",   # v7 pilot
        "tripwire_prompt_dense5e6": "Twpdense", "seq_sft": "Seqsft", "tripwire_prompt_sft": "Twpsft"}
VARS = {"ratio|0.03": "RatioA", "ratio|0.1": "RatioB", "ratio|0.3": "RatioC", "twosided|0.03": "TwoA",
        "twosided|0.1": "TwoB", "ratio_union|0.03": "UnionA", "twosided_union|0.03": "TwounionA",
        "remedy|0.03": "RemedyA", "remedy|0.1": "RemedyB", "protect|0.03": "ProtectA", "protect|0.1": "ProtectB",
        "twosided_remedy|0.03": "TworemA", "twosided_remedy|0.1": "TworemB", "router|": "Router"}
POOLS = {"dolly": "Dolly", "advbench": "Adv", "xstest": "Xs", "mmlu": "Mmlu", "dolly_sys": "Sys",
         "dolly_turn": "Turn", "advbench_sys": "Advsys", "advbench_turn": "Advturn", "jb_prefix": "Jbp",
         "jb_roleplay": "Jbr", "jb_hypo": "Jbh"}
JB = ("jb_prefix", "jb_roleplay", "jb_hypo")   # AdvBench requests in jailbreak templates (v5.2)


def pct(v, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return s.replace("-", "$-$", 1) if signed else s


def m(a):
    return None if a is None else a[0]


def task_metric(summ, kind):
    return "task:acc" if kind == "stream" else f"{summ['tasks'][0]}:correct"


def wrap_mean(d, key):
    """mean over the harmful-in-task-format pools of metric key (refused change or firing)"""
    vals = [m(v) for k, v in d.items() if k.startswith("advwrap_") and k.endswith(":" + key) and m(v) is not None]
    return sum(vals) / len(vals) if vals else None


def predicted(summ, fam, var, kind):
    """gain kept behind a gate if non-fired new-task prompts gain as much as the others: beta * G per task"""
    F = summ["families"][fam]
    pw = summ["gates"].get(var, {}).get("power_pre", {})
    if kind == "stream":
        parts = []
        for t in summ["tasks"]:
            acc, d = m(F["own"].get(f"{t}:correct_abs")), m(F["own"].get(f"{t}:correct"))
            if acc is None or d is None or pw.get(t) is None:
                return None
            parts.append(acc - (1 - pw[t]) * d)
        return sum(parts) / len(parts)
    g = m(F["own"].get(task_metric(summ, kind)))
    t = summ["tasks"][0]
    return None if g is None or pw.get(t) is None else pw[t] * g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="paper/v2_results")
    args = ap.parse_args()
    out = os.path.join(args.results, "overview")
    os.makedirs(out, exist_ok=True)
    macros, rows, grows, pred_rows = {}, [], [], []
    errs, errs_sep = [], []   # all overlap tasks; without Concise (new prompts drawn from the reference)
    errs_bl = []              # cases with beta < 0.99: (|error|, |measured gain|, experiment)
    cp_dolly = {}             # kind -> 95% bounds on held-out Dolly firing (single tasks: deployed; streams: one budget)
    for folder, name, label, kind in EXPS:
        p = os.path.join(args.results, folder, "switch", "switch_summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        tm = task_metric(s, kind)
        gain_fmt = (lambda v: pct(v)) if kind == "stream" else (lambda v: pct(v, True))
        # families: own and switched metrics
        for fam, fn in FAMS.items():
            F = s["families"].get(fam)
            if F is None:
                continue
            blocks = [("Own", F["own"])] + [(VARS[v], d) for v, d in F["switch"].items() if v in VARS]
            for vn, d in blocks:
                pre = f"ovw{name}{fn}{vn}"
                if d.get(tm) is not None:
                    macros[pre + "Gain"] = gain_fmt(m(d[tm]))
                    macros[pre + "GainSd"] = f"{100 * d[tm][1]:.1f}"
                    macros[pre + "N"] = str(d[tm][2])
                for key, mn, signed in (("dolly:chg", "Chg", False), ("dolly:subst", "Subst", False),
                                        ("mmlu:correct", "Mmlu", True), ("mmlu:chg", "Mmluchg", False),
                                        ("advbench:refused", "Adv", True), ("xstest:refused_safe", "Xs", True),
                                        ("advbench:chg", "Advchg", False)):
                    if d.get(key) is not None:
                        macros[pre + mn] = pct(m(d[key]), signed)
                w = wrap_mean(d, "refused")
                if w is not None:
                    macros[pre + "Wrap"] = pct(w, True)
                if kind == "stream":
                    for t in s["tasks"]:
                        if d.get(f"{t}:correct_abs") is not None:
                            macros[pre + t.capitalize().replace("8k", "Ek")] = pct(m(d[f"{t}:correct_abs"]))
            if F["method"] not in ("tripwire_prompt", "router", "tripwire"):
                for v in ("ratio|0.03", "ratio|0.1", "ratio|0.3", "twosided|0.03"):
                    pr = predicted(s, fam, v, kind)
                    act = m(F["switch"].get(v, {}).get(tm))
                    if pr is None or act is None:
                        continue
                    macros[f"ovw{name}{fn}{VARS[v]}Pred"] = gain_fmt(pr)
                    if kind == "overlap" and v.startswith("ratio"):
                        pred_rows.append((name, fam, v.split("|")[1], 100 * pr, 100 * act))
                        errs.append(abs(pr - act))
                        if name != "Concise":
                            errs_sep.append(abs(pr - act))
                        # Oct 29 (review): where beta = 1 the prediction is the gain itself; report beta < 1 apart
                        b = s["gates"].get(v, {}).get("power_pre", {}).get(s["tasks"][0])
                        if b is not None and b < 0.99:
                            errs_bl.append((abs(pr - act), abs(act), name))
        # Oct 29 (review): the largest 95% bound on held-out Dolly firing of the gates the paper deploys at eps 0.03
        dv = "ratio|0.03" if kind == "overlap" else "ratio_union|0.03"
        cpd = s["gates"].get(dv, {}).get("cp", {}).get("dolly")
        if cpd is not None:
            cp_dolly.setdefault(kind, []).append(cpd)
        # gates
        for v, st in s["gates"].items():
            if v not in VARS:
                continue
            pre = f"ovg{name}{VARS[v]}"
            pw = [x for x in st["power_pre"].values() if x is not None]
            if pw:
                macros[pre + "Power"] = pct(sum(pw) / len(pw))
            for pool, pn in POOLS.items():
                if st["fire"].get(pool) is not None:
                    macros[pre + pn] = pct(st["fire"][pool])
                    macros[pre + "Cp" + pn] = pct(st["cp"][pool])
            wf = [st["fire"][k] for k in st["fire"] if k.startswith("advwrap_") and st["fire"][k] is not None]
            if wf:
                macros[pre + "Wrap"] = pct(sum(wf) / len(wf))
            jf = [st["fire"][k] for k in JB if st["fire"].get(k) is not None]
            if jf:
                macros[pre + "Jb"] = pct(sum(jf) / len(jf))
                macros[pre + "JbMax"] = pct(max(jf))
            for t in s["tasks"]:
                if st["fire"].get(t) is not None:
                    macros[pre + "Task" + t.capitalize().replace("8k", "Ek")] = pct(st["fire"][t])
        # overview rows (Seq-GRPO)
        F = s["families"].get("seq")
        if F is not None:
            sw, tw = F["switch"].get("ratio|0.03", {}), F["switch"].get("twosided|0.03", {})
            pw = [x for x in s["gates"]["ratio|0.03"]["power_pre"].values()]
            pwt = [x for x in s["gates"]["twosided|0.03"]["power_pre"].values()]
            rows.append([label, str(F["n"]),
                         pct(sum(pw) / len(pw)), gain_fmt(m(F["own"].get(tm))), pct(m(F["own"].get("dolly:chg"))),
                         pct(m(F["own"].get("dolly:subst"))),
                         gain_fmt(predicted(s, "seq", "ratio|0.03", kind)), gain_fmt(m(sw.get(tm))),
                         pct(m(sw.get("dolly:chg"))), pct(m(sw.get("dolly:subst"))), pct(m(sw.get("mmlu:chg"))),
                         pct(sum(pwt) / len(pwt)), gain_fmt(m(tw.get(tm))), pct(m(tw.get("dolly:chg"))),
                         pct(m(tw.get("mmlu:chg")))])
        # gate overview rows
        for v, vl in (("ratio|0.03", "probe"), ("twosided|0.03", "+ region"),
                      ("ratio_union|0.03", "one budget"), ("twosided_union|0.03", "+ region, one budget"),
                      ("remedy|0.03", "+ harmful in format"),
                      ("twosided_remedy|0.03", "+ region, + harmful in format"),
                      ("protect|0.03", "protect safety, task-like")):
            st = s["gates"].get(v)
            if st is None:
                continue
            pw = [x for x in st["power_pre"].values() if x is not None]
            wf = [st["fire"][k] for k in st["fire"] if k.startswith("advwrap_")]
            jf = [st["fire"][k] for k in JB if st["fire"].get(k) is not None]
            grows.append([SHORT.get(name, label) if v == "ratio|0.03" else "", vl, pct(sum(pw) / len(pw)),
                          pct(st["fire"].get("dolly")), pct(st["cp"].get("dolly")), pct(st["fire"].get("xstest")),
                          pct(st["fire"].get("mmlu")), pct(sum(wf) / len(wf)) if wf else "--",
                          pct(st["fire"].get("dolly_sys")), pct(sum(jf) / len(jf)) if jf else "--"])
    if errs:
        macros["ovwPredN"] = str(len(errs))
        macros["ovwPredMae"] = f"{100 * sum(errs) / len(errs):.2f}"
        macros["ovwPredMax"] = f"{100 * max(errs):.1f}"
    if errs_sep:
        macros["ovwPredMaeSep"] = f"{100 * sum(errs_sep) / len(errs_sep):.2f}"
    if errs_bl:
        e = [x[0] for x in errs_bl]
        macros["ovwPredBlN"] = str(len(e))
        macros["ovwPredBlExps"] = str(len({x[2] for x in errs_bl}))
        macros["ovwPredBlMae"] = f"{100 * sum(e) / len(e):.2f}"
        macros["ovwPredBlMax"] = f"{100 * max(e):.1f}"
        macros["ovwPredBlGain"] = f"{100 * sum(x[1] for x in errs_bl) / len(errs_bl):.1f}"   # mean |measured gain|
        macros["ovwPredOneN"] = str(len(errs) - len(errs_bl))
    if cp_dolly.get("overlap"):
        macros["ovwCpDollyMax"] = pct(max(cp_dolly["overlap"]))
    if cp_dolly.get("stream"):
        macros["ovwCpDollyMaxStream"] = pct(max(cp_dolly["stream"]))
    # v6 (review): the points share gates, and most have small gains; report the relative error where the gain behind
    # the gate is at least 1 point, and how many distinct gates (experiment, eps) the points come from
    big = [abs(r[3] - r[4]) / abs(r[4]) for r in pred_rows if abs(r[4]) >= 1.0]
    if big:
        big.sort()
        macros["ovwPredRelN"] = str(len(big))
        macros["ovwPredRelMed"] = f"{100 * big[len(big) // 2]:.1f}"
        macros["ovwPredRelMax"] = f"{100 * big[-1]:.1f}"
    if pred_rows:
        macros["ovwPredGates"] = str(len({(r[0], r[2]) for r in pred_rows}))
        macros["ovwPredExps"] = str(len({r[0] for r in pred_rows}))
    with open(os.path.join(out, "overview_macros.tex"), "w") as fh:
        fh.write("% generated by scripts/make_switch_overview.py\n")
        for k in sorted(macros):
            if k.replace("ovw", "").replace("ovg", "").isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
    with open(os.path.join(out, "switch_pred.dat"), "w") as fh:
        fh.write("exp fam eps pred actual\n")
        for r in pred_rows:
            fh.write(f"{r[0]} {r[1]} {r[2]} {r[3]:.2f} {r[4]:.2f}\n")
    cols = (r"Experiment & $n$ & $\beta$ & Gain & Chg. & Subst. & Pred. & Gain & Chg. & Subst. & MMLU & $\beta$ & Gain & "
            r"Chg. & MMLU \\")
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Seq-GRPO behind the certified switch at $\varepsilon=0.03$, every experiment, three seeds. "
             r"A prompt on which the gate fires is answered by the trained model, every other prompt by the base "
             r"model. $\beta$: firing on unlabeled RL prompts of the new task(s), known before training; Pred.: the "
             r"gain kept if missed prompts gain as much as the others ($\beta$ times the gain, per task); Gain: new-task "
             r"accuracy gain in points (streams: average accuracy); Chg.\ and Subst.: share of held-out Dolly responses "
             r"that differ from the base model's, and that a judge rates as substantively different; MMLU: share of "
             r"MMLU responses that change (MMLU is never used in calibration); --: not judged.}",
             r"\label{tab:switch_overview}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l c r rrr rrrrr rrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Seq-GRPO as trained} & \multicolumn{5}{c}{behind the deployed gate} & "
             r"\multicolumn{4}{c}{behind the two-sided gate} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-11}\cmidrule(lr){12-15}", cols, r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    with open(os.path.join(out, "switch_overview.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    has_jb = any(r[-1] != "--" for r in grows)
    if not has_jb:
        grows = [r[:-1] for r in grows]
    gl = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
          r"\caption{Firing of gate variants on held-out prompts (\%, $\varepsilon=0.03$), computed from unlabeled "
          r"prompts before any training; it bounds the drift of every model behind the gate. Probe: the deployed "
          r"prompt-time tripwire; + region: the two-sided tripwire; one budget: the union of a stream's tripwires "
          r"calibrated at $\varepsilon$; + harmful in format: AdvBench requests in the tasks' formats added to the "
          r"reference; protect: calibrated on AdvBench and task-like prompts only. Power: firing on unlabeled new-task prompts; CP: 95\% "
          r"Clopper--Pearson bound on Dolly; Wrap: AdvBench requests in a trained task's format; Sys: Dolly requests "
          r"behind a system prompt" + (r"; JB: AdvBench requests in jailbreak templates (mean of three)" if has_jb
                                       else "") + (r". XSTest, MMLU and JB" if has_jb else r". XSTest and MMLU") + r" are never used in calibration.}",
          r"\label{tab:gate_overview}", r"\resizebox{\linewidth}{!}{%",
          r"\begin{tabular}{l l r rr r r r r" + (" r" if has_jb else "") + "}", r"\toprule",
          r"Experiment & Gate & Power & Dolly & CP & XSTest & MMLU & Wrap & Sys" + (r" & JB" if has_jb else "") + r" \\",
          r"\midrule"]
    for i, r in enumerate(grows):
        if r[0] and i > 0:
            gl.append(r"\midrule")
        gl.append(" & ".join(r) + r" \\")
    gl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(out, "gate_overview.tex"), "w") as fh:
        fh.write("\n".join(gl) + "\n")
    print(f"wrote {out}: {len(rows)} experiments, {len(macros)} macros, {len(pred_rows)} predicted-vs-actual points"
          + (f", mean |error| {100 * sum(errs) / len(errs):.2f} points, max {100 * max(errs):.1f}" if errs else ""))


if __name__ == "__main__":
    main()
