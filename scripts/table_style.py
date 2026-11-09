"""Readability fixes for the generated paper tables (idempotent; Nov 1, paper v52), run by "bash slurm/plan.sh
tables" / "collect" after every generator:

    python scripts/table_style.py paper/v2_results

Tables are only ever shrunk to the line width, never enlarged; shorter row labels ("Seq, scaled", "whole seq.",
"Router (uncal.)"); arrows on the columns where higher or lower is better; a note on why ungated rows have no firing
or certificate and why gated rows have no switch columns; readable names in the judge-agreement table; shorter labels
and headers in the score, per-stage, calibration, sensitivity and overview tables."""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "paper/v2_results"
SHRINK = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
UP, DOWN = r"$\uparrow$", r"$\downarrow$"

FILES = {
    "main": ["v2_stream_q15/results/main_table.tex", "v2_stream_q7/results/main_table.tex",
             "v2_stream5_q15/results/main_table.tex"],
    "overlap": ["v2_overlap_q15/results/overlap_table.tex", "v2_overlap_q7/results/overlap_seven_table.tex",
                "v2_overlap_olmo7/results/overlap_olmo_table.tex", "v2_ladder_ifc_q15/results/overlap_ifc_table.tex",
                "v2_ladder_ifc_q7/results/overlap_ifcseven_table.tex",
                "v2_ladder_concisectx_q15/results/overlap_concisectx_table.tex",
                "v2_ladder_concise_q15/results/overlap_concise_table.tex"],
    "switch": ["v2_overlap_q15/switch/switch_table.tex"],
    "zoo": ["v2_overlap_q15/switch/review/zoo_table.tex", "v2_ladder_concisectx_q15/switch/review/zoo_table.tex"],
    "gates": ["overview/gate_overview.tex"],
    "judge": ["overview/judge_agreement.tex"],
    "cert": ["v2_stream_q15/results/cert_table.tex"],
    "other": ["v2_stream_q15/results/extra_table.tex"],
    "score": ["v2_overlap_q15/analysis/score_table.tex"],
    "stage": ["v2_stream_q15/results/stage_table.tex", "v2_stream_q7/results/stage_table.tex",
              "v2_stream5_q15/results/stage_table.tex"],
    "calib": ["v2_stream_q15/results/calib_table.tex", "v2_stream_q7/results/calib_table.tex",
              "v2_stream5_q15/results/calib_table.tex"],
    "names": ["overview/switch_overview.tex", "overview/review2/covtask_table.tex", "overview/review2/covcons_table.tex"],
    "sens": ["overview/review2/sens_table.tex"],
}


def sub(s, a, b, count=0):
    return s.replace(a, b) if count == 0 else s.replace(a, b, count)


ROUTER_NOTE = (r" The uncalibrated router has a held-out firing rate but no level chosen before training (uncal.).")


def uncal(s):
    """the router's certificate cell: it has a firing rate, but its level is set by the logit-0 rule"""
    s = re.sub(r"(?m)^(Router \(uncal\.\).*& )none( \\\\)$", r"\1uncal.\2", s)
    return s


def common(s):
    s = s.replace(r"\resizebox{\linewidth}{!}{%", SHRINK).replace(r"\resizebox{\textwidth}{!}{%", SHRINK)
    return s


def main_table(s):
    s = uncal(s)
    s = sub(s, "The last column is the label-free certificate:", "The last column is the certificate:")
    s = sub(s, "; the other methods have none.", "; ungated methods have none, and the router's level is uncalibrated (uncal.).")
    s = sub(s, "; ungated methods have none, and the router's level is uncalibrated (uncal.).",
            "; ungated methods have none, and the router has a held-out bound but no level chosen before training (uncal.).")
    if "Router (uncal.)" not in s:   # no router row (five-task stream)
        s = sub(s, "; ungated methods have none, and the router has a held-out bound but no level chosen before training "
                "(uncal.).", "; ungated methods have none.")
    s = sub(s, r"\method{}-prompt, whole sequence (ours)", r"\quad whole sequence")
    s = sub(s, "Router (uncalibrated)", "Router (uncal.)")
    s = sub(s, r"\quad + Null-space LoRA", r"\quad + Null-space")
    s = sub(s, r"\quad adapters $\times$0.5", r"\quad adapter $\times$0.5")
    s = sub(s, r"{Stream tasks} &", r"{Stream tasks " + UP + "} &")
    s = sub(s, r"& Avg & BWT & Refusal & Over-ref. & MMLU & Chg. & Leak & drift (\%) \\",
            r"& Avg" + UP + r" & BWT" + UP + r" & Refusal & Over-ref. & MMLU & Chg." + DOWN + r" & Leak" + DOWN
            + r" & drift (\%)" + DOWN + r" \\")
    s = sub(s, r"Mean $\pm$ standard deviation over $n$ seeds (of 3; training randomness; same data and evaluation prompts); "
            r"the certificate is the largest over seeds. Rows marked $^{\mathrm{A}}$ ran on NVIDIA A100 GPUs (in at least one "
            r"seed), the others on H200; each run is compared with the base model evaluated on its own GPU model.",
            r"Mean $\pm$ standard deviation over $n$ of 3 seeds, with the same data and evaluation prompts; the certificate "
            r"is the largest over seeds. $^{\mathrm{A}}$: run on A100 GPUs in at least one seed, the others on H200, each "
            r"compared with the base model on its own GPU model.")
    if "Arrows:" not in s:
        s = sub(s, r"each run is compared with the base model evaluated on its own GPU model.}",
                r"each run is compared with the base model evaluated on its own GPU model. Arrows: higher ($\uparrow$) or "
                r"lower ($\downarrow$) is better; for changes relative to the base model, zero is best.}", 1)
    return s


REF_POOLS, OFF_POOLS = ("Dolly", "AdvBench"), ("XSTest", "MMLU")


def cert_cells(path):
    """Largest Clopper-Pearson bounds of the stream's cert_table.tex over the reference pools and over the pools never
    used in calibration: {"prompt": (ref, off), "anytime": (ref, off)}."""
    p = os.path.join(os.path.dirname(path), "cert_table.tex")
    if not os.path.exists(p):
        return None
    cp = {"prompt": {}, "anytime": {}}
    for line in open(p).read().split("\n"):
        c = [x.strip() for x in line.rstrip("\\ ").split("&")]
        if len(c) < 7 or not c[1].startswith(r"$\pi_0$"):
            continue
        pool = c[0].replace(r"$^\ast$", "")
        cp["prompt"][pool] = float(c[4])
        if len(c) >= 10:
            cp["anytime"][pool] = float(c[7])
    out = {}
    for k, d in cp.items():
        if all(q in d for q in REF_POOLS + OFF_POOLS):
            out[k] = (max(d[q] for q in REF_POOLS), max(d[q] for q in OFF_POOLS))
    return out


def cert_split(s, path):
    """Nov 1 (positioning): the main tables' certificate column becomes two, the bound over the reference pools and the
    bound over the pools never used in calibration, so that the off-reference MMLU bound is not read as the method's
    certificate on the reference."""
    if r"& drift (\%)" not in s:
        return s
    cells = cert_cells(path)
    if not cells:
        return s
    s = re.sub(r"(\\begin\{tabular\}\{[^}]*) c\}", r"\1 cc}", s, count=1)
    s = sub(s, r"\multicolumn{2}{c}{Dolly (\%)} & Certified \\",
            r"\multicolumn{2}{c}{Dolly (\%)} & \multicolumn{2}{c}{Cert. (\%)} \\")
    last = max(int(b) for a, b in re.findall(r"\\cmidrule\(lr\)\{(\d+)-(\d+)\}", s))
    s = s.replace(f"\\cmidrule(lr){{{last - 1}-{last}}}\n",
                  f"\\cmidrule(lr){{{last - 1}-{last}}}\\cmidrule(lr){{{last + 1}-{last + 2}}}\n", 1)
    s = sub(s, r"& drift (\%)" + DOWN + r" \\", r"& ref." + DOWN + r" & off-ref." + DOWN + r" \\")
    body_a, body_b = s.index(r"\midrule"), s.index(r"\bottomrule")
    rows = s[body_a:body_b].split("\n")
    for i, r in enumerate(rows):
        if not r.endswith(r" \\") or "&" not in r:
            continue
        head, last_cell = r[:-3].rsplit("&", 1)
        v = last_cell.strip()
        if v in ("none", "uncal."):
            new = r"\multicolumn{2}{c}{" + v + "}"
        elif v == "0.0":
            new = "0.0 & 0.0"
        elif v.startswith(r"$\le$"):
            k = "anytime" if "anytime" in head else "prompt"
            if k not in cells:
                continue
            new = r"$\le$%.1f & $\le$%.1f" % cells[k]
        else:
            continue
        rows[i] = head + "& " + new + r" \\"
    s = s[:body_a] + "\n".join(rows) + s[body_b:]
    s = re.sub(r"The last column is the certificate: the largest 95\\% Clopper--Pearson bound on the firing over the four "
               r"evaluation pools \(Dolly, AdvBench, XSTest, MMLU; ",
               lambda _: r"The last two columns are the certificate, the largest 95\% Clopper--Pearson bound on the firing "
               r"over the reference pools Dolly and AdvBench (ref.) and over XSTest and MMLU, which are never used in "
               r"calibration (off-ref.), where MMLU sets it (", s)
    return s


def cert_v3(s):
    """Paper v3: the certificate on a reference component is the PAC level of each stage (Theorem 3.7), set before
    training; the held-out Clopper-Pearson bounds in the last two columns check it on the reference pools and are the
    only certificate on pools never calibrated on."""
    s = sub(s, r"\multicolumn{2}{c}{Cert. (\%)} \\", r"\multicolumn{2}{c}{Held-out bound (\%)} \\")
    s = sub(s, r"The last two columns are the certificate, the largest 95\% Clopper--Pearson bound on the firing over the "
            r"reference pools Dolly and AdvBench (ref.) and over XSTest and MMLU, which are never used in calibration "
            r"(off-ref.), where MMLU sets it (",
            r"Each stage's gate is certified before training at the PAC level 3\% on every reference component "
            r"(\cref{thm:calib}). The last two columns are held-out bounds, the largest 95\% Clopper--Pearson bound on "
            r"the firing over held-out prompts of the reference pools Dolly and AdvBench (ref.), which checks that level, "
            r"and of XSTest and MMLU, which are never used in calibration (off-ref.) and have no other certificate; MMLU "
            r"sets it (")
    s = sub(s, r"; ungated methods have none, and the router has a held-out bound but no level chosen before training "
            r"(uncal.).", r"; ungated methods have no gate (none), and the router has held-out bounds but no level chosen "
            r"before training (uncal.).")
    s = sub(s, "; ungated methods have none.", "; ungated methods have no gate (none).")
    s = sub(s, "the certificate is the largest over seeds.", "each bound is the largest over seeds.")
    return s


def overlap(s):
    s = sub(s, r"Seq, adapter $\times\alpha$ $\alpha{=}", r"Seq, scaled $\alpha{=}")
    # notation (Nov 1 review): alpha is the region level, beta the coverage; the adapter scale is a factor, the KL
    # coefficient lambda_KL
    s = re.sub(r"Seq, scaled \$\\alpha\{=\}([0-9.]+)\$", r"Seq, scaled $\\times\1$", s)
    s = re.sub(r"(\+ (?:Ref-)?KL(?:, 4 samples)? )\$\\beta\{=\}", r"\1$\\lambda_{\\mathrm{KL}}{=}", s)
    s = sub(s, "against Dolly as the reference;", "against Dolly, one reference component;")
    s = sub(s, r"\textit{Oracle task gate (not label-free)}", r"\textit{Oracle task gate (uses task labels)}")
    s = uncal(s)
    if "Router (uncal.)" in s and "no level chosen before training" not in s:
        s = re.sub(r"(\\caption\{.*?)\}\n\\label", lambda m: m.group(1) + ROUTER_NOTE + "}\n\\label", s, count=1, flags=re.S)
    s = sub(s, r"\method{}-prompt, whole sequence", r"\method{}-prompt, whole seq.")
    s = sub(s, "Concise (context)", "Concise-ctx")
    s = re.sub(r"(Method & \$n\$ & gain) & Chg\. & Leak & ", r"\1" + UP.replace("\\", "\\\\") + " & Chg." +
               DOWN.replace("\\", "\\\\") + " & Leak" + DOWN.replace("\\", "\\\\") + " & ", s)
    s = sub(s, r"& Dolly & Cert. \\", r"& Dolly & Cert." + DOWN + r" \\")
    if "Ungated methods have no tripwire" not in s and "Ungated methods have no gate" not in s:
        s = re.sub(r"(\\caption\{.*?)\}\n\\label", lambda m: m.group(1) + r" Ungated methods have no gate: no "
                   r"firing (--) and no certificate (none). Rows with $n=1$ have no standard deviation.}" + "\n\\label",
                   s, count=1, flags=re.S)
    return s


def switch(s):
    s = sub(s, r"Seq, adapter $\times\alpha$ (a", r"Seq, scaled (")
    s = sub(s, "(whole sequence", "(whole seq.")
    s = sub(s, r"& Gain$^{\mathrm{sw}}$ & Chg. & Chg.$^{\mathrm{sw}}$ &",
            r"& Gain$^{\mathrm{sw}}$" + UP + r" & Chg. & Chg.$^{\mathrm{sw}}$" + DOWN + " &")
    if r"\emph{Gated methods" not in s:
        ncol = len(re.search(r"\\begin\{tabular\}\{([^}]*)\}", s).group(1).replace(" ", ""))
        s = re.sub(r"\n(Router \(uncal\.\))", "\n\\\\midrule\n\\\\multicolumn{" + str(ncol) + r"}{l}{\\emph{Gated methods: "
                   r"each already answers only where its own gate fires, so the switch columns do not apply (--).}} \\\\" +
                   r"\n\1", s, count=1)
    s = sub(s, "(L14)", r"($\lg{=}15$)")
    s = sub(s, "(L4)", r"($\lg{=}5$)")
    s = sub(s, "; L, split block;", r"; $\lg$, split block;")
    s = sub(s, "Gated rows are shown as trained.",
            r"Gated rows, below the line, are shown as trained. Tags: a, adapter scale; b, KL or Ref-KL coefficient "
            r"(s4: four samples per reference prompt); e, null-space energy share or $\varepsilon$; l, EWC weight; L, "
            r"split block; rf25, null space refreshed every 25 steps.")
    return s


def zoo(s):
    s = sub(s, "Expert Gate (PCA autoencoders)", "PCA autoencoders, Expert-Gate style")
    s = sub(s, r"Gate & AUROC & Threshold & Power & Dolly & CP &",
            r"Gate & AUROC & Threshold & Power" + UP + r" & Dolly" + DOWN + r" & CP" + DOWN + " &")
    return s


def gates(s):
    s = re.sub(r"(?m)^([^&\n]*& )probe( &)", r"\1deployed\2", s)
    s = sub(s, "Probe: the deployed prompt-time tripwire;", "Deployed: the deployed prompt-time gate;")
    s = sub(s, r"Experiment & Gate & Power & Dolly & CP &",
            r"Experiment & Gate & Power" + UP + r" & Dolly" + DOWN + r" & CP" + DOWN + " &")
    return s


JUDGE_EXP = {"Nq": "NQ, 1.5B", "Ifc": "IF, 1.5B", "Concise": "Concise, 1.5B", "Stream": "3 tasks, 1.5B",
             "Seven": "NQ, 7B", "Ifcseven": "IF, 7B"}
JUDGE_POOL = {"advbench": "AdvBench", "dolly": "Dolly", "xstest": "XSTest"}


def judge(s):
    out = []
    for line in s.split("\n"):
        parts = line.split(" & ")
        if len(parts) > 2 and parts[0] in JUDGE_EXP:
            parts[0] = JUDGE_EXP[parts[0]]
        if len(parts) > 2 and parts[1] in JUDGE_POOL:
            parts[1] = JUDGE_POOL[parts[1]]
        out.append(" & ".join(parts))
    s = "\n".join(out)
    s = sub(s, r"\multicolumn{2}{c}{D vs rest}", r"\multicolumn{2}{c}{DIFFERENT vs rest}")
    return s


def score(s):
    s = sub(s, "Gaussian density ratio (LDA)", "Gaussian ratio (LDA)")
    s = sub(s, "Mahalanobis to reference (one-class)", "Mahalanobis, one-class")
    s = sub(s, r"Text only (hashed $n$-grams)", r"Hashed text $n$-grams")
    s = sub(s, r"Score & $L$ & AUROC & $\varepsilon{=}0.03$ & $0.1$ & $0.3$ & Dolly & MMLU \\",
            r"Score & $L$ & AUROC" + UP + r" & $\varepsilon{=}0.03$ & $0.1$ & $0.3$ & Dolly" + DOWN + r" & MMLU" + DOWN
            + r" \\")
    s = sub(s, r"\multicolumn{3}{c}{Fires on NQ (\%)}", r"\multicolumn{3}{c}{Fires on NQ (\%) " + UP + "}")
    return s


def cert(s):
    # the variant names span three narrow columns; shorter, so the columns are not stretched
    s = re.sub(r"\\multicolumn\{3\}\{c\}\{\\method\{\} (prompt-time|anytime) \(\$n\{=\}(\d+)\$\)\}",
               r"\\multicolumn{3}{c}{\1, $n{=}\2$}", s)
    return s


def stage(s):
    s = sub(s, r"\method{}-prompt, whole sequence (ours)", r"\quad whole sequence")
    s = sub(s, "Router (uncalibrated)", "Router (uncal.)")
    s = sub(s, r"\quad + Null-space LoRA", r"\quad + Null-space")
    s = sub(s, r"\quad adapters $\times$0.5", r"\quad adapter $\times$0.5")
    return s


def calib(s):
    s = sub(s, r"& Certified $\varepsilon$ per component (\%) & Activation (\%) & Median pos. \\",
            r"& Cert.\ $\varepsilon$ per comp.\ (\%) & Fires (\%) & Median pos. \\")
    s = sub(s, "activation (share of held-out new-task prompts on which the tripwire fires)",
            "Fires: share of held-out new-task prompts on which the gate fires")
    return s


def names(s):
    s = sub(s, "Concise on context requests, ", "Concise-ctx, ")
    # Nov 1: on OLMo-2 NQ both the firing and the changed share are 9 of 400 = 2.25%; the firing prints as 2.2 (Table 1),
    # the mean changed share as 2.3, which reads as more changes than firings. Print both as 2.2.
    s = sub(s, r"NQ, OLMo-2-7B & 3 & 96.8 & +5.9 & 91.2 & 14.6 & +5.7 & +5.7 & 2.3 & 1.2 & 0.0 & 95.0 & +5.6 & 2.3 & 0.0 \\",
            r"NQ, OLMo-2-7B & 3 & 96.8 & +5.9 & 91.2 & 14.6 & +5.7 & +5.7 & 2.2 & 1.2 & 0.0 & 95.0 & +5.6 & 2.2 & 0.0 \\")
    s = s.replace(r"$\beta G$", r"$\beta\bar G$").replace(r"& $G$ &", r"& $\bar G$ &").replace(r"$G$: Seq-GRPO", r"$\bar G$: Seq-GRPO")
    s = s.replace(r"Kept $-\,\beta G$", r"Kept $-\,\beta\bar G$")
    s = sub(s, "(known before training)", "(estimated before training)")
    s = sub(s, r"($\beta$ held fixed, as it is known before training)", r"($\beta$ held fixed, as it is estimated before training)")
    if "Worst / best" in s:   # covcons: its beta is the firing on held-out prompts
        s = sub(s, r"Experiment & Task & $\beta$ & ", r"Experiment & Task & $\beta_{\mathrm{eval}}$ & ")
        if "firing on held-out new-task prompts. Tasks" not in s:
            s = sub(s, "Tasks with $\\beta<1$.", r"$\beta_{\mathrm{eval}}$: firing on held-out new-task prompts. Tasks with $\beta<1$.")
        s = sub(s, r"($=\beta G$ in expectation)", r"($=\beta\bar G$ in expectation)")
        s = sub(s, "oracle on each prompt's gain (not label-free)", "oracle on each prompt's gain, which needs labels")
    return s


def sens(s):
    return sub(s, r"paper: $\alpha$ 0.02, shrinkage 0.1, all n &", r"paper's setting &")


FIX = {"main": main_table, "overlap": overlap, "switch": switch, "zoo": zoo, "gates": gates, "judge": judge,
       "other": lambda s: s, "cert": cert, "score": score, "stage": stage, "calib": calib, "names": names, "sens": sens}

# Paper v3.4 (Nov 2): the paper never says "tripwire" (Zhao et al. 2026 use the name); the component is a "gate".
# Printed text only: control sequences, labels, references, citations, paths and comments are kept. Run last.
PROTECT = re.compile(
    r"\\(?:label|ref|eqref|pageref|autoref|[cC]ref|cite[a-z]*|input|include|IfFileExists|inputif|inputor)"
    r"\*?(?:\[[^\]]*\])*\{[^}]*\}|\\[A-Za-z@]+|(?<!\\)%[^\n]*|[A-Za-z0-9_./]*\.(?:tex|dat|json|csv)")


def no_tripwire(s):
    out, last = [], 0
    for m in PROTECT.finditer(s):
        out.append(re.sub(r"\bTripwire", "Gate", re.sub(r"\btripwire", "gate", s[last:m.start()])))
        out.append(m.group(0))
        last = m.end()
    out.append(re.sub(r"\bTripwire", "Gate", re.sub(r"\btripwire", "gate", s[last:])))
    return "".join(out)


n = 0
for kind, files in FILES.items():
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            print("missing", f)
            continue
        s0 = open(p).read()
        s = FIX[kind](common(s0))
        if kind == "main":
            s = cert_v3(cert_split(s, p))
        if s != s0:
            open(p, "w").write(s)
            n += 1
print("changed", n, "files")
def cp_check(s):
    """Paper v3.4: the certificate is the calibrated level; a held-out Clopper-Pearson bound is its check (CP)."""
    s = s.replace(r"& Cert.$\downarrow$ \\", r"& CP$\downarrow$ \\").replace(r"& AUROC & Cert. \\", r"& AUROC & CP \\")
    return s.replace(r"Cert.: largest 95\% Clopper--Pearson bound on the Dolly firing rate over seeds (\cref{thm:coupling}).",
                     r"CP: largest held-out 95\% Clopper--Pearson bound on the Dolly firing rate over seeds (\cref{prop:cp}), "
                     r"a check of the certified level of 3\% (\cref{thm:calib}).")


n = 0
for d, _, names_ in os.walk(ROOT):
    for f in names_:
        if f.endswith(".tex"):
            p = os.path.join(d, f)
            s0 = open(p).read()
            s = cp_check(no_tripwire(s0))
            if s != s0:
                open(p, "w").write(s)
                n += 1
print("renamed tripwire -> gate in", n, "files")
