#!/usr/bin/env python
"""Report for the overlap experiment (configs/overlap_nq.yaml): what each method gains on the new task and what
it changes on the reference distributions, Tripwire's certificates at every trained operating point, and the
activation-vs-certificate frontier (Theorem 4.1) traced from the post-hoc tripwire scores. No torch needed.

    python scripts/make_overlap_report.py --runs runs/overlap_nq --paper paper

Tripwire runs trained at other certificate levels (e.g. runs/overlap_nq_eps0.10, made with
EXTRA="exp=overlap_nq_eps0.10 tripwire.eps=0.10") are found automatically as <runs>_eps*; --tripwire-runs
overrides the list. KL runs with other coefficients (e.g. runs/overlap_nq_kl0.2, made with
EXTRA="exp=overlap_nq_kl0.2 grpo.kl_beta=0.2" METHODS=kl) are found as <runs>_kl*. Writes
<paper>/results/overlap_table.tex, overlap_numbers.tex (LaTeX macros) and overlap_frontier_{tw,kl,base}.dat
(pgfplots data for the frontier figure).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.certify import clopper_pearson_upper  # noqa: E402

BASELINES = [("seq", "Seq-GRPO", "Seq"), ("kl", "+ KL", "Kl"), ("router", "Router (uncal.)", "Router")]
TW_LETTERS = "ABCDEFGH"  # macro names may not contain digits: operating points sorted by eps are A, B, C, ...


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    out.sort(key=lambda r: r["i"])
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load(mdir, k):
    edir = os.path.join(mdir, f"stage{k}", "eval")
    if not os.path.exists(os.path.join(edir, "DONE")):
        return None
    return {fn[:-6]: rows(os.path.join(edir, fn)) for fn in os.listdir(edir) if fn.endswith(".jsonl")}


def auroc(pos, neg):
    s = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    i, rsum = 0, 0.0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rsum += avg * sum(1 for t in s[i:j] if t[1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def fired(r):
    return r.get("fire", {}).get("1") is not None


def summarize(mdir, gated, task="nq"):
    """All numbers for one method run (stage 0 = its own base model evaluation, stage 1 = after the new task).
    Field names keep the "nq" prefix for the new task whatever it is (nq, ifc, concise)."""
    s0, s1 = load(mdir, 0), load(mdir, 1)
    if s0 is None or s1 is None:
        return None
    if task != "nq":
        s0, s1 = dict(s0), dict(s1)
        s0["nq"], s1["nq"] = s0[task], s1[task]
    acc = lambda st, p: mean(r["m"]["correct"] for r in st[p])  # noqa: E731
    ref = lambda st: mean(r["m"]["refused"] for r in st["advbench"])  # noqa: E731
    over = lambda st: mean(r["m"]["refused"] for r in st["xstest"] if r.get("label") == "safe")  # noqa: E731
    d = dict(nq0=acc(s0, "nq"), nq1=acc(s1, "nq"),
             chg=mean(float(a["out"] != b["out"]) for a, b in zip(s1["dolly"], s0["dolly"])),
             dlen=mean(len(a["out"]) - len(b["out"]) for a, b in zip(s1["dolly"], s0["dolly"])),
             adv=ref(s1) - ref(s0), xs=over(s1) - over(s0), mmlu=acc(s1, "mmlu") - acc(s0, "mmlu"),
             chg_mmlu=mean(float(a["out"] != b["out"]) for a, b in zip(s1["mmlu"], s0["mmlu"])),
             chg_xs=mean(float(a["out"] != b["out"]) for a, b in zip(s1["xstest"], s0["xstest"])))
    d["gain"] = d["nq1"] - d["nq0"]
    # format leakage on ordinary chat: the trained ifc formats (lowercase, closing sentence, ...) or answer tags
    lk = "ifleak" if (task == "ifc" and "ifleak" in s1["dolly"][0]["m"]) else "leak"
    d["leak"] = mean(r["m"][lk] for r in s1["dolly"]) - mean(r["m"][lk] for r in s0["dolly"])
    if gated:
        nq1, nq0 = s1["nq"], s0["nq"]
        d["fire_nq"] = mean(float(fired(r)) for r in nq1)
        d["fire_dolly"] = mean(float(fired(r)) for r in s1["dolly"])
        X, n = sum(fired(r) for r in s1["dolly"]), len(s1["dolly"])
        d["cert"] = clopper_pearson_upper(X, n)
        # gain decomposition (exact for a gated update: non-fired prompts keep the base response):
        # gain = P(fire) * E[accuracy change | fired]
        f_idx = [i for i, r in enumerate(nq1) if fired(r)]
        d["cond_gain"] = mean(nq1[i]["m"]["correct"] - nq0[i]["m"]["correct"] for i in f_idx)
        pos = [nq1[i]["fire"]["1"] for i in f_idx]
        d["in_prompt"] = mean(float(p < 0) for p in pos)
        # Theorem 3.2 check on this run: a non-fired prompt must keep its base response exactly
        same_dev = all(s1[p][0].get("dev") == s0[p][0].get("dev") for p in ("nq", "dolly") if s1[p] and s0[p])
        if same_dev:
            viol = sum(int(a["out"] != b["out"]) for p in ("nq", "dolly", "mmlu", "xstest", "advbench")
                       for a, b in zip(s1[p], s0[p]) if not fired(a))
            d["exact_viol"] = viol
    return d


def pct(v, nd=1):
    return "--" if v is None or v != v else f"{100 * v:.{nd}f}"


def pts(v, nd=1):
    if v is None or v != v:
        return "--"
    r = round(100 * v, nd)
    return f"{r:+.{nd}f}" if r != 0 else f"+{0:.{nd}f}"  # never "-0.0"


def tpts(v, nd=1):
    """pts() for LaTeX: a typographic minus sign"""
    return pts(v, nd).replace("-", "$-$", 1) if pts(v, nd) != "--" else "--"


# ----------------------------------------------------------------------------- v2 layout: seeds and tagged runs
FAMILIES = [("seq", "Seq-GRPO", None), ("scale", r"Seq, adapter $\times\alpha$", ("__alpha__", "", r"\alpha")),
            ("kl", "+ KL", ("grpo", "kl_beta", r"\beta")),
            ("refkl", "+ Ref-KL", ("grpo", "refkl_beta", r"\beta")),
            ("refkl_s", "+ Ref-KL, 4 samples", ("grpo", "refkl_beta", r"\beta")),   # v5: grpo.refkl_samples > 1
            ("nullspace", "+ Null-space", ("grpo", "nullspace_energy", "e")),
            ("nullspace_rf", "+ Null-space, refreshed", ("grpo", "nullspace_energy", "e")),   # v5: nullspace_refresh
            ("ewc", "+ EWC", ("grpo", "ewc_lambda", r"\lambda")), ("mas", "+ MAS", ("grpo", "mas_lambda", r"\lambda")),
            ("l2", "+ L2 to old weights", ("grpo", "l2_lambda", r"\lambda")), ("agem", "+ A-GEM", None),
            ("refkl_protect", "+ Ref-KL, protected components", ("grpo", "refkl_beta", r"\beta")),   # paper v3
            ("safemix", "+ Safety replay", None),                                                      # paper v3
            ("router", "Router (uncal.)", None),
            ("tripwire_prompt", r"\method{}-prompt", ("tripwire", "eps", r"\varepsilon")),
            ("tripwire_prompt_protect", r"\method{}-prompt, protected components", ("tripwire", "eps", r"\varepsilon")),
            ("router_seq", "Router, whole sequence (uncal.)", None),
            ("tripwire_prompt_seq", r"\method{}-prompt, whole sequence", ("tripwire", "eps", r"\varepsilon")),
            ("tripwire", r"\method{}-anytime", ("tripwire", "eps", r"\varepsilon"))]
MACRO = {"seq": "Seq", "scale": "Scale", "kl": "Kl", "refkl": "Refkl", "nullspace": "Null", "ewc": "Ewc", "mas": "Mas", "l2": "Ltwo",
         "refkl_s": "Refkls", "nullspace_rf": "Nullrf",
         "agem": "Agem", "router": "Router",
         "tripwire_prompt": "Twp", "tripwire": "Tw", "router_seq": "Routers", "tripwire_prompt_seq": "Tws",
         "refkl_protect": "Refklp", "safemix": "Safemix", "tripwire_prompt_protect": "Twpp"}
GATED_FAMS = ("tripwire", "tripwire_prompt", "router", "tripwire_prompt_seq", "router_seq", "tripwire_prompt_protect")
CERT_FAMS = ("tripwire", "tripwire_prompt", "tripwire_prompt_seq", "tripwire_prompt_protect")


def _agg(vals):
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd, len(v)


def _fmt(a, signed=False):
    if a is None:
        return "--"
    m, sd, n = a
    core = tpts(m) if signed else pct(m)
    return core if n < 2 else f"{core}{{\\scriptsize$\\pm${100 * sd:.1f}}}"


def seeds_report(root, paper, name=""):
    """name (letters only, e.g. "Ifc"): output files overlap_<name>_*.tex/.dat and macros \\ovl<name>..., so that
    several overlap experiments (NQ, IF, Concise, 7B) can sit in one paper."""
    if name and not name.isalpha():
        raise SystemExit("--name must be letters only (it becomes part of LaTeX macro names)")
    fpre = "overlap" + (f"_{name.lower()}" if name else "")
    mpre = "ovl" + name
    seeds = sorted((d for d in os.listdir(root) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
    runs = {}  # (family, param) -> list of per-seed summaries, at the experiment's own gate layer
    by_gl = {}  # (family, param, gate layer) -> list of per-seed summaries, every gate layer (split-depth study)
    tasks = set()
    gpus = {}   # (family, param) -> GPU models of its runs
    for sd in seeds:
        sroot = os.path.join(root, sd)
        for d in sorted(os.listdir(sroot)):
            md = os.path.join(sroot, d)
            cj = os.path.join(md, "config.json")
            if not os.path.isdir(os.path.join(md, "stage0")) or not os.path.exists(cj):
                continue
            info = json.load(open(cj))
            fam = info.get("method")
            if fam in ("tripwire_prompt", "router") and \
                    info["config"].get("tripwire", {}).get("activation", "decision") == "sequence":
                fam += "_seq"   # whole-sequence activation: its own rows, curves and macros
            gcfg = info["config"].get("grpo", {})
            if fam == "refkl" and int(gcfg.get("refkl_samples", 1)) > 1:
                fam = "refkl_s"         # v5: several samples per reference prompt (not the same runs as refkl)
            if fam == "nullspace" and int(gcfg.get("nullspace_refresh", 0)) > 0:
                fam = "nullspace_rf"    # v5: subspaces recomputed during training
            if fam in ("refkl", "tripwire_prompt") and info["config"].get("tripwire", {}).get("protect"):
                fam += "_protect"       # paper v3: only the protected reference components (tripwire.protect)
            if info["config"].get("lora", {}).get("dense") or gcfg.get("objective", "grpo") != "grpo":
                continue                # v7 pilot (full-rank updates, SFT): reported by switch_report.py only
            spec = next((f for f in FAMILIES if f[0] == fam), None)
            if spec is None or (fam == "scale" and info.get("base_run", "seq") != "seq"):
                continue
            gated = fam in GATED_FAMS
            task = info["config"].get("stream", ["nq"])[0]
            tasks.add(task)
            summ = summarize(md, gated, task)
            if summ is None:
                continue
            if gated and fam not in ("router", "router_seq"):
                tj = os.path.join(md, "stage1", "tripwire", "tripwire.json")
                if os.path.exists(tj):
                    t = json.load(open(tj))
                    ph_new, ph_dolly = t["posthoc"].get("new", []), sorted(t["posthoc"].get("dolly", []))
                    summ["auroc"] = auroc(ph_new, ph_dolly)
            param = None
            if spec[2] is not None:
                param = float(info["alpha"]) if spec[2][0] == "__alpha__" else float(info["config"][spec[2][0]][spec[2][1]])
            by_gl.setdefault((fam, param, int(info["config"].get("gate_layer", 0))), []).append(summ)
            dev = info.get("device") or ""
            gpus.setdefault((fam, param), set()).add(next((g for g in ("H200", "H100", "A100", "V100") if g in dev), dev or "?"))
    if by_gl:
        from collections import Counter
        base_gl = Counter(gl for (_, _, gl), v in by_gl.items() for _ in v).most_common(1)[0][0]
        runs = {(f, p): v for (f, p, gl), v in by_gl.items() if gl == base_gl}
    if not runs:
        print("no finished runs under", root)
        return
    fields = ("gain", "chg", "adv", "xs", "mmlu", "leak", "fire_nq", "fire_dolly", "cert", "cond_gain", "in_prompt",
              "auroc")
    agg = {k: {f: _agg([r.get(f) for r in v]) for f in fields} for k, v in runs.items()}
    order = [(f, p) for f, _, _ in FAMILIES for (ff, p) in sorted(k for k in runs if k[0] == f)]

    mixed = len({g for v in gpus.values() for g in v}) > 1

    def label(fam, param):
        name = next(n for f, n, _ in FAMILIES if f == fam)
        spec = next(s for f, _, s in FAMILIES if f == fam)
        lab = name if spec is None else f"{name} ${spec[2]}{{=}}{param:g}$"
        return lab + (r"$^{\mathrm{A}}$" if mixed and "A100" in gpus.get((fam, param), ()) else "")

    task = sorted(tasks)[0] if len(tasks) == 1 else "/".join(sorted(tasks))
    tname = {"nq": "NQ", "ifc": "IF", "concise": "Concise", "concisectx": "Concise (context)", "code": "Code"}.get(task, task)
    long = {"nq": "Natural Questions asked as plain questions",
            "ifc": "held-out Dolly requests with one verifiable format constraint",
            "concise": "plain held-out Dolly requests, rewarded for short answers",
            "concisectx": "held-out Dolly requests with a context passage, rewarded for short answers",
            "code": "Python functions checked by unit tests (KodCode)"}.get(task, task)
    print(f"{task}: {len(seeds)} seeds, mean (sd) over seeds")
    print(f"| run | n | {tname} gain | Dolly changed | Dolly leak | AdvB refusal | over-ref | MMLU | fires {tname} | "
          "fires Dolly | cert (max) |")
    print("|" + "---|" * 11)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{New task: " + long + r", against Dolly as the reference; mean $\pm$ standard "
           r"deviation over $n$ seeds (of " + str(len(seeds)) + r"). Gain: new-task accuracy (points). Changes relative to each "
           r"run's base model: share of Dolly responses that differ (Chg.), share showing a trained format (Leak), "
           r"AdvBench refusal, XSTest over-refusal and MMLU accuracy (points). Fires: share of prompts on which the "
           r"gate fires. Cert.: largest 95\% Clopper--Pearson bound on the Dolly firing rate over seeds "
           r"(\cref{thm:coupling})." + (r" Rows marked $^{\mathrm{A}}$ ran on A100 GPUs (in at least one seed), the "
           r"others on H200; each run is compared with the base model evaluated on its own GPU model." if mixed else "")
           + "}",
           r"\label{tab:overlap_" + task + "}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l c r rr rrr rr r}",
           r"\toprule",
           r" & & " + tname + r" & \multicolumn{5}{c}{Change on reference} & \multicolumn{2}{c}{Fires (\%)} & \\",
           r"\cmidrule(lr){4-8}\cmidrule(lr){9-10}",
           r"Method & $n$ & gain & Chg. & Leak & Refusal & Over-ref. & MMLU & " + tname + r" & Dolly & Cert. \\", r"\midrule"]
    prev_fam = None
    for fam, param in order:
        a = agg[(fam, param)]
        n = len(runs[(fam, param)])
        cert = max((r["cert"] for r in runs[(fam, param)] if r.get("cert") is not None), default=None)
        certified = fam in CERT_FAMS
        print(f"| {fam}{'' if param is None else f' {param:g}'} | {n} | " + " | ".join(
            "--" if a[f] is None else (f"{100 * a[f][0]:+.1f} ({100 * a[f][1]:.1f})" if f in ("gain", "leak", "adv", "xs", "mmlu")
                                      else f"{100 * a[f][0]:.1f} ({100 * a[f][1]:.1f})")
            for f in ("gain", "chg", "leak", "adv", "xs", "mmlu", "fire_nq", "fire_dolly")) +
              f" | {pct(cert) if certified else 'none'} |")
        if prev_fam is not None and fam != prev_fam and fam in ("router",):
            tex.append(r"\midrule")
        prev_fam = fam
        tex.append(f"{label(fam, param)} & {n} & {_fmt(a['gain'], True)} & {_fmt(a['chg'])} & {_fmt(a['leak'], True)} & "
                   f"{_fmt(a['adv'], True)} & "
                   f"{_fmt(a['xs'], True)} & {_fmt(a['mmlu'], True)} & {_fmt(a['fire_nq'])} & {_fmt(a['fire_dolly'])} & "
                   f"{pct(cert) if certified else 'none'} \\\\")
    oracle = agg.get(("seq", None))
    if oracle is not None and oracle["gain"] is not None:
        # Seq-GRPO's adapter switched on exactly for the new task's prompts: needs the task identity of every prompt,
        # so it is not label-free; computed from Seq-GRPO's evaluations (every other prompt keeps the base output)
        g = oracle["gain"]
        print(f"| oracle task gate | {len(runs[('seq', None)])} | {100 * g[0]:+.1f} ({100 * g[1]:.1f}) | 0.0 | +0.0 | +0.0 | "
              f"+0.0 | +0.0 | 100.0 | 0.0 | none |")
        tex.append(r"\midrule")
        tex.append(r"\textit{Oracle task gate (not label-free)}" + f" & {len(runs[('seq', None)])} & {_fmt(g, True)} & 0.0 & "
                   r"+0.0 & +0.0 & +0.0 & +0.0 & 100.0 & 0.0 & -- \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]

    # macros: one letter per sweep point (sorted by the swept value), means over seeds
    nums = {mpre + "Seeds": str(len(seeds))}
    if oracle is not None and oracle["gain"] is not None:
        nums[mpre + "OracleGain"] = tpts(oracle["gain"][0])
    for fam, _, spec in FAMILIES:
        pts_ = sorted(p for f, p in runs if f == fam)
        for i, param in enumerate(pts_):
            L = "" if spec is None else TW_LETTERS[i]
            a = agg[(fam, param)]
            pre = f"{mpre}{MACRO[fam]}{L}"
            if spec is not None:
                nums[pre + "Param"] = f"{param:g}"
            for f in fields:
                v = a[f]
                if f == "auroc":
                    nums[pre + "Auroc"] = "--" if v is None else f"{v[0]:.2f}"
                    continue
                key = pre + f.replace("_", " ").title().replace(" ", "")
                nums[key] = "--" if v is None else (tpts(v[0]) if f in ("gain", "leak", "adv", "xs", "mmlu", "cond_gain") else pct(v[0]))
                if f in ("gain", "chg") and v is not None:
                    nums[key + "Sd"] = f"{100 * v[1]:.1f}"
            if fam in CERT_FAMS:
                nums[pre + "CertMax"] = pct(max((r["cert"] for r in runs[(fam, param)] if r.get("cert") is not None), default=None))
    res = os.path.join(paper, "results")
    os.makedirs(res, exist_ok=True)
    if name:
        tex = [t.replace(r"\label{tab:overlap_", r"\label{tab:overlap" + name.lower() + "_") for t in tex]
    with open(os.path.join(res, f"{fpre}_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    split_report(by_gl, base_gl, len(seeds), res, fpre, mpre, nums)
    with open(os.path.join(res, f"{fpre}_numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_overlap_report.py (seed folders)\n")
        for k in sorted(nums):
            f.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    # frontier curves: x = Dolly responses changed (%), y = NQ gain (points), with standard deviations
    for fam, fname in (("tripwire", "tw"), ("tripwire_prompt", "twp"), ("tripwire_prompt_seq", "tws"), ("kl", "kl"),
                       ("refkl", "refkl"), ("nullspace", "null"), ("scale", "scale"), ("ewc", "ewc"), ("mas", "mas"),
                       ("l2", "ltwo"), ("refkl_s", "refkls"), ("nullspace_rf", "nullrf")):
        if fam in ("ewc", "mas", "l2", "tripwire_prompt_seq", "refkl_s", "nullspace_rf") and \
                not any(f == fam for f, _ in runs):
            continue
        spec = next(s for f, _, s in FAMILIES if f == fam)
        lines = ["chg gain chg_sd gain_sd label"]
        if fam.startswith("tripwire") or fam == "scale":   # these curves start at the base model
            lines.append("0 0 0 0 {}")
        for param in sorted(p for f, p in runs if f == fam):
            a = agg[(fam, param)]
            lines.append(f"{100 * a['chg'][0]:.2f} {100 * a['gain'][0]:.2f} {100 * a['chg'][1]:.2f} {100 * a['gain'][1]:.2f} "
                         f"${spec[2]}{{=}}{param:g}$")
        with open(os.path.join(res, f"{fpre}_frontier_{fname}.dat"), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(res, f"{fpre}_frontier_base.dat"), "w") as f:
        f.write("chg gain chg_sd gain_sd label\n")
        for fam, lab in (("seq", "Seq"), ("router", "Router"), ("agem", "A-GEM")):
            if (fam, None) in agg:
                a = agg[(fam, None)]
                f.write(f"{100 * a['chg'][0]:.2f} {100 * a['gain'][0]:.2f} {100 * a['chg'][1]:.2f} {100 * a['gain'][1]:.2f} {lab}\n")
    if oracle is not None and oracle["gain"] is not None:   # its own file: a separate marker in the figure
        with open(os.path.join(res, f"{fpre}_frontier_oracle.dat"), "w") as f:
            f.write("chg gain chg_sd gain_sd label\n")
            f.write(f"0.00 {100 * oracle['gain'][0]:.2f} 0.00 {100 * oracle['gain'][1]:.2f} Oracle\n")
    print("\nwrote", res, f"({fpre}_table.tex, {fpre}_numbers.tex, {fpre}_frontier_*.dat)")


def split_report(by_gl, base_gl, n_seeds, res, fpre, mpre, nums):
    """Split-depth study: every (method, setting) that was also run with another gate layer, one row per layer."""
    keys = sorted({(f, p) for (f, p, gl) in by_gl if gl != base_gl})
    if not keys:
        return
    order = {f: i for i, (f, _, _) in enumerate(FAMILIES)}
    rows = sorted(((f, p, gl) for (f, p, gl) in by_gl if (f, p) in keys), key=lambda k: (order[k[0]], k[1] or 0, k[2]))
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Split depth: the tripwire reads the input of block $\lg$ and adapters sit in blocks $\geq \lg$ "
           r"(blocks below $\lg$ are the frozen trunk; blocks counted from 1). Mean $\pm$ standard deviation over up to " + str(n_seeds) +
           r" seeds; columns as in the main overlap table; AUROC of the probe between new-task and Dolly prompts.}",
           r"\label{tab:" + fpre + "_split}", r"\begin{tabular}{l r r r rr r r}", r"\toprule",
           r"Method & $\lg$ & Gain & Chg. & \multicolumn{2}{c}{Fires (\%)} & AUROC & Cert. \\",
           r"\cmidrule(lr){5-6}", r" & & & & New & Dolly & & \\", r"\midrule"]
    print("\nsplit depth")
    for f, p, gl in rows:
        v = by_gl[(f, p, gl)]
        a = {k: _agg([r.get(k) for r in v]) for k in ("gain", "chg", "fire_nq", "fire_dolly", "auroc")}
        cert = max((r["cert"] for r in v if r.get("cert") is not None), default=None)
        certified = f in CERT_FAMS
        name = next(n for ff, n, _ in FAMILIES if ff == f)
        spec = next(s for ff, _, s in FAMILIES if ff == f)
        lab = name if spec is None or p is None else f"{name} ${spec[2]}{{=}}{p:g}$"
        au = "--" if a["auroc"] is None else f"{a['auroc'][0]:.2f}"
        print(f"| {f} {'' if p is None else p} | L={gl} | n={len(v)} | gain {pts(a['gain'][0]) if a['gain'] else '--'} | "
              f"chg {pct(a['chg'][0]) if a['chg'] else '--'} | auroc {au} | cert {pct(cert) if certified else 'none'} |")
        tex.append(f"{lab} & {gl + 1} & {_fmt(a['gain'], True)} & {_fmt(a['chg'])} & {_fmt(a['fire_nq'])} & "
                   f"{_fmt(a['fire_dolly'])} & {au} & {pct(cert) if certified else 'none'} \\\\")
        mac = f"{mpre}Split{MACRO[f]}L{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[gl % 26]}"
        nums[mac + "Gain"] = "--" if a["gain"] is None else tpts(a["gain"][0])
        nums[mac + "Chg"] = "--" if a["chg"] is None else pct(a["chg"][0])
        nums[mac + "Auroc"] = au
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(res, f"{fpre}_split.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/overlap_nq")
    ap.add_argument("--tripwire-runs", nargs="*", default=None,
                    help="experiment dirs whose tripwire/ run is one operating point (default: --runs and <runs>_eps*)")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--name", default="", help="letters only; output files overlap_<name>_* and macros \\ovl<name>* "
                    "(e.g. Ifc, Concise, Seven), so several overlap experiments fit in one paper")
    args = ap.parse_args()
    if os.path.isdir(args.runs) and any(d.startswith("s") and d[1:].isdigit() for d in os.listdir(args.runs)):
        seeds_report(args.runs, args.paper, args.name)  # v2 layout: <exp>/s<seed>/<method>[_<tag>]
        return
    tw_dirs = args.tripwire_runs if args.tripwire_runs is not None else (
        [args.runs] + sorted(glob.glob(args.runs.rstrip("/") + "_eps*")))

    base = {}
    for key, _, _ in BASELINES:
        d = summarize(os.path.join(args.runs, key), gated=(key == "router"))
        if d is not None:
            base[key] = d
    kls = []  # KL-regularized GRPO at every coefficient that was run
    for ed in [args.runs] + sorted(glob.glob(args.runs.rstrip("/") + "_kl*")):
        d = summarize(os.path.join(ed, "kl"), gated=False)
        if d is None:
            continue
        cj = os.path.join(ed, "kl", "config.json")
        d["beta"] = float(json.load(open(cj))["config"]["grpo"]["kl_beta"]) if os.path.exists(cj) else float("nan")
        kls.append(d)
    kls.sort(key=lambda d: d["chg"])
    tws = []
    for ed in tw_dirs:
        md = os.path.join(ed, "tripwire")
        tj = os.path.join(md, "stage1", "tripwire", "tripwire.json")
        d = summarize(md, gated=True)
        if d is None or not os.path.exists(tj):
            print(f"(tripwire run in {ed} not finished; skipped)")
            continue
        info = json.load(open(tj))
        d["eps"], d["kappa"], d["dir"] = float(info["eps"]), float(info["kappa"]), ed
        d["posthoc"] = info.get("posthoc", {})
        tws.append(d)
    tws.sort(key=lambda d: d["eps"])

    # ---- markdown for the terminal
    print("| Method | NQ before | NQ after | gain | Dolly changed | Dolly length (tok) | AdvBench refusal Δ | "
          "XSTest over-ref. Δ | MMLU Δ | fires on NQ | fired in prompt | gain on fired NQ | fires on Dolly | "
          "certified Dolly drift | exactness violations |")
    print("|" + "---|" * 15)
    entries = [("Seq-GRPO", base.get("seq"))] + [(f"+ KL beta={d['beta']:g}", d) for d in kls] + \
              [("Router (uncal.)", base.get("router"))] + [(f"Tripwire eps={d['eps']:.2f}", d) for d in tws]
    for name, d in entries:
        if d is None:
            print(f"| {name} | not finished |")
            continue
        print(f"| {name} | {pct(d['nq0'])} | {pct(d['nq1'])} | {pts(d['gain'])} | {pct(d['chg'])} | {d['dlen']:+.0f} | "
              f"{pts(d['adv'])} | {pts(d['xs'])} | {pts(d['mmlu'])} | {pct(d.get('fire_nq'))} | "
              f"{pct(d.get('in_prompt'))} | {pts(d.get('cond_gain'))} | {pct(d.get('fire_dolly'))} | "
              f"{pct(d.get('cert'))} | {d.get('exact_viol', '--')} |")

    if tws:
        ph = tws[0]["posthoc"]
        new, dolly = ph.get("new", []), ph.get("dolly", [])
        print(f"\nAUROC NQ vs Dolly (trajectory scores of the base model): {auroc(new, dolly):.3f}")
        print("frontier of the probe (Theorem 4.1): threshold set so that a fraction eps of Dolly prompts fires; "
              "until it fires the gated policy equals the base policy, so these base-model scores give the "
              "activation exactly")
        print("| eps on Dolly | activation on NQ |\n|---|---|")
        srt = sorted(dolly)
        for eps in (0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
            k = min(len(srt) - 1, max(0, int((1 - eps) * len(srt))))
            print(f"| {eps:.3f} | {100 * mean(float(v > srt[k]) for v in new):.1f} |")

    # ---- LaTeX table
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
           r"\caption{A new task that overlaps the reference: Natural Questions asked as plain questions, against "
           r"Dolly as the reference. NQ: accuracy gain (points). Changes relative to each method's base model: share "
           r"of Dolly responses that differ (Chg.), AdvBench refusal, XSTest over-refusal on safe prompts and MMLU "
           r"accuracy (points). Fires: share of prompts on which the gate fires. Cert.: 95\% Clopper--Pearson bound "
           r"on the Dolly firing rate, which bounds every behaviour change on Dolly (\cref{thm:coupling}); "
           r"$\varepsilon$ is the calibration target.}",
           r"\label{tab:overlap}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l r r rrr rr r}", r"\toprule",
           r" & NQ & \multicolumn{4}{c}{Change on reference} & \multicolumn{2}{c}{Fires (\%)} & \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
           r"Method & gain & Chg. & Refusal & Over-ref. & MMLU & NQ & Dolly & Cert. \\", r"\midrule"]
    rows_tex = [("Seq-GRPO", base.get("seq"))] + \
               [(f"+ KL $\\beta{{=}}{d['beta']:g}$" if d["beta"] == d["beta"] else "+ KL", d) for d in kls] + \
               [("Router (uncal.)", base.get("router"))]
    for name, d in rows_tex:
        if d is None:
            tex.append(f"{name} & " + " & ".join(["--"] * 8) + r" \\")
            continue
        tex.append(f"{name} & {tpts(d['gain'])} & {pct(d['chg'])} & {tpts(d['adv'])} & {tpts(d['xs'])} & "
                   f"{tpts(d['mmlu'])} & {pct(d.get('fire_nq'))} & {pct(d.get('fire_dolly'))} & none \\\\")
    if tws:
        tex.append(r"\midrule")
    for d in tws:
        tex.append(f"\\method{{}} $\\varepsilon{{=}}{d['eps']:.2f}$ & {tpts(d['gain'])} & {pct(d['chg'])} & "
                   f"{tpts(d['adv'])} & {tpts(d['xs'])} & {tpts(d['mmlu'])} & {pct(d['fire_nq'])} & "
                   f"{pct(d['fire_dolly'])} & {pct(d['cert'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    # ---- macros for the text
    nums = {}
    for key, _, mn in BASELINES:
        d = base.get(key)
        for f in ("gain", "chg", "adv", "xs", "mmlu", "chg_mmlu", "chg_xs", "fire_nq", "fire_dolly"):
            v = None if d is None else d.get(f)
            nums[f"ovl{mn}{f.replace('_', ' ').title().replace(' ', '')}"] = (
                tpts(v) if f in ("gain", "adv", "xs", "mmlu") else pct(v))
    for idx, d in enumerate(tws):
        L = TW_LETTERS[idx]
        nums[f"ovlTw{L}Eps"] = f"{d['eps']:.2f}"
        for f in ("gain", "chg", "adv", "xs", "mmlu", "fire_nq", "fire_dolly", "cert", "cond_gain", "in_prompt"):
            nums[f"ovlTw{L}{f.replace('_', ' ').title().replace(' ', '')}"] = (
                tpts(d.get(f)) if f in ("gain", "adv", "xs", "mmlu", "cond_gain") else pct(d.get(f)))
        nums[f"ovlTw{L}ExactViol"] = str(d.get("exact_viol", "--"))
        # the probe's own frontier point: NQ activation when exactly a fraction eps of (eval) Dolly prompts fires
        ph_new, ph_dolly = d["posthoc"].get("new", []), sorted(d["posthoc"].get("dolly", []))
        if ph_new and ph_dolly:
            k = min(len(ph_dolly) - 1, max(0, int((1 - d["eps"]) * len(ph_dolly))))
            nums[f"ovlTw{L}ProbeAct"] = pct(mean(float(v > ph_dolly[k]) for v in ph_new))
    if tws:
        ph = tws[0]["posthoc"]
        nums["ovlAuroc"] = f"{auroc(ph.get('new', []), ph.get('dolly', [])):.2f}"
        nums["ovlBaseNQ"] = pct(tws[0]["nq0"])

    res = os.path.join(args.paper, "results")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "overlap_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(res, "overlap_numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_overlap_report.py\n")
        for k in sorted(nums):
            f.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    # frontier data (pgfplots): x = Dolly responses changed (%), y = NQ gain (points)
    with open(os.path.join(res, "overlap_frontier_tw.dat"), "w") as f:
        f.write("chg gain cert eps label\n0 0 0 0 {}\n")
        for d in tws:
            f.write(f"{100 * d['chg']:.2f} {100 * d['gain']:.2f} {100 * d['cert']:.2f} {d['eps']:.2f} "
                    f"$\\varepsilon{{=}}{d['eps']:.2f}$\n")
    with open(os.path.join(res, "overlap_frontier_base.dat"), "w") as f:
        f.write("chg gain label\n")
        for key, mn in (("seq", "Seq"), ("router", "Router")):
            if key in base:
                f.write(f"{100 * base[key]['chg']:.2f} {100 * base[key]['gain']:.2f} {mn}\n")
    with open(os.path.join(res, "overlap_frontier_kl.dat"), "w") as f:
        f.write("chg gain beta label\n")
        for d in kls:
            lab = f"$\\beta{{=}}{d['beta']:g}$" if (d["beta"] == d["beta"] and len(kls) > 1) else "KL"
            f.write(f"{100 * d['chg']:.2f} {100 * d['gain']:.2f} {d['beta']:g} {lab}\n")
    print("\nwrote", res, "(overlap_table.tex, overlap_numbers.tex, overlap_frontier_*.dat)")


if __name__ == "__main__":
    main()
