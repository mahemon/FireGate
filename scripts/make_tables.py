#!/usr/bin/env python
"""Turn run outputs into the LaTeX tables, inline numbers and figures used by paper/main.tex.

    python scripts/make_tables.py --runs runs/stream_qwen15 --paper paper
    python scripts/make_tables.py --placeholder --paper paper      # "--" everywhere (before runs)

Reads, for every method directory under --runs:
    stage{k}/eval/{pool}.jsonl     one row per prompt (written by twcore/evaluate.py)
    stage{k}/tripwire/tripwire.json  calibration record (Tripwire only)
Writes paper/results/{numbers,main_table,cert_table,calib_table,stage_table}.tex,
paper/figures/roc.pdf (if matplotlib is available) and paper/results/summary.json.
Pure Python apart from the optional figure; no torch needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.certify import clopper_pearson_upper  # noqa: E402

METHODS = [("seq", "Seq-GRPO"), ("kl", "\\quad + KL"), ("replay", "\\quad + Replay"),
           ("refkl", "\\quad + Ref-KL"), ("olora", "\\quad + O-LoRA"), ("nullspace", "\\quad + Null-space LoRA"),
           ("ewc", "\\quad + EWC"), ("mas", "\\quad + MAS"), ("l2", "\\quad + L2 to old weights"),
           ("agem", "\\quad + A-GEM"), ("scale_a0.50", "\\quad adapters $\\times$0.5"),
           ("router", "Router (uncalibrated)"), ("tripwire_prompt", "\\method{}-prompt (ours)"),
           ("tripwire_prompt_seq", "\\method{}-prompt, whole sequence (ours)"),
           ("tripwire", "\\method{}-anytime (ours)")]
TASKS = ["gsm8k", "countdown", "kk"]   # replaced by the runs' own stream in configure()
TASK_NAMES = {"gsm8k": "GSM8K", "countdown": "Countdown", "kk": "K\\&K", "math": "MATH", "sudoku": "Sudoku",
              "nq": "NQ", "ifc": "IF", "concise": "Concise"}
MODEL_NAME = ["Qwen2.5-1.5B-Instruct"]


def configure(runs_root: str) -> None:
    """Take the task stream and the model name from the first run's config.json under runs_root."""
    for dirpath, _, files in os.walk(runs_root):
        if "config.json" in files:
            try:
                cfg = json.load(open(os.path.join(dirpath, "config.json")))["config"]
            except Exception:
                continue
            TASKS[:] = list(cfg["stream"])
            MODEL_NAME[0] = os.path.basename(os.path.normpath(cfg["model_path"]))
            for t in TASKS:
                TASK_NAMES.setdefault(t, t)
                COMP_NAMES.setdefault(f"old_{t}", TASK_NAMES[t])
            return
REF_POOLS = ["dolly", "advbench", "xstest", "mmlu"]
REF_NAMES = {"dolly": "Dolly", "advbench": "AdvBench",
             "xstest": "XSTest$^\\ast$", "mmlu": "MMLU$^\\ast$"}
# the bounded behaviour score whose change is reported for each pool
POOL_METRIC = {"gsm8k": "correct", "countdown": "correct", "kk": "correct", "math": "correct", "sudoku": "correct",
               "mmlu": "correct", "advbench": "refused", "xstest": "refused", "dolly": "leak"}
DASH = "--"
COMP_NAMES = {"dolly": "Dolly", "advbench": "AdvBench", "general": "task-like", "old_gsm8k": "GSM8K",
              "old_countdown": "Countdown", "old_kk": "K\\&K", "old_math": "MATH", "old_sudoku": "Sudoku"}


# ----------------------------------------------------------------------------- loading
def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_method(mdir):
    """{stage: {pool: [rows sorted by i]}}, {stage: tripwire.json}"""
    stages, tws = {}, {}
    if not os.path.isdir(mdir):
        return stages, tws
    for name in sorted(os.listdir(mdir)):
        if not name.startswith("stage"):
            continue
        try:
            k = int(name[5:])
        except ValueError:
            continue
        edir = os.path.join(mdir, name, "eval")
        if os.path.isfile(os.path.join(edir, "DONE")):
            pools = {}
            for fn in os.listdir(edir):
                if fn.endswith(".jsonl"):
                    rows = read_jsonl(os.path.join(edir, fn))
                    rows.sort(key=lambda r: r["i"])
                    pools[fn[:-6]] = rows
            stages[k] = pools
        tj = os.path.join(mdir, name, "tripwire", "tripwire.json")
        if os.path.isfile(tj):
            with open(tj) as f:
                tws[k] = json.load(f)
    return stages, tws


# ----------------------------------------------------------------------------- metrics
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def metric(rows, name, label=None):
    sel = [r for r in rows if label is None or r.get("label") == label]
    return mean(float(r["m"][name]) for r in sel)


def changed(rows_a, rows_b):
    assert len(rows_a) == len(rows_b)
    return mean(float(a["out"] != b["out"]) for a, b in zip(rows_a, rows_b))


def fired_any(rows, tids):
    return [any(r.get("fire", {}).get(t) is not None for t in tids) for r in rows]


def pct(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    return f"{100 * x:.{nd}f}"


def signed_pts(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    v = round(100 * x, nd) + 0.0  # + 0.0 turns -0.0 into 0.0
    return (f"{v:+.{nd}f}" if v != 0 else f"+{0:.{nd}f}").replace("-", "$-$", 1)


def final_stage(stages):
    return max(stages) if stages else None


def task_acc_matrix(stages):
    """acc[k][task] for all evaluated stages k."""
    return {k: {t: metric(p[t], "correct") for t in TASKS if t in p} for k, p in stages.items()}


def gains(stages):
    """plasticity = mean_j acc_j(after stage j) - acc_j(base); BWT = mean_{j<K} acc_j(K) - acc_j(j)."""
    acc = task_acc_matrix(stages)
    K = final_stage(stages)
    if K is None or 0 not in acc:
        return None, None
    g = [acc[j][t] - acc[0][t] for j, t in enumerate(TASKS, start=1) if j in acc and t in acc[j]]
    b = [acc[K][t] - acc[j][t] for j, t in enumerate(TASKS, start=1) if j < K and j in acc]
    return (mean(g) if g else None), (mean(b) if b else None)


# ----------------------------------------------------------------------------- tables
CERTIFIED = ("tripwire", "tripwire_prompt", "tripwire_prompt_seq")  # methods whose drift is certified before training
# method key -> part of the inline-number macro names (macro names may not contain digits)
MACRO_NAMES = {"seq": "Seq", "kl": "Kl", "replay": "Replay", "refkl": "Refkl", "olora": "Olora", "nullspace": "Null",
               "scale_a0.50": "Scale", "router": "Router", "tripwire_prompt": "Twp", "tripwire": "Tw", "ewc": "Ewc",
               "tripwire_prompt_seq": "Tws",
               "mas": "Mas", "l2": "Ltwo", "agem": "Agem"}


GPU_MARK = {}   # method key -> "A" when it ran on an A100 (v2: an experiment may mix H200 and A100 runs)


def gpu_short(name):
    for g in ("H200", "H100", "A100", "V100"):
        if g in (name or ""):
            return g
    return name or "?"


def main_cells(data):
    """Numeric cells of the main table: {"__base__": {...}, "__base__:<GPU>": {...}, method: {...}} (fractions or
    None). Every method is compared with its own stage-0 evaluation; one base-model row per GPU model."""
    out = {}
    base = None
    bases = {}
    for key, _ in METHODS:
        st = data.get(key, {}).get("stages", {})
        if 0 in st:
            base = base or st[0]
            bases.setdefault(data[key].get("gpu", "?"), st[0])
    if base is None:
        return out

    def base_cells(b):
        accs = [metric(b[t], "correct") if t in b else None for t in TASKS]
        return {"acc": accs, "avg": mean(accs) if all(a is not None for a in accs) else None,
                "leak": metric(b["dolly"], "leak") if "dolly" in b else None}

    out["__base__"] = base_cells(base)
    for g, b in bases.items():
        out["__base__:" + g] = base_cells(b)
    for key, _ in METHODS:
        st = data.get(key, {}).get("stages", {})
        K = final_stage(st)
        if K is None or K == 0:
            continue
        fin = st[K]
        b0 = st.get(0, base)  # each method's own stage-0 evaluation
        accs = [metric(fin[t], "correct") if t in fin else None for t in TASKS]
        out[key] = {"acc": accs, "avg": mean(accs) if all(a is not None for a in accs) else None,
                    "bwt": gains(st)[1],
                    "d_ref": metric(fin["advbench"], "refused") - metric(b0["advbench"], "refused"),
                    "d_xs": metric(fin["xstest"], "refused", "safe") - metric(b0["xstest"], "refused", "safe"),
                    "d_mm": metric(fin["mmlu"], "correct") - metric(b0["mmlu"], "correct"),
                    "chg": changed(fin["dolly"], b0["dolly"]), "leak": metric(fin["dolly"], "leak"),
                    "cert": cert_max(st) if key in CERTIFIED else None}
    return out


def main_head(show_n=False):
    n = len(TASKS)
    s = 1 if show_n else 0   # an extra column with the number of seeds behind each row
    return [r"\begin{tabular}{l " + ("c " if show_n else "") + "c" * n + r" cc ccc cc c}", r"\toprule",
            r" & " + ("& " if show_n else "") + r"\multicolumn{" + str(n + 2) + r"}{c}{Stream tasks} & "
            r"\multicolumn{3}{c}{$\Delta$ vs.\ base (pts)} & \multicolumn{2}{c}{Dolly (\%)} & Certified \\",
            r"\cmidrule(lr){" + str(2 + s) + "-" + str(n + 3 + s) + r"}\cmidrule(lr){" + str(n + 4 + s) + "-" +
            str(n + 6 + s) + r"}\cmidrule(lr){" + str(n + 7 + s) + "-" + str(n + 8 + s) + "}",
            "Method & " + (r"$n$ & " if show_n else "") + " & ".join(TASK_NAMES[t] for t in TASKS) +
            r" & Avg & BWT & Refusal & Over-ref. & MMLU & Chg. & Leak & drift (\%) \\", r"\midrule"]


def main_caption():
    return (r"After the stream " + r" $\to$ ".join(TASK_NAMES[t] for t in TASKS) + f" ({MODEL_NAME[0]}, greedy "
            r"decoding). Accuracy (\%) on each task, average and backward transfer (BWT); changes relative to "
            r"the base model in AdvBench refusal, XSTest over-refusal on safe prompts and MMLU accuracy "
            r"(points); share of Dolly responses that differ from the base model's (Chg.) or use a trained "
            r"answer format (Leak). The last column is the label-free certificate: the largest 95\% "
            r"Clopper--Pearson bound on the firing over the four evaluation pools (Dolly, AdvBench, XSTest, MMLU; "
            r"\cref{tab:cert}); the other methods have none.")


def _main_rows(cells, fmt_pct, fmt_pts, show_n=False):
    lines = []
    nb = f"{DASH} & " if show_n else ""   # the seeds column of the base-model rows
    b = cells.get("__base__")
    per_gpu = sorted(k for k in cells if k.startswith("__base__:"))
    if b is not None and len(per_gpu) > 1:   # runs on several GPU models: one base row per model
        for k in per_gpu:
            bb = cells[k]
            lines.append(f"Base model ({k.split(':', 1)[1]}) & {nb}" + " & ".join(fmt_pct(a) for a in bb["acc"]) +
                         f" & {fmt_pct(bb['avg'])} & {DASH} & " + " & ".join(["0.0"] * 3) +
                         f" & 0.0 & {fmt_pct(bb['leak'])} & 0.0 \\\\")
    elif b is not None:
        lines.append(f"Base model & {nb}" + " & ".join(fmt_pct(a) for a in b["acc"]) +
                     f" & {fmt_pct(b['avg'])} & {DASH} & " + " & ".join(["0.0"] * 3) +
                     f" & 0.0 & {fmt_pct(b['leak'])} & 0.0 \\\\")
    else:
        lines.append("Base model & " + " & ".join([DASH] * (len(TASKS) + 8 + (1 if show_n else 0))) + r" \\")
    lines.append(r"\midrule")
    for key, name in METHODS:
        c = cells.get(key)
        if c is None:   # no finished run of this method: no row
            continue
        cert_s = "none" if key not in CERTIFIED else ("$\\le$" + fmt_pct(c["cert"]) if c["cert"] is not None else DASH)
        if GPU_MARK.get(key):
            name = name + r"$^{\mathrm{" + GPU_MARK[key] + "}}$"
        if show_n:
            name = f"{name} & {c.get('n', DASH)}"
        lines.append(f"{name} & " + " & ".join(fmt_pct(a) for a in c["acc"]) +
                     f" & {fmt_pct(c['avg'])} & {fmt_pts(c['bwt'])} & {fmt_pts(c['d_ref'])} & {fmt_pts(c['d_xs'])} & "
                     f"{fmt_pts(c['d_mm'])} & {fmt_pct(c['chg'])} & {fmt_pct(c['leak'])} & {cert_s} \\\\")
    return lines


def main_table(data):
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{" + main_caption() + "}", r"\label{tab:main}"] + main_head()
    lines += _main_rows(main_cells(data), pct, signed_pts)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- several seeds
def _agg(vals):
    """mean and sample standard deviation of the non-missing values (None if none)."""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return (m, sd, len(v))


def _pm(agg, signed=False):
    if agg is None:
        return DASH
    m, sd, n = agg
    core = signed_pts(m) if signed else pct(m)
    return core if n < 2 else f"{core}{{\\scriptsize$\\pm${100 * sd:.1f}}}"


def main_table_seeds(datas):
    cells = [main_cells(d) for d in datas]
    keys = {k for c in cells for k in c}
    agg = {}
    for k in keys:
        cs = [c[k] for c in cells if k in c]
        agg[k] = {f: ([_agg([c[f][i] for c in cs]) for i in range(len(TASKS))] if f == "acc"
                      else _agg([c.get(f) for c in cs])) for f in cs[0]}
        certs = [c.get("cert") for c in cs if c.get("cert") is not None]
        if "cert" in agg[k]:
            agg[k]["cert"] = (max(certs), 0.0, 1) if certs else None  # the certificate is the worst seed's
        agg[k]["n"] = len(cs)   # finished seeds behind the row
    n = len(datas)
    cap = main_caption() + (f" Mean $\\pm$ standard deviation over $n$ seeds (of {n}; training "
                          f"randomness; same data and evaluation prompts); the certificate is the largest over seeds.")
    if GPU_MARK:
        cap += (r" Rows marked $^{\mathrm{A}}$ ran on NVIDIA A100 GPUs (in at least one seed), the others on H200; "
                r"each run is compared with the base model evaluated on its own GPU model.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{" + cap + "}", r"\label{tab:main}", r"\resizebox{\textwidth}{!}{%"] + main_head(True)
    lines += _main_rows(agg, lambda a: _pm(a), lambda a: _pm(a, signed=True), show_n=True)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def cert_table_seeds(datas):
    """Certificates per pool over seeds for each certified variant (prompt-time and anytime tripwire side by side):
    mean X/n, largest CP bound, mean share of changed responses; and Seq-GRPO's share of changed responses."""
    variants = [(k, n) for k, n in (("tripwire_prompt", "prompt-time"), ("tripwire", "anytime"))
                if any(d.get(k, {}).get("stages") for d in datas)]
    per = {}
    seen = set()   # Seq-GRPO's share is counted once per seed and pool, whichever variant produced the row
    for i, d in enumerate(datas):
        seq = d.get("seq", {}).get("stages", {})
        Ks = final_stage(seq)
        for key, _ in variants:
            for r in cert_rows(d.get(key, {}).get("stages", {})):
                e = per.setdefault((r["pool"], r["ref"]), {"n": r["n"], "seq": []})
                v = e.setdefault(key, {"frac": [], "cp": [], "chg": []})
                v["frac"].append(r["frac"]); v["cp"].append(r["cp"]); v["chg"].append(r["chg"])
                if (i, r["pool"], r["ref"]) not in seen and Ks is not None and r["pool"] in seq.get(Ks, {}) \
                        and r["ref"] in seq:
                    seen.add((i, r["pool"], r["ref"]))
                    e["seq"].append(changed(seq[Ks][r["pool"]], seq[r["ref"]][r["pool"]]))
    names = dict(REF_NAMES, **{t: TASK_NAMES[t] for t in TASKS})
    nv = max(len(variants), 1)
    n_s = len(datas)
    n_v = {key: sum(1 for d in datas if d.get(key, {}).get("stages")) for key, _ in variants}
    head = " & ".join(r"\multicolumn{3}{c}{\method{} " + n + f" ($n{{=}}{n_v[k]}$)" + "}" for k, n in variants)
    rules = "".join(r"\cmidrule(lr){" + f"{4 + 3 * i}-{6 + 3 * i}" + "}" for i in range(len(variants)))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Certificates of the final models over $n$ seeds (of " + str(n_s) +
             r"). Reference pools are compared with the base model $\pi_0$, earlier tasks with the model right after "
             r"that task (\cref{thm:stream}). $X/n$: share of prompts on which a tripwire fired (mean over seeds); CP: "
             r"largest 95\% Clopper--Pearson bound over seeds; Chg.: share of responses that changed (mean). Seq: "
             r"share changed by Seq-GRPO (mean over the seeds with a certified run). $^\ast$Not used in calibration.}",
             r"\label{tab:cert}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l c r " + "r r r " * nv + "r}", r"\toprule",
             r" & & & " + head + r" & \\", rules,
             r"Pool & vs. & $n$ & " + " & ".join([r"$X/n$ & CP & Chg."] * len(variants)) + r" & Seq \\", r"\midrule"]
    for (pool, ref), e in sorted(per.items(), key=lambda kv: (kv[0][1], (REF_POOLS + TASKS).index(kv[0][0]))):
        cells = []
        for key, _ in variants:
            v = e.get(key)
            cells += ([pct(mean(v["frac"])), pct(max(v["cp"])), pct(mean(v["chg"]))] if v else [DASH] * 3)
        lines.append(f"{names[pool]} & $\\pi_{ref}$ & {e['n']} & " + " & ".join(cells) +
                     f" & {pct(mean(e['seq'])) if e['seq'] else DASH} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def numbers_seeds(datas):
    """Mean of each inline number over seeds (counts are summed), plus <name>Sd macros with the standard deviation."""
    per = [numbers(d) for d in datas]
    out = {}
    for k in per[0]:
        vals = []
        for p in per:
            v = p.get(k, DASH)
            if v == DASH:
                continue
            signed = v.startswith(("+", "$-$"))
            try:
                vals.append(float(v.replace("$-$", "-")))
            except ValueError:
                pass
        if not vals:
            out[k] = DASH
            continue
        if k in ("resExactN", "resExactViol", "resExactPN", "resExactPViol"):
            out[k] = str(int(sum(vals)))
            continue
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        out[k] = (signed_pts(m / 100) if signed else (f"{m:.0f}" if k == "resRetain" else f"{m:.1f}"))
        out[k + "Sd"] = f"{sd:.1f}"
    # average accuracy, BWT and Dolly change from the unrounded per-seed values, as in the main table
    cells = [main_cells(d) for d in datas]
    for key, mn in MACRO_NAMES.items():
        cs = [c[key] for c in cells if key in c]
        for f, mac, signed in (("avg", "resAvg", False), ("bwt", "resBwt", True), ("chg", "resChg", False)):
            a = _agg([c.get(f) for c in cs])
            if a is None:
                continue
            name = f"{mac}{mn}" + ("Dolly" if f == "chg" else "")
            out[name] = signed_pts(a[0]) if signed else pct(a[0])
            out[name + "Sd"] = f"{100 * a[1]:.1f}"
    out["resSeeds"] = str(len(datas))
    return out


def all_tids(stages):
    K = final_stage(stages)
    return [str(k) for k in range(1, (K or 0) + 1)]


def cert_rows(st, delta=0.05):
    """Certificates for the final Tripwire model (Theorem 3), one dict per pool."""
    K = final_stage(st)
    out = []
    if K is None or K == 0:
        return out
    fin = st[K]
    for pool in REF_POOLS:
        if pool not in fin or 0 not in st:
            continue
        f = fired_any(fin[pool], all_tids(st))
        X, n = sum(f), len(f)
        out.append(dict(pool=pool, ref=0, n=n, X=X, frac=X / n, cp=clopper_pearson_upper(X, n, delta),
                        dmetric=abs(metric(fin[pool], POOL_METRIC[pool]) - metric(st[0][pool], POOL_METRIC[pool])),
                        chg=changed(fin[pool], st[0][pool])))
    for j, task in enumerate(TASKS, start=1):
        if j >= K or task not in fin or j not in st:
            continue
        later = [str(i) for i in range(j + 1, K + 1)]
        f = fired_any(fin[task], later)
        X, n = sum(f), len(f)
        out.append(dict(pool=task, ref=j, n=n, X=X, frac=X / n, cp=clopper_pearson_upper(X, n, delta),
                        dmetric=abs(metric(fin[task], "correct") - metric(st[j][task], "correct")),
                        chg=changed(fin[task], st[j][task])))
    return out


def cert_max(st):
    rows = [r for r in cert_rows(st) if r["ref"] == 0]
    return max(r["cp"] for r in rows) if rows else None


def cert_table(data):
    tw = data.get("tripwire", {}).get("stages", {})
    seq = data.get("seq", {}).get("stages", {})
    rows = cert_rows(tw)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Certificates of the final \method{} model. Reference pools are compared with the base "
             r"model $\pi_0$, earlier tasks with the model right after that task ($\pi_1$, $\pi_2$; "
             r"\cref{thm:stream}). $X$ of $n$ prompts fired a tripwire; at most $X/n$ of greedy responses can "
             r"differ, and CP is the 95\% Clopper--Pearson bound on the population firing rate. Measured: "
             r"change of the pool's score (refusal, leak or accuracy, points) and share of changed responses, "
             r"for \method{}, and share of changed responses for Seq-GRPO (Seq). $^\ast$Not used in calibration.}", r"\label{tab:cert}",
             r"\begin{tabular}{l c r r r r r r}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Certified (\%)} & \multicolumn{3}{c}{Measured} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Pool & vs. & $n$ & $X/n$ & CP & $|\Delta|$ & Chg. & Seq \\", r"\midrule"]
    names = dict(REF_NAMES, **{t: TASK_NAMES[t] for t in TASKS})
    if not rows:
        for p in REF_POOLS + TASKS[:2]:
            lines.append(f"{names[p]} & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} & {DASH} \\\\")
    for r in rows:
        Ks = final_stage(seq)
        if Ks is not None and r["pool"] in seq.get(Ks, {}) and r["ref"] in seq:
            chg_seq = pct(changed(seq[Ks][r["pool"]], seq[r["ref"]][r["pool"]]))
        else:
            chg_seq = DASH
        lines.append(f"{names[r['pool']]} & $\\pi_{r['ref']}$ & {r['n']} & {pct(r['frac'])} & {pct(r['cp'])} & "
                     f"{pct(r['dmetric'])} & {pct(r['chg'])} & {chg_seq} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def calib_table(data):
    tw = data.get("tripwire", {})
    st, tws = tw.get("stages", {}), tw.get("tripwires", {})
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{Tripwire calibration per stage: AUROC of the trajectory score (new task vs.\ reference, "
             r"held-out fitting trajectories), certified $\varepsilon$ per reference component (\cref{thm:calib}, target 3\%; "
             r"components: Dolly, AdvBench, task-like prompts and the stored prompts of each earlier task), "
             r"activation (share of held-out new-task prompts on which the tripwire fires) and the median "
             r"first-firing position relative to the end of the prompt (negative: inside the prompt).}",
             r"\label{tab:calib}", r"\begin{tabular}{l c l c c}", r"\toprule",
             r"Stage & AUROC & Certified $\varepsilon$ per component (\%) & Activation (\%) & Median pos. \\",
             r"\midrule"]
    if not tws:
        for k, t in enumerate(TASKS, start=1):
            lines.append(f"{k}: {TASK_NAMES[t]} & {DASH} & {DASH} & {DASH} & {DASH} \\\\")
    for k in sorted(tws):
        info = tws[k]
        vals = [pct(v["eps_cert"]) for v in info["components"].values()]
        if len(set(vals)) == 1:
            comps = f"{vals[0]} on each of {len(vals)}"
        else:  # component names such as old_gsm8k must not reach LaTeX with an underscore
            comps = ", ".join(f"{COMP_NAMES.get(c, c.replace('_', ' '))}: {pct(v['eps_cert'])}"
                              for c, v in info["components"].items())
        task = TASKS[k - 1]
        act, med = DASH, DASH
        if k in st and task in st[k]:
            rows = st[k][task]
            pos = [r["fire"].get(str(k)) for r in rows]
            fired = [p for p in pos if p is not None]
            act = pct(len(fired) / len(rows))
            if fired:
                fired.sort()
                med = str(fired[len(fired) // 2])
        lines.append(f"{k}: {TASK_NAMES[task]} & {info.get('auroc', float('nan')):.3f} & {comps} & {act} & {med} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def stage_table(data):
    lines = [r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{All evaluations after every stage (\%). Stage 0 is the base model.}",
             r"\label{tab:stages}", r"\begin{tabular}{l c " + "c" * len(TASKS) + r" cccc}", r"\toprule",
             "Method & Stage & " + " & ".join(TASK_NAMES[t] for t in TASKS) +
             r" & AdvB.\ refusal & XSTest over-ref. & MMLU & Dolly leak \\",
             r"\midrule"]
    any_rows = False
    for key, name in METHODS:
        st = data.get(key, {}).get("stages", {})
        for k in sorted(st):
            p = st[k]
            vals = [pct(metric(p[t], "correct")) if t in p else DASH for t in TASKS]
            vals += [pct(metric(p["advbench"], "refused")) if "advbench" in p else DASH,
                     pct(metric(p["xstest"], "refused", "safe")) if "xstest" in p else DASH,
                     pct(metric(p["mmlu"], "correct")) if "mmlu" in p else DASH,
                     pct(metric(p["dolly"], "leak")) if "dolly" in p else DASH]
            lines.append(f"{name} & {k} & " + " & ".join(vals) + r" \\")
            any_rows = True
        if st:
            lines.append(r"\midrule")
    if not any_rows:
        lines.append(" & ".join([DASH] * (len(TASKS) + 6)) + r" \\")
    elif lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def exact_check(st):
    """Theorem 1 check: non-fired prompts of stage k must keep stage k-1's greedy response.
    Only pools evaluated on the same GPU type at both stages are compared (bit-identity needs identical kernels)."""
    n_checked, n_viol = 0, 0
    for k in sorted(st):
        if k == 0 or (k - 1) not in st:
            continue
        for pool, rows in st[k].items():
            prev = st[k - 1].get(pool)
            if prev is None:
                continue
            if rows and prev and rows[0].get("dev") != prev[0].get("dev"):
                print(f"exactness check: {pool} stage {k - 1}->{k} ran on different GPUs; skipped", file=sys.stderr)
                continue
            for a, b in zip(rows, prev):
                if a["fire"].get(str(k)) is None:
                    n_checked += 1
                    n_viol += int(a["out"] != b["out"])
    return n_checked, n_viol


def numbers(data):
    tw = data.get("tripwire", {}).get("stages", {})
    seq = data.get("seq", {}).get("stages", {})
    nums = {}
    g_tw, _ = gains(tw) if tw else (None, None)
    g_seq, _ = gains(seq) if seq else (None, None)
    nums["resRetain"] = f"{100 * g_tw / g_seq:.0f}" if (g_tw is not None and g_seq) and g_seq > 0 else DASH
    twp = data.get("tripwire_prompt", {}).get("stages", {})
    crp = {r["pool"]: r for r in cert_rows(twp) if r["ref"] == 0}
    for pool, mac in (("advbench", "Adv"), ("dolly", "Dolly"), ("xstest", "XS"), ("mmlu", "MMLU")):
        nums[f"resCertP{mac}"] = pct(crp[pool]["cp"]) if pool in crp else DASH
    if twp:
        n_c, n_v = exact_check(twp)
        nums["resExactPN"], nums["resExactPViol"] = (str(n_c), str(n_v)) if n_c else (DASH, DASH)
    else:
        nums["resExactPN"] = nums["resExactPViol"] = DASH
    cr = {r["pool"]: r for r in cert_rows(tw) if r["ref"] == 0}
    nums["resCertAdv"] = pct(cr["advbench"]["cp"]) if "advbench" in cr else DASH
    nums["resCertDolly"] = pct(cr["dolly"]["cp"]) if "dolly" in cr else DASH
    nums["resCertXS"] = pct(cr["xstest"]["cp"]) if "xstest" in cr else DASH
    nums["resCertMMLU"] = pct(cr["mmlu"]["cp"]) if "mmlu" in cr else DASH
    Ks = final_stage(seq)
    if Ks and 0 in seq:
        nums["resSeqChangeDolly"] = pct(changed(seq[Ks]["dolly"], seq[0]["dolly"]))
        nums["resSeqLeak"] = pct(metric(seq[Ks]["dolly"], "leak"))
        nums["resSeqAdvDelta"] = signed_pts(metric(seq[Ks]["advbench"], "refused") - metric(seq[0]["advbench"], "refused"))
    else:
        nums["resSeqChangeDolly"] = nums["resSeqLeak"] = nums["resSeqAdvDelta"] = DASH
    Kt = final_stage(tw)
    nums["resTwLeak"] = pct(metric(tw[Kt]["dolly"], "leak")) if Kt else DASH
    if tw:
        n_c, n_v = exact_check(tw)
        nums["resExactN"], nums["resExactViol"] = (str(n_c), str(n_v)) if n_c else (DASH, DASH)
        acts = []
        for k, t in enumerate(TASKS, start=1):
            if k in tw and t in tw[k]:
                rows = tw[k][t]
                acts.append(mean(float(r["fire"].get(str(k)) is not None) for r in rows))
        nums["resActMin"] = pct(min(acts)) if acts else DASH
    else:
        nums["resExactN"] = nums["resExactViol"] = nums["resActMin"] = DASH
    # per method (macro names may not contain digits): final average accuracy, BWT, share of responses changed
    # relative to the method's own base model on each reference pool, and (gated methods) firing on those pools
    pnames = {"dolly": "Dolly", "advbench": "Adv", "xstest": "Xs", "mmlu": "Mmlu"}
    for key, mn in MACRO_NAMES.items():
        st = data.get(key, {}).get("stages", {})
        K = final_stage(st)
        ok = K is not None and K > 0 and 0 in st
        accs = [metric(st[K][t], "correct") for t in TASKS if ok and t in st[K]]
        nums[f"resAvg{mn}"] = pct(mean(accs)) if ok and len(accs) == len(TASKS) else DASH
        nums[f"resBwt{mn}"] = signed_pts(gains(st)[1]) if ok else DASH
        for pool, pn in pnames.items():
            have = ok and pool in st[K] and pool in st[0]
            nums[f"resChg{mn}{pn}"] = pct(changed(st[K][pool], st[0][pool])) if have else DASH
            if key in ("router", "tripwire", "tripwire_prompt", "tripwire_prompt_seq"):
                nums[f"resFire{mn}{pn}"] = pct(mean(fired_any(st[K][pool], all_tids(st)))) if have else DASH
    return nums


# ----------------------------------------------------------------------------- robustness (eval_extra.py)
EXTRA_METHODS = [("seq", "Seq-GRPO"), ("refkl", "\\quad + Ref-KL"), ("nullspace", "\\quad + Null-space LoRA"),
                 ("router", "Router (uncalibrated)"), ("tripwire_prompt", "\\method{}-prompt (ours)"),
                 ("tripwire", "\\method{}-anytime (ours)")]
SAMPLE_POOLS = ("dolly", "advbench", "xstest", "mmlu")


def load_extra(runs_root, seeds):
    """{method: [(seed folder, summary), ...]} for every finished robustness evaluation (summary with "sample")."""
    out = {}
    for sd in seeds:
        for key, _ in EXTRA_METHODS:
            p = os.path.join(runs_root, sd, key, "extra", "summary.json")
            if os.path.isfile(p):
                summ = json.load(open(p))
                if "sample" in summ:
                    out.setdefault(key, []).append((sd, summ))
    return out


def _avg(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def extra_cells(per_seed):
    """Means over seeds (and over sampling seeds / task formats) of the robustness numbers of one method."""
    c = {}
    samp = [s["sample"] for _, s in per_seed]
    c["samp_chg"] = _avg([r["changed"] for sm in samp for r in sm.get("dolly", {}).get("seeds", [])])
    c["samp_viol"] = sum(r["nonfired_changed"] for sm in samp for p in SAMPLE_POOLS for r in sm.get(p, {}).get("seeds", []))
    c["samp_nonfired"] = sum(r["n"] - r["fired"] for sm in samp for p in SAMPLE_POOLS
                             for r in sm.get(p, {}).get("seeds", []))
    c["samp_equal"] = all(r.get("fire_record_equal", True) for sm in samp for p in SAMPLE_POOLS
                          for r in sm.get(p, {}).get("seeds", []))
    wraps = [v for _, s in per_seed for v in s.get("advwrap", {}).values()]
    c["wrap_fire"] = _avg([v["fire_rate"] for v in wraps])
    c["wrap_dref"] = _avg([v.get("refused_delta") for v in wraps])
    c["wrap_dref_min"] = min((v["refused_delta"] for v in wraps if v.get("refused_delta") is not None), default=None)
    for name, k in (("dolly_sys", "sys"), ("dolly_turn", "turn"), ("advbench_sys", "advsys")):
        vs = [s.get("shift", {}).get(name) for _, s in per_seed]
        vs = [v for v in vs if v]
        c[f"{k}_fire"] = _avg([v["fire_rate"] for v in vs])
        c[f"{k}_chg"] = _avg([v["changed"] for v in vs])
        c[f"{k}_dref"] = _avg([v.get("refused_delta") for v in vs])
    c["n"] = len(per_seed)
    return c


def extra_table(extra):
    gated = ("router", "tripwire", "tripwire_prompt")
    n_s = sorted({sd for v in extra.values() for sd, _ in v})
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Robustness of the final stream models (" + ", ".join("seed " + s[1:] for s in n_s) + r"). "
             r"Sampling: reference pools decoded at temperature 0.7 with two seeds, final and base model with shared "
             r"randomness; Chg.: share of Dolly responses that differ; Viol.: non-fired responses that differ, over all "
             r"four pools and both seeds (\cref{thm:coupling} requires 0). Harmful requests in task formats: AdvBench "
             r"requests rewritten in each trained task's answer format, mean over the formats; Fires: share on which a "
             r"tripwire fires; $\Delta$Ref.: change in refusal (points; worst format in brackets). Format shift: Dolly "
             r"requests behind a system prompt or after an earlier exchange, and AdvBench behind a system prompt.}",
             r"\label{tab:extra}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l rr rr rr rr r}", r"\toprule",
             r" & \multicolumn{2}{c}{Sampling} & \multicolumn{2}{c}{Harmful, task format} & "
             r"\multicolumn{2}{c}{Dolly + system} & \multicolumn{2}{c}{Dolly after a turn} & AdvB.\ + system \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
             r"Method & Chg.\ (\%) & Viol. & Fires (\%) & $\Delta$Ref. & Fires (\%) & Chg.\ (\%) & Fires (\%) & "
             r"Chg.\ (\%) & $\Delta$Ref. \\", r"\midrule"]
    for key, name in EXTRA_METHODS:
        if key not in extra:
            continue
        c = extra_cells(extra[key])
        g = key in gated
        viol = f"{c['samp_viol']}/{c['samp_nonfired']}" if g else DASH
        wr = signed_pts(c["wrap_dref"]) + (f" ({signed_pts(c['wrap_dref_min'])})" if c["wrap_dref_min"] is not None
                                           else "")
        lines.append(" & ".join([name, pct(c["samp_chg"]), viol, pct(c["wrap_fire"]) if g else DASH, wr,
                                 pct(c["sys_fire"]) if g else DASH, pct(c["sys_chg"]),
                                 pct(c["turn_fire"]) if g else DASH, pct(c["turn_chg"]),
                                 signed_pts(c["advsys_dref"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def extra_numbers(extra):
    """Inline numbers: resX<Field><Method> (method part as in MACRO_NAMES)."""
    out = {}
    viol = nonfired = 0
    for key, per_seed in extra.items():
        mn = MACRO_NAMES.get(key)
        if mn is None:
            continue
        c = extra_cells(per_seed)
        out[f"resXSampChg{mn}"] = pct(c["samp_chg"])
        out[f"resXWrapFire{mn}"] = pct(c["wrap_fire"])
        out[f"resXWrapRef{mn}"] = signed_pts(c["wrap_dref"])
        out[f"resXWrapRefMin{mn}"] = signed_pts(c["wrap_dref_min"])
        out[f"resXSysFire{mn}"] = pct(c["sys_fire"])
        out[f"resXSysChg{mn}"] = pct(c["sys_chg"])
        out[f"resXTurnFire{mn}"] = pct(c["turn_fire"])
        out[f"resXTurnChg{mn}"] = pct(c["turn_chg"])
        out[f"resXAdvSysRef{mn}"] = signed_pts(c["advsys_dref"])
        if key in ("tripwire", "tripwire_prompt"):
            viol += c["samp_viol"]
            nonfired += c["samp_nonfired"]
    out["resXSampViol"], out["resXSampNonfired"] = str(viol), str(nonfired)
    return out


def numbers_tex(nums):
    lines = ["% generated by scripts/make_tables.py"]
    for k in sorted(nums):
        lines.append(f"\\newcommand{{\\{k}}}{{{nums[k]}}}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figure
def roc_figure(data, path):
    tws = data.get("tripwire", {}).get("tripwires", {})
    if not tws:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure", file=sys.stderr)
        return False
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    colors = ["#1f5fa8", "#c2541b", "#3a8a3a", "#7b4ea3", "#a8861f"]
    for idx, k in enumerate(sorted(tws)):
        ph = tws[k].get("posthoc", {})
        new = ph.get("new", [])
        ref = [s for c in ("dolly", "advbench") for s in ph.get(c, [])]
        if not new or not ref:
            continue
        thr = sorted(set(new + ref), reverse=True)
        xs, ys = [0.0], [0.0]
        for t in thr:
            xs.append(mean(float(s >= t) for s in ref))
            ys.append(mean(float(s >= t) for s in new))
        ax.plot(xs, ys, color=colors[idx % len(colors)], lw=1.4, label=f"stage {k}: {TASKS[k - 1]}")
        kap = tws[k]["kappa"]
        ax.plot([mean(float(s > kap) for s in ref)], [mean(float(s > kap) for s in new)], "o",
                color=colors[idx % len(colors)], ms=4)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("firing rate on reference (certified drift)")
    ax.set_ylabel("firing rate on new task")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main
def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/stream_qwen15", help="experiment directory holding one dir per method")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--placeholder", action="store_true")
    args = ap.parse_args()
    if not args.placeholder and os.path.isdir(args.runs):
        configure(args.runs)

    seeds = sorted((d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit()
                    and os.path.isdir(os.path.join(args.runs, d))), key=lambda d: int(d[1:])) \
        if (not args.placeholder and os.path.isdir(args.runs)) else []
    if seeds:  # v2 layout <exp>/s<seed>/<method>: tables over seeds
        datas = []
        for sd in seeds:
            data, unfinished = {}, []
            for key, _ in METHODS:
                md = os.path.join(args.runs, sd, key)
                if not os.path.isdir(md):
                    continue
                if not os.path.exists(os.path.join(md, "DONE")):   # tables use finished runs only
                    unfinished.append(key)
                    continue
                st, tw = load_method(md)
                if st or tw:
                    cj = os.path.join(md, "config.json")
                    dev = json.load(open(cj)).get("device") if os.path.exists(cj) else None
                    data[key] = {"stages": st, "tripwires": tw, "gpu": gpu_short(dev)}
            print(f"{sd}: finished:", {k: f"{len(v['stages']) - 1} stages, {v['gpu']}" for k, v in data.items()},
                  "| unfinished (left out):", unfinished or "none")
            if data:
                datas.append(data)
        gpus = {v["gpu"] for d in datas for v in d.values()}
        if len(gpus) > 1:
            GPU_MARK.update({k: "A" for d in datas for k, v in d.items() if v["gpu"] == "A100"})
        if not datas:
            print("no finished runs under", args.runs)
            return
        res = os.path.join(args.paper, "results")
        write(os.path.join(res, "numbers.tex"), numbers_tex(numbers_seeds(datas)))
        write(os.path.join(res, "main_table.tex"), main_table_seeds(datas))
        write(os.path.join(res, "cert_table.tex"), cert_table_seeds(datas))
        write(os.path.join(res, "calib_table.tex"), calib_table(datas[0]).replace(r"\caption{", r"\caption{Seed " + seeds[0][1:] + ". ", 1))
        write(os.path.join(res, "stage_table.tex"), stage_table(datas[0]).replace(r"\caption{", r"\caption{Seed " + seeds[0][1:] + ". ", 1))
        extra = load_extra(args.runs, seeds)
        if extra:   # robustness evaluations (scripts/eval_extra.py): table, inline numbers and the raw summaries
            write(os.path.join(res, "extra_table.tex"), extra_table(extra))
            write(os.path.join(res, "extra_numbers.tex"), numbers_tex(extra_numbers(extra)))
            for key, per_seed in extra.items():
                for sd, summ in per_seed:
                    write(os.path.join(args.paper, "extra", f"{sd}_{key}.json"), json.dumps(summ, indent=1))
            print("robustness evaluations:", {k: [sd for sd, _ in v] for k, v in extra.items()})
        print("wrote", res, f"({len(datas)} seeds)")
        return

    data = {}
    if not args.placeholder:
        for key, _ in METHODS:
            st, tw = load_method(os.path.join(args.runs, key))
            if st or tw:
                data[key] = {"stages": st, "tripwires": tw}
        print("methods found:", {k: sorted(v["stages"]) for k, v in data.items()})

    res = os.path.join(args.paper, "results")
    write(os.path.join(res, "numbers.tex"), numbers_tex(numbers(data)))
    write(os.path.join(res, "main_table.tex"), main_table(data))
    write(os.path.join(res, "cert_table.tex"), cert_table(data))
    write(os.path.join(res, "calib_table.tex"), calib_table(data))
    write(os.path.join(res, "stage_table.tex"), stage_table(data))
    if not args.placeholder:
        os.makedirs(os.path.join(args.paper, "figures"), exist_ok=True)
        roc_figure(data, os.path.join(args.paper, "figures", "roc.pdf"))
        summary = {k: {"final_stage": final_stage(v["stages"]),
                       "gains_bwt": gains(v["stages"]) if v["stages"] else None} for k, v in data.items()}
        summary["numbers"] = numbers(data)
        if "tripwire" in data:
            summary["certificates"] = cert_rows(data["tripwire"]["stages"])
        write(os.path.join(res, "summary.json"), json.dumps(summary, indent=2, default=str))
    print("wrote", res)


if __name__ == "__main__":
    main()
