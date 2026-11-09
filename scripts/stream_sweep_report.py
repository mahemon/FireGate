#!/usr/bin/env python
"""Step p3.25 (Nov 5, PLAN_PAPER3.md): continual-learning baselines on a task stream at their best settings (CPU).

    python scripts/stream_sweep_report.py --runs runs/v2_stream_q15 --out paper/v3_results/p3_25

A review asks whether the ungated continual-learning baselines were tuned too narrowly. This reads every finished run
of one stream experiment (each method family at every setting trained, every seed) and reports, per family:
  Avg   mean accuracy over the stream's tasks after the last stage
  BWT   backward transfer, mean over earlier tasks j of acc_j(after the last stage) - acc_j(after stage j)
  Chg.  share of held-out Dolly responses that differ from the run's own base model (stage 0)
  AdvBench refusal and MMLU accuracy changes (whole pools; not used for any choice)
The setting of each family is chosen on a validation half of every task's and Dolly's evaluation prompts (a fixed
random half, seed 0, as scripts/budget_report.py) and reported on the other half, so the choice does not inflate the
result. Rules, fixed before the runs: "best" is the setting with the largest validation Avg; "within b" the largest
validation Avg among settings whose validation Dolly change is at most b. A family with no setting within b gets none.
Writes, in --out: stream_<exp>.json, stream_<exp>_table.tex, stream_<exp>_summary.txt (with the predictions of
PLAN_PAPER3.md, step p3.25) and chosen_<exp>.txt, the "method:tag:overrides" spec of each ungated family's best
setting (for the record; no plan step trains more seeds of it).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# family -> (label, config key of its setting or None, symbol)
FAMILIES = [("seq", "Seq-GRPO", None, ""), ("kl", "+ KL", ("grpo", "kl_beta"), r"\beta"),
            ("refkl", "+ Ref-KL", ("grpo", "refkl_beta"), r"\beta"),
            ("replay", "+ Replay", ("grpo", "replay_frac"), "r"),
            ("olora", "O-LoRA", ("grpo", "olora_lambda"), r"\lambda"),
            ("nullspace", "Null-space LoRA", ("grpo", "nullspace_energy"), "e"),
            ("ewc", "+ EWC", ("grpo", "ewc_lambda"), r"\lambda"), ("agem", "+ A-GEM", None, ""),
            ("router", "Router (uncal.)", None, ""),
            ("tripwire_prompt", r"\methodp{}", ("tripwire", "eps"), r"\varepsilon"),
            ("tripwire", r"\methoda{}", ("tripwire", "eps"), r"\varepsilon")]
SPEC = {f: (lab, key, sym) for f, lab, key, sym in FAMILIES}
GATED = ("router", "tripwire_prompt", "tripwire")
TAGCHAR = {"kl_beta": "b", "refkl_beta": "b", "replay_frac": "r", "olora_lambda": "o", "nullspace_energy": "e",
           "ewc_lambda": "l"}


def read_rows(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows.sort(key=lambda r: r["i"])
    return rows


def halves(n, seed=0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return sorted(idx[: n // 2]), sorted(idx[n // 2:])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def family_of(info, md):
    """(family, setting, method, tag) of a run folder, or None for runs this report leaves out"""
    fam, cfg = info.get("method"), info["config"]
    g, tw = cfg.get("grpo", {}), cfg.get("tripwire", {})
    if fam not in SPEC or cfg.get("lora", {}).get("dense") or g.get("objective", "grpo") != "grpo":
        return None
    if tw.get("protect") or (fam in GATED and tw.get("activation", "decision") == "sequence"):
        return None
    if fam == "refkl" and int(g.get("refkl_samples", 1)) > 1 or fam == "nullspace" and int(g.get("nullspace_refresh", 0)):
        return None
    key = SPEC[fam][1]
    param = float(cfg[key[0]][key[1]]) if key else None
    base = os.path.basename(md)
    tag = base[len(fam) + 1:] if base.startswith(fam + "_") else ""
    return fam, param, fam, tag


def stage_acc(md, k, task, idx=None):
    rows = read_rows(os.path.join(md, f"stage{k}", "eval", f"{task}.jsonl"))
    if rows is None:
        return None
    sel = rows if idx is None else [rows[i] for i in idx if i < len(rows)]
    return mean(float(r["m"]["correct"]) for r in sel)


def run_metrics(md, tasks):
    K = len(tasks)
    for k in (0, K):
        if not os.path.exists(os.path.join(md, f"stage{k}", "eval", "DONE")):
            return None
    out = {}
    n_t = {t: len(read_rows(os.path.join(md, f"stage{K}", "eval", f"{t}.jsonl")) or []) for t in tasks}
    if not all(n_t.values()):
        return None
    hv = {t: halves(n_t[t]) for t in tasks}
    d0, d1 = (read_rows(os.path.join(md, f"stage{k}", "eval", "dolly.jsonl")) for k in (0, K))
    dh = halves(len(d0)) if d0 else ([], [])
    for h, name in ((0, "val"), (1, "test"), (None, "all")):
        idx = {t: (None if h is None else hv[t][h]) for t in tasks}
        accs = [stage_acc(md, K, t, idx[t]) for t in tasks]
        out[f"avg_{name}"] = mean(accs) if all(a is not None for a in accs) else None
        bw = []
        for j, t in enumerate(tasks[:-1], start=1):
            a_j = stage_acc(md, j, t, idx[t])
            if a_j is None or accs[j - 1] is None:
                bw = None
                break
            bw.append(accs[j - 1] - a_j)
        out[f"bwt_{name}"] = mean(bw) if bw else None
        if d0 and d1:
            di = range(len(d0)) if h is None else dh[h]
            out[f"chg_{name}"] = mean(float(d1[i]["out"] != d0[i]["out"]) for i in di)
    for p, key, f in (("advbench", "adv", "refused"), ("mmlu", "mmlu", "correct")):
        a0, a1 = (read_rows(os.path.join(md, f"stage{k}", "eval", f"{p}.jsonl")) for k in (0, K))
        if a0 and a1:
            out[key] = mean(r["m"][f] for r in a1) - mean(r["m"][f] for r in a0)
    return out


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return {"mean": m, "sd": sd, "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs/<exp>")
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_25"))
    ap.add_argument("--budgets", default="0.1,0.3")
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    exp = os.path.basename(os.path.normpath(args.runs))
    seeds = sorted((d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
    by, specs, tasks_seen, gls = {}, {}, None, Counter()
    for sd in seeds:
        for d in sorted(os.listdir(os.path.join(args.runs, sd))):
            md = os.path.join(args.runs, sd, d)
            cj = os.path.join(md, "config.json")
            if not (os.path.exists(cj) and os.path.exists(os.path.join(md, "DONE"))):
                continue
            info = json.load(open(cj))
            fp = family_of(info, md)
            if fp is None:
                continue
            tasks = list(info["config"]["stream"])
            tasks_seen = tasks_seen or tasks
            if tasks != tasks_seen:
                continue
            m = run_metrics(md, tasks)
            if m is None:
                continue
            gl = int(info["config"].get("gate_layer", 0))
            gls[gl] += 1
            by.setdefault((fp[0], fp[1], gl), []).append((sd, m))
            specs[(fp[0], fp[1])] = (fp[2], fp[3])
    if not by:
        raise SystemExit(f"no finished runs under {args.runs}")
    base_gl = gls.most_common(1)[0][0]
    keys = ("avg_val", "avg_test", "avg_all", "bwt_val", "bwt_test", "bwt_all", "chg_val", "chg_test", "chg_all",
            "adv", "mmlu")
    points = {(f, p): {"seeds": [sd for sd, _ in v], **{k: agg(m.get(k) for _, m in v) for k in keys}}
              for (f, p, gl), v in by.items() if gl == base_gl}
    order = [f for f, *_ in FAMILIES if any(fp[0] == f for fp in points)]
    res = {"exp": exp, "tasks": tasks_seen, "budgets": budgets, "points": [], "chosen": {}}
    for (f, p), a in sorted(points.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1] or 0)):
        res["points"].append({"family": f, "param": p, "run": "_".join(x for x in specs[(f, p)] if x), **a})
    val = lambda a, k: a[k]["mean"] if a.get(k) else None  # noqa: E731
    for b in budgets + ["best"]:
        ch = {}
        for f in order:
            cand = [(p, a) for (ff, p), a in points.items() if ff == f and val(a, "avg_val") is not None
                    and (b == "best" or (val(a, "chg_val") is not None and val(a, "chg_val") <= b + 1e-12))]
            if cand:
                p, a = max(cand, key=lambda pa: val(pa[1], "avg_val"))
                ch[f] = {"param": p, "run": "_".join(x for x in specs[(f, p)] if x), **a}
        res["chosen"][str(b)] = ch
    res["predictions"] = predictions(res)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"stream_{exp}.json"), "w"), indent=1)
    lines = []
    for f, c in res["chosen"]["best"].items():   # the best setting of every ungated family, for the record
        if f in GATED:
            continue
        method, tag = specs[(f, c["param"])]
        key = SPEC[f][1]
        ovr = f"{key[0]}.{key[1]}={c['param']:g}" if (key and tag) else ""
        lines.append(f"{method}:{tag}:{ovr}")
    open(os.path.join(args.out, f"chosen_{exp}.txt"), "w").write("\n".join(lines) + "\n")
    txt = summary(res)
    open(os.path.join(args.out, f"stream_{exp}_summary.txt"), "w").write(txt)
    write_table(res, args.out)
    print(txt + f"wrote {args.out}/stream_{exp}.json, _table.tex, _summary.txt and chosen_{exp}.txt")


def predictions(res):
    """PLAN_PAPER3.md, step p3.25 (written Nov 5, before any run of the step)"""
    pts = res["points"]
    m = lambda a, k: a[k]["mean"] if a.get(k) else None  # noqa: E731
    ung = [a for a in pts if a["family"] not in GATED]
    twp = [a for a in pts if a["family"] == "tripwire_prompt" and a["param"] is not None and abs(a["param"] - 0.03) < 1e-9]
    out = []
    low = [m(a, "chg_all") for a in ung if m(a, "chg_all") is not None]
    out.append(("P1 no ungated setting of the sweep changes at most 10% of held-out Dolly responses",
                bool(low) and min(low) > 0.10, {"lowest ungated change": min(low) if low else None}))
    best = res["chosen"]["best"]
    if twp and best:
        t = m(twp[0], "avg_test")
        top = max((m(c, "avg_test") for f, c in best.items() if f not in GATED and m(c, "avg_test") is not None),
                  default=None)
        out.append(("P2 no ungated family at its best setting exceeds the gated model's test Avg by more than 3 points",
                    t is not None and top is not None and top - t <= 0.03, {"gated": t, "best ungated": top}))
        bw = m(twp[0], "bwt_all")
        out.append(("P3 the gated model's BWT is within 1 point of zero (earlier tasks change only where a later gate "
                    "fires)", bw is not None and abs(bw) <= 0.01, {"bwt": bw}))
    return [{"text": a, "pass": bool(b), "values": c} for a, b, c in out]


def pc(v, d=1, signed=False):
    if v is None:
        return "--"
    x = 100 * v
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s.replace("-", "$-$", 1) if signed else s


def setting(f, p):
    sym = SPEC[f][2]
    return "" if p is None or not sym else f"${sym}{{=}}{p:g}$"


def summary(res):
    L = [f"p3.25: {res['exp']} ({' -> '.join(res['tasks'])}); every setting (mean over seeds, all prompts)"]
    for a in res["points"]:
        g = lambda k: (a.get(k) or {}).get("mean")  # noqa: E731
        L.append(f"  {a['run']:28s} seeds {len(a['seeds'])}: Avg {pc(g('avg_all'))}  BWT {pc(g('bwt_all'), 1, True)}  "
                 f"Dolly chg {pc(g('chg_all'))}  AdvBench {pc(g('adv'), 1, True)}  MMLU {pc(g('mmlu'), 1, True)}")
    for b, ch in res["chosen"].items():
        L.append(f"chosen ({'largest Avg' if b == 'best' else f'largest Avg with Dolly change <= {float(b):.0%}'}), "
                 "test half:")
        for f, c in ch.items():
            g = lambda k: (c.get(k) or {}).get("mean")  # noqa: E731
            L.append(f"  {c['run']:28s} Avg {pc(g('avg_test'))}  BWT {pc(g('bwt_test'), 1, True)}  Dolly chg {pc(g('chg_test'))}")
    L.append("predictions of PLAN_PAPER3.md, step p3.25 (written before the runs)")
    for p in res["predictions"]:
        L.append(f"{p['text']}: {'PASS' if p['pass'] else 'FAIL'}")
    return "\n".join(L) + "\n"


def write_table(res, out):
    tasks = r" $\to$ ".join({"gsm8k": "GSM8K", "countdown": "Countdown", "kk": r"K\&K", "math": "MATH",
                             "sudoku": "Sudoku"}.get(t, t) for t in res["tasks"])
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Continual-learning baselines at their best settings on the stream " + tasks + r" (" +
         res["exp"].replace("_", r"\_") + r"). Each family's setting is chosen on half of the evaluation prompts and "
         r"reported on the other half. Best: largest Avg; within 30\%: largest Avg changing at most 30\% of held-out "
         r"Dolly responses. \#: settings tried; Avg: mean accuracy after the stream (\%); BWT: backward transfer "
         r"(points); Chg.: Dolly responses changed (\%); --: no setting within the budget.}",
         r"\label{tab:clsweep}", r"\begin{tabular}{l r l rrr l rrr}", r"\toprule",
         r" & & \multicolumn{4}{c}{best} & \multicolumn{4}{c}{within 30\%} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Method & \# & Setting & Avg & BWT & Chg. & Setting & Avg & BWT & Chg. \\", r"\midrule"]
    tried = Counter(a["family"] for a in res["points"])
    for f, lab, *_ in FAMILIES:
        if f not in tried:
            continue
        cells = []
        for b in ("best", "0.3"):
            c = res["chosen"].get(b, {}).get(f)
            if c is None:
                cells += ["--", "--", "--", "--"]
                continue
            g = lambda k: (c.get(k) or {}).get("mean")  # noqa: E731
            cells += [setting(f, c["param"]) or "--", pc(g("avg_test")), pc(g("bwt_test"), 1, True), pc(g("chg_test"))]
        L.append(f"{lab} & {tried[f]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, f"stream_{res['exp']}_table.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
