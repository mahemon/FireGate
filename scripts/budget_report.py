#!/usr/bin/env python
"""Step p3.15 (Nov 2, PLAN_PAPER3.md): every method at a matched budget of changed reference responses (CPU).

    python scripts/budget_report.py --runs runs/v2_overlap_q15 --out paper/v3_results/p3_15 [--budgets 0.03,0.1,0.3,1]

Reviewers ask whether the ungated baselines were tuned to the same budget as the gate. This reads the stored
evaluations of every finished run of one experiment (each method family at every setting trained or scaled, every
seed) and, per budget b on the share of held-out Dolly responses that differ from the run's own base model, picks for
each family the setting with the largest new-task gain among those whose change is at most b. The choice is made on
a validation half of the Dolly and new-task prompts (a fixed random half, seed 0) and the chosen setting is reported
on the other half, so the selection does not inflate the gain. AdvBench refusal and MMLU changes of the chosen
setting are reported on the whole pools (not used for the choice). A family with no setting within b gets none: no
operating point is interpolated. Setting families and run folders are those of scripts/make_overlap_report.py.
Outputs in --out: budget_<exp>.json, budget_<exp>_table.tex, budget_<exp>_summary.txt.
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
sys.path.insert(0, HERE)

from make_overlap_report import FAMILIES, CERT_FAMS, rows, mean  # noqa: E402

NAMES = {f: n for f, n, _ in FAMILIES}


def family_of(info):
    """(family, param) of a run folder's config.json, as scripts/make_overlap_report.py groups them, or None"""
    fam = info.get("method")
    cfg = info["config"]
    g = cfg.get("grpo", {})
    if fam in ("tripwire_prompt", "router") and cfg.get("tripwire", {}).get("activation", "decision") == "sequence":
        fam += "_seq"
    if fam == "refkl" and int(g.get("refkl_samples", 1)) > 1:
        fam = "refkl_s"
    if fam == "nullspace" and int(g.get("nullspace_refresh", 0)) > 0:
        fam = "nullspace_rf"
    if fam in ("refkl", "tripwire_prompt") and cfg.get("tripwire", {}).get("protect"):
        fam += "_protect"
    if cfg.get("lora", {}).get("dense") or g.get("objective", "grpo") != "grpo":
        return None
    spec = next((f for f in FAMILIES if f[0] == fam), None)
    if spec is None or (fam == "scale" and info.get("base_run", "seq") != "seq"):
        return None
    param = None
    if spec[2] is not None:
        param = float(info["alpha"]) if spec[2][0] == "__alpha__" else float(cfg[spec[2][0]][spec[2][1]])
    return fam, param


def halves(n, seed=0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[: n // 2]), set(idx[n // 2:])


def run_metrics(md, task, K):
    """per-half gain and Dolly change, whole-pool refusal and MMLU change; None if the run is not finished"""
    e0, e1 = os.path.join(md, "stage0", "eval"), os.path.join(md, f"stage{K}", "eval")
    if not (os.path.exists(os.path.join(e0, "DONE")) and os.path.exists(os.path.join(e1, "DONE"))):
        return None
    get = lambda d, p: rows(os.path.join(d, f"{p}.jsonl")) if os.path.exists(os.path.join(d, f"{p}.jsonl")) else None  # noqa: E731
    t0, t1, d0, d1 = get(e0, task), get(e1, task), get(e0, "dolly"), get(e1, "dolly")
    if not (t0 and t1 and d0 and d1):
        return None
    tv, tt = halves(len(t0))
    dv, dt = halves(len(d0))
    gain = lambda S: mean(t1[i]["m"]["correct"] - t0[i]["m"]["correct"] for i in sorted(S))  # noqa: E731
    chg = lambda S: mean(float(d1[i]["out"] != d0[i]["out"]) for i in sorted(S))  # noqa: E731
    out = {"gain_val": gain(tv), "gain_test": gain(tt), "chg_val": chg(dv), "chg_test": chg(dt),
           "gain_all": gain(range(len(t0))), "chg_all": chg(range(len(d0)))}
    for p, key, f in (("advbench", "adv", "refused"), ("mmlu", "mmlu", "correct")):
        a0, a1 = get(e0, p), get(e1, p)
        if a0 and a1:
            out[key] = mean(r["m"][f] for r in a1) - mean(r["m"][f] for r in a0)
    return out


def agg(vals):
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return {"mean": m, "sd": sd, "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs/<exp>")
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_15"))
    ap.add_argument("--budgets", default="0.03,0.1,0.3,1")
    args = ap.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    exp = os.path.basename(os.path.normpath(args.runs))
    seeds = sorted((d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
    by_gl, tasks = {}, set()
    for sd in seeds:
        for d in sorted(os.listdir(os.path.join(args.runs, sd))):
            md = os.path.join(args.runs, sd, d)
            cj = os.path.join(md, "config.json")
            if not os.path.exists(cj):
                continue
            info = json.load(open(cj))
            fp = family_of(info)
            if fp is None:
                continue
            task = info["config"].get("stream", ["nq"])[0]
            m = run_metrics(md, task, len(info["config"].get("stream", ["nq"])))
            if m is None:
                continue
            tasks.add(task)
            by_gl.setdefault((fp, int(info["config"].get("gate_layer", 0))), []).append(m)
    if not by_gl:
        raise SystemExit(f"no finished runs under {args.runs}")
    base_gl = Counter(gl for (_, gl), v in by_gl.items() for _ in v).most_common(1)[0][0]
    runs = {fp: v for (fp, gl), v in by_gl.items() if gl == base_gl}
    keys = ("gain_val", "gain_test", "chg_val", "chg_test", "gain_all", "chg_all", "adv", "mmlu")
    points = {fp: {k: agg(r.get(k) for r in v) for k in keys} for fp, v in runs.items()}
    order = [f for f, _, _ in FAMILIES if any(fp[0] == f for fp in points)]
    res = {"exp": exp, "task": "/".join(sorted(tasks)), "budgets": budgets, "points": [], "chosen": {}}
    for (f, p), a in sorted(points.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1] or 0)):
        res["points"].append({"family": f, "param": p, **a})
    for b in budgets:
        ch = {}
        for f in order:
            cand = [(p, a) for (ff, p), a in points.items() if ff == f and a["chg_val"]["mean"] <= b + 1e-12]
            if cand:
                p, a = max(cand, key=lambda pa: pa[1]["gain_val"]["mean"])
                ch[f] = {"param": p, **a}
        res["chosen"][str(b)] = ch
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"budget_{exp}.json"), "w"), indent=1)
    txt = summary(res)
    open(os.path.join(args.out, f"budget_{exp}_summary.txt"), "w").write(txt)
    write_table(res, args.out)
    print(txt + f"wrote {args.out}/budget_{exp}.json, _table.tex, _summary.txt")


def label(f, p):
    spec = next(s for ff, _, s in FAMILIES if ff == f)
    return NAMES[f] if spec is None or p is None else f"{NAMES[f]} ${spec[2]}{{=}}{p:g}$"


def plain(f, p):
    return f if p is None else f"{f} {p:g}"


def summary(res):
    pc = lambda a: "--" if a is None else f"{100 * a['mean']:.1f}"  # noqa: E731
    pt = lambda a: "--" if a is None else f"{100 * a['mean']:+.1f}"  # noqa: E731
    L = [f"{res['exp']} ({res['task']}): every trained setting, mean over seeds (validation / test halves)"]
    for q in res["points"]:
        L.append(f"  {plain(q['family'], q['param']):28s} n {q['gain_val']['n']}  gain {pt(q['gain_val'])} / "
                 f"{pt(q['gain_test'])}  Dolly changed {pc(q['chg_val'])} / {pc(q['chg_test'])}%  "
                 f"AdvB {pt(q['adv'])}  MMLU {pt(q['mmlu'])}")
    for b, ch in res["chosen"].items():
        L.append(f"budget {100 * float(b):g}% of Dolly responses changed (validation half): "
                 + ("no family" if not ch else ""))
        for f, a in sorted(ch.items(), key=lambda kv: -kv[1]["gain_test"]["mean"]):
            L.append(f"  {plain(f, a['param']):28s} test gain {pt(a['gain_test'])}  test Dolly changed "
                     f"{pc(a['chg_test'])}%  AdvB {pt(a['adv'])}  MMLU {pt(a['mmlu'])}")
    ch = res["chosen"].get(str(res["budgets"][0]), {})
    gated = [f for f in ch if f in CERT_FAMS]
    if ch and gated:
        best = max(ch, key=lambda f: ch[f]["gain_test"]["mean"])
        ok = best in CERT_FAMS
        L.append(f"pre-registered prediction (PLAN_PAPER3.md, step p3.15): at the smallest budget a certified gate "
                 f"keeps the largest test gain: {'PASS' if ok else 'FAIL'} (best: {plain(best, ch[best]['param'])})")
    else:
        L.append("pre-registered prediction (step p3.15): n/a (no certified gate within the smallest budget)")
    return "\n".join(L) + "\n"


def write_table(res, out):
    tp = lambda a: "--" if a is None else (f"{100 * a['mean']:+.1f}".replace("-", "$-$"))  # noqa: E731
    pc = lambda a: "--" if a is None else f"{100 * a['mean']:.1f}"  # noqa: E731
    bud = [b for b in res["budgets"]]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Every method at a matched budget (" + res["task"].upper() + r", " + res["exp"].replace("_", r"\_")
         + r"). For a budget $b$ on the share of held-out Dolly responses that change, each method's setting is the "
         r"one with the largest new-task gain among its trained settings within $b$, chosen on a random half of the "
         r"prompts; gain (points) and Dolly change (\%) are reported on the other half, mean over seeds. "
         r"--: no trained setting within $b$.}",
         r"\label{tab:budget}", r"\begin{tabular}{l " + "rr" * len(bud) + "}", r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$b={100 * b:g}\\%$}}" for b in bud) + r" \\",
         "Method & " + " & ".join("Gain & Chg." for _ in bud) + r" \\", r"\midrule"]
    fams = [f for f, _, _ in FAMILIES if any(f in ch for ch in res["chosen"].values())]
    for f in fams:
        cells = []
        for b in bud:
            a = res["chosen"][str(b)].get(f)
            cells += ["--", "--"] if a is None else [tp(a["gain_test"]), pc(a["chg_test"])]
        L.append(f"{NAMES[f]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, f"budget_{res['exp']}_table.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
