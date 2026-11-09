#!/usr/bin/env python
"""Per-seed accuracy after every stage and the reward curve of every training stage (CPU, pure Python).

    python scripts/seed_report.py --runs runs/v2_stream_q7 --methods tripwire_prompt_seq,tripwire_prompt,seq

Written to check single seeds behind a large standard deviation (Oct 26: the whole-sequence Tripwire-prompt on the
7B stream, Countdown 49.7 +- 42.9 over three seeds). For every seed folder s<k>/<method>/ it prints
  * the accuracy (%) on each task pool after each stage (stage<k>/eval/<task>.jsonl, metric "correct"), and the
    format rate where the rows have one;
  * per training stage (stage<k>/train/log.jsonl, "step" lines): steps logged, mean reward of the first and the last
    10 steps, the highest 10-step mean, the mean response length of the last 10 steps, and a flag when the reward
    of the last 10 steps fell below half of the highest 10-step mean (a collapse during training) or stayed near 0.
A run that is not finished is reported as far as it got.
"""
from __future__ import annotations

import argparse
import json
import os


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # a line cut by a job that was stopped
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "  --" if x is None else f"{100 * x:5.1f}"


def stages_of(mdir):
    ks = []
    for name in os.listdir(mdir):
        if name.startswith("stage") and name[5:].isdigit():
            ks.append(int(name[5:]))
    return sorted(ks)


def eval_table(mdir, ks):
    """{stage: {pool: (correct, format)}} for the pools that have a 'correct' metric."""
    out = {}
    for k in ks:
        edir = os.path.join(mdir, f"stage{k}", "eval")
        if not os.path.isfile(os.path.join(edir, "DONE")):
            continue
        res = {}
        for fn in sorted(os.listdir(edir)):
            if not fn.endswith(".jsonl"):
                continue
            rows = read_jsonl(os.path.join(edir, fn))
            ms = [r.get("m", {}) for r in rows]
            if not ms or "correct" not in ms[0]:
                continue
            fmt = mean(float(m["format"]) for m in ms if "format" in m) if "format" in ms[0] else None
            res[fn[:-6]] = (mean(float(m["correct"]) for m in ms), fmt)
        out[k] = res
    return out


def train_curve(mdir, k):
    path = os.path.join(mdir, f"stage{k}", "train", "log.jsonl")
    if not os.path.isfile(path):
        return None
    steps = [r for r in read_jsonl(path) if r.get("msg") == "step" and "reward" in r]
    if not steps:
        return None
    by_step = {}
    for r in steps:   # a resumed run logs some steps twice: keep the last
        by_step[int(r.get("step", len(by_step) + 1))] = r
    rs = [by_step[s] for s in sorted(by_step)]
    rew = [float(r["reward"]) for r in rs]
    win = [mean(rew[i:i + 10]) for i in range(0, max(1, len(rew) - 9))]
    first, last, peak = mean(rew[:10]), mean(rew[-10:]), max(win)
    rlen = mean(float(r["resp_len"]) for r in rs[-10:] if "resp_len" in r)
    flag = ""
    if peak is not None and last is not None and peak > 0.05 and last < 0.5 * peak:
        flag = "COLLAPSE (last 10 steps < half of the best 10)"
    elif last is not None and last < 0.02:
        flag = "near-zero reward"
    return len(rs), first, last, peak, rlen, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs/<exp> (with s<seed>/ folders)")
    ap.add_argument("--methods", required=True, help="comma-separated run folder names, e.g. tripwire_prompt_seq,seq")
    args = ap.parse_args()
    seeds = sorted(d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit())
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"\n===== {method}")
        for sd in seeds:
            mdir = os.path.join(args.runs, sd, method)
            if not os.path.isdir(mdir):
                continue
            ks = stages_of(mdir)
            done = "done" if os.path.isfile(os.path.join(mdir, "DONE")) else "NOT finished"
            gpu = ""
            cj = os.path.join(mdir, "config.json")
            if os.path.isfile(cj):
                try:
                    gpu = json.load(open(cj)).get("device", "")
                except (json.JSONDecodeError, OSError):
                    gpu = ""
            print(f"--- {sd}/{method} ({done}{', ' + gpu if gpu else ''})")
            ev = eval_table(mdir, ks)
            pools = sorted({p for r in ev.values() for p in r})
            if pools:
                print("  accuracy (%) after each stage" + (" [format rate]" if any(
                    v[1] is not None for r in ev.values() for v in r.values()) else ""))
                print("    stage " + " ".join(f"{p:>14s}" for p in pools))
                for k in sorted(ev):
                    cells = []
                    for p in pools:
                        c = ev[k].get(p)
                        if c is None:
                            cells.append(f"{'--':>14s}")
                        else:
                            cells.append(f"{pct(c[0]):>8s}" + (f" [{pct(c[1]).strip():>4s}]" if c[1] is not None
                                                               else " " * 7))
                    print(f"    {k:5d} " + " ".join(cells))
            for k in ks:
                if k == 0:
                    continue
                tc = train_curve(mdir, k)
                if tc is None:
                    continue
                n, first, last, peak, rlen, flag = tc
                print(f"  train stage {k}: {n} steps, reward first10 {first:.3f} last10 {last:.3f} best10 {peak:.3f}"
                      + (f", response length last10 {rlen:.0f}" if rlen is not None else "") + (f"  <-- {flag}" if flag
                                                                                           else ""))


if __name__ == "__main__":
    main()
