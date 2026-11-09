#!/usr/bin/env python
"""Progress of every method of an experiment (no torch).

    python scripts/status.py --runs runs/stream_qwen15
"""
from __future__ import annotations

import argparse
import json
import os


def last_line(path):
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/stream_qwen15",
                    help="an experiment folder (method folders inside, or s<seed> folders holding them)")
    args = ap.parse_args()
    if not os.path.isdir(args.runs):
        print("no runs in", args.runs)
        return
    seeds = sorted(d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit()
                   and os.path.isdir(os.path.join(args.runs, d)))
    if seeds:  # v2 layout: <exp>/s<seed>/<method>
        for sd in seeds:
            print(f"== {sd}")
            show(os.path.join(args.runs, sd))
    else:
        show(args.runs)


def show(root):
    for m in sorted(os.listdir(root)):
        md = os.path.join(root, m)
        if not os.path.isdir(os.path.join(md, "stage0")):  # not a method run (e.g. a paper/ folder)
            continue
        done = os.path.exists(os.path.join(md, "DONE"))
        stages = sorted(int(d[5:]) for d in os.listdir(md) if d.startswith("stage") and d[5:].isdigit())
        parts = []
        for k in stages:
            sd = os.path.join(md, f"stage{k}")
            if os.path.exists(os.path.join(sd, "eval", "DONE")):
                summ = json.load(open(os.path.join(sd, "eval", "summary.json")))
                p = summ["pools"]
                acc = " ".join(f"{t}={100 * p[t]['correct']:.1f}" for t in ("gsm8k", "countdown", "kk", "nq", "math", "ifeval")
                               if t in p and "correct" in p[t])
                parts.append(f"stage{k}: done ({acc}; adv-refusal={100 * p.get('advbench', {}).get('refused', float('nan')):.1f})")
                continue
            tl = last_line(os.path.join(sd, "train", "log.jsonl"))
            if tl:
                parts.append(f"stage{k}: training step {tl.get('step')} reward={tl.get('reward', 0):.3f}"
                             + (f" active={tl['active']:.2f}" if "active" in tl else ""))
            elif os.path.isdir(os.path.join(sd, "tripwire")):
                parts.append(f"stage{k}: building tripwire")
            else:
                parts.append(f"stage{k}: evaluating / starting")
        print(f"{m:16s} {'DONE ' if done else 'running'}  " + " | ".join(parts))


if __name__ == "__main__":
    main()
