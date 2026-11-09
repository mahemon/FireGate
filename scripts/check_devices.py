#!/usr/bin/env python
"""Which GPU model evaluated each stage of each run (no torch).

Greedy bf16 decoding is deterministic on one GPU model but not across GPU models, so two methods' stage-0 (base
model) numbers agree only if both ran on the same model. This prints the device of every stage evaluation.

    python scripts/check_devices.py --runs runs/stream_qwen15_v2
    python scripts/check_devices.py --runs runs/v2_stream_q15        (seed folders are found automatically)
"""
from __future__ import annotations

import argparse
import json
import os


def devices(mdir):
    out = {}
    for d in sorted(os.listdir(mdir)):
        if not (d.startswith("stage") and d[5:].isdigit()):
            continue
        edir = os.path.join(mdir, d, "eval")
        if not os.path.isdir(edir):
            continue
        devs = set()
        for fn in os.listdir(edir):
            if fn.endswith(".jsonl"):
                with open(os.path.join(edir, fn)) as f:
                    line = f.readline()
                if line.strip():
                    devs.add(json.loads(line).get("dev", "?"))
        out[int(d[5:])] = ", ".join(sorted(devs)) or "?"
    return out


def show(root, indent=""):
    for m in sorted(os.listdir(root)):
        md = os.path.join(root, m)
        if os.path.isdir(os.path.join(md, "stage0")):
            dv = devices(md)
            print(f"{indent}{m:18s} " + " | ".join(f"stage{k}: {v}" for k, v in sorted(dv.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    seeds = sorted(d for d in os.listdir(args.runs) if d.startswith("s") and d[1:].isdigit())
    if seeds:
        for s in seeds:
            print(f"== {s}")
            show(os.path.join(args.runs, s), "  ")
    else:
        show(args.runs)


if __name__ == "__main__":
    main()
