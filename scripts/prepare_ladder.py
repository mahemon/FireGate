#!/usr/bin/env python
"""Data for the overlap-ladder tasks (no download: uses data/dolly.jsonl written by prepare_data.py).

    python scripts/prepare_ladder.py --data data
    python scripts/prepare_ladder.py --data data --only-missing      # write only files that do not exist yet

Dolly rows 0..1999 are left to the reference (fit / calibration / evaluation use rows 0..1699). The ifc task uses
rows 2000..6099 and the concise task rows 6100..10199: 3600 training rows each (RL, tripwire fitting, earlier-task
fitting and calibration) and 500 test rows. Each ifc example gets one constraint, cycling through the 8 kinds.
concisectx (v6): the Dolly rows from 2000 onward whose category has a context passage (closed QA, information
extraction, summarization; non-empty context), in order: 3000 training rows (RL and tripwire fitting; a single-task
experiment needs no earlier-task rows) and 500 test rows (about 3900 such rows exist).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.ladder import CONSTRAINT_ORDER, CTX_CATEGORIES  # noqa: E402

N_TRAIN, N_TEST = 3600, 500


ONLY_MISSING = False
N_TRAIN_CTX = 3000


def write_jsonl(path, rows):
    if ONLY_MISSING and os.path.exists(path):
        print(f"  kept {path} (exists)")
        return
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--only-missing", action="store_true", help="write only the files that do not exist yet")
    args = ap.parse_args()
    global ONLY_MISSING
    ONLY_MISSING = args.only_missing
    with open(os.path.join(args.data, "dolly.jsonl")) as f:
        dolly = [json.loads(l) for l in f if l.strip()]
    need = 10200
    if len(dolly) < need:
        raise SystemExit(f"data/dolly.jsonl has {len(dolly)} rows, need {need}")
    ifc = []
    for j, ex in enumerate(dolly[2000:2000 + N_TRAIN + N_TEST]):
        r = dict(ex)
        r["id"] = f"ifc_{j}"
        r["constraint"] = CONSTRAINT_ORDER[j % len(CONSTRAINT_ORDER)]
        ifc.append(r)
    concise = []
    for j, ex in enumerate(dolly[6100:6100 + N_TRAIN + N_TEST]):
        r = dict(ex)
        r["id"] = f"concise_{j}"
        concise.append(r)
    write_jsonl(os.path.join(args.data, "ifc_train.jsonl"), ifc[:N_TRAIN])
    write_jsonl(os.path.join(args.data, "ifc_test.jsonl"), ifc[N_TRAIN:])
    write_jsonl(os.path.join(args.data, "concise_train.jsonl"), concise[:N_TRAIN])
    write_jsonl(os.path.join(args.data, "concise_test.jsonl"), concise[N_TRAIN:])
    ctx = []
    for i, ex in enumerate(dolly[2000:]):
        if ex.get("category") in CTX_CATEGORIES and (ex.get("context") or "").strip():
            r = dict(ex)
            r["id"] = f"concisectx_{len(ctx)}"
            ctx.append(r)
    if len(ctx) < N_TRAIN_CTX + 500:   # configs/v2/ladder_concisectx_q15.yaml evaluates 500 test rows
        raise SystemExit(f"concisectx: only {len(ctx)} Dolly rows with a context passage (need {N_TRAIN_CTX + 500})")
    write_jsonl(os.path.join(args.data, "concisectx_train.jsonl"), ctx[:N_TRAIN_CTX])
    write_jsonl(os.path.join(args.data, "concisectx_test.jsonl"), ctx[N_TRAIN_CTX:N_TRAIN_CTX + 500])


if __name__ == "__main__":
    main()
