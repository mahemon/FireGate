#!/usr/bin/env python
"""Data for the 5-task stream's two extra tasks. Run once on the login node (MATH is downloaded):

    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python scripts/prepare_stream5.py --data data

math    Hendrycks MATH, train and test splits pooled, levels 1-4, only problems whose answer (the last \\boxed{} of
        the reference solution) is a plain number; shuffled with a fixed seed; 3600 training and 500 test rows.
        Sources tried in order: EleutherAI/hendrycks_math (7 subject configs), DigitalLearningGmbH/MATH-lighteval,
        qwedsacf/competition_math.
sudoku  4x4 puzzles with a unique solution and 6-11 blanks (generated, seed 777); 3600 training and 500 test rows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.gen_sudoku import make_instances as make_sudoku  # noqa: E402
from twcore.stream5 import math_number  # noqa: E402
from twcore.tasks import last_boxed  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
N_TRAIN, N_TEST = 3600, 500
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus"]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(rows)} rows)")


def _load(name, config, split):
    from datasets import load_dataset
    try:
        return load_dataset(name, config, split=split, token=False)
    except TypeError:
        return load_dataset(name, config, split=split)


def math_rows():
    """All MATH problems (train + test) as dicts with problem, level, type, solution."""
    errors = []
    try:
        rows = []
        for sub in SUBJECTS:
            for split in ("train", "test"):
                rows += [dict(r) for r in _load("EleutherAI/hendrycks_math", sub, split)]
        return rows, "EleutherAI/hendrycks_math"
    except Exception as e:  # noqa: BLE001
        errors.append(f"EleutherAI/hendrycks_math: {e}")
    for name, splits in (("DigitalLearningGmbH/MATH-lighteval", ("train", "test")),
                         ("qwedsacf/competition_math", ("train",))):
        try:
            rows = []
            for split in splits:
                rows += [dict(r) for r in _load(name, None, split)]
            return rows, name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise SystemExit("could not download MATH:\n  " + "\n  ".join(errors))


def prepare_math(data):
    rows, source = math_rows()
    out, seen = [], set()
    for r in rows:
        level = str(r.get("level", ""))
        if level not in ("Level 1", "Level 2", "Level 3", "Level 4"):
            continue
        ans = math_number(last_boxed(r["solution"]))
        if ans is None or r["problem"] in seen:
            continue
        seen.add(r["problem"])
        out.append({"problem": r["problem"], "answer": str(ans), "level": int(level.split()[-1]),
                    "type": r.get("type", "")})
    out.sort(key=lambda r: r["problem"])          # independent of the download order
    random.Random(2027).shuffle(out)
    print(f"  MATH from {source}: {len(rows)} problems, {len(out)} at levels 1-4 with a numeric answer")
    if len(out) < N_TRAIN + N_TEST:
        raise SystemExit(f"need {N_TRAIN + N_TEST} MATH rows, found {len(out)}")
    for i, r in enumerate(out):
        r["id"] = f"math_{i}"
    write_jsonl(os.path.join(data, "math_train.jsonl"), out[:N_TRAIN])
    write_jsonl(os.path.join(data, "math_test.jsonl"), out[N_TRAIN:N_TRAIN + N_TEST])


def prepare_sudoku(data):
    rows = make_sudoku(N_TRAIN + N_TEST, seed=777)
    random.Random(778).shuffle(rows)               # blank counts mixed in both splits
    write_jsonl(os.path.join(data, "sudoku_train.jsonl"), rows[:N_TRAIN])
    write_jsonl(os.path.join(data, "sudoku_test.jsonl"), rows[N_TRAIN:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--only", choices=["math", "sudoku"], default=None)
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    if args.only in (None, "sudoku"):
        prepare_sudoku(args.data)
    if args.only in (None, "math"):
        prepare_math(args.data)


if __name__ == "__main__":
    main()
