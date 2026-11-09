"""Data splits used by the stream (no torch). Files are written by scripts/prepare_data.py.

Stream task t:   data/<t>_train.jsonl  ->  rl / fit_new / fit_old / cal_old index ranges (config "splits")
                 data/<t>_test.jsonl   ->  first pools[t] rows are the held-out evaluation pool
Reference pools: data/dolly.jsonl, data/advbench.jsonl, data/general.jsonl -> [fit | calibration | evaluation]
Evaluation only: data/xstest.jsonl, data/mmlu_eval.jsonl, data/wildchat_eval.jsonl (Nov 1: real-user requests,
                 first turns of WildChat conversations; a config may list it in pools, and scripts/eval_extra.py
                 --what wildchat evaluates finished runs on its first WILDCHAT_N rows)
"""
from __future__ import annotations

import os
from typing import Dict, List

from .io_utils import read_jsonl

TASK_FILES = {"gsm8k": "gsm8k", "countdown": "countdown", "kk": "kk", "nq": "nq", "ifc": "ifc", "concise": "concise",
              "math": "math", "sudoku": "sudoku", "concisectx": "concisectx", "code": "code", "squad2": "squad2",
              "trivia": "trivia"}
REF_FILES = {"dolly": "dolly.jsonl", "advbench": "advbench.jsonl", "general": "general.jsonl"}
EVAL_ONLY = {"xstest": "xstest.jsonl", "mmlu": "mmlu_eval.jsonl", "wildchat": "wildchat_eval.jsonl"}
WILDCHAT_N = 500   # WildChat requests evaluated for runs whose config does not list the pool (eval_extra, gate features)

_cache: Dict[str, List[dict]] = {}


def _load(path: str) -> List[dict]:
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing; run scripts/prepare_data.py first")
        _cache[path] = read_jsonl(path)
    return _cache[path]


def _take(rows: List[dict], a: int, b: int, what: str) -> List[dict]:
    if b > len(rows):
        raise ValueError(f"{what}: need rows [{a}:{b}] but only {len(rows)} exist")
    return rows[a:b]


def task_split(cfg: dict, task: str, which: str) -> List[dict]:
    a, b = cfg["splits"][which]
    rows = _load(os.path.join(cfg["data_dir"], f"{TASK_FILES[task]}_train.jsonl"))
    return _take(rows, a, b, f"{task} {which}")


def ref_split(cfg: dict, name: str, which: str) -> List[dict]:
    """which in {fit, cal, eval} for the reference pools (rows are used as [fit | calibration | evaluation])."""
    rows = _load(os.path.join(cfg["data_dir"], REF_FILES[name]))
    f = cfg["tripwire"]["fit_ref"][name]
    c = cfg["tripwire"]["cal_ref"][name]
    e = cfg["pools"].get(name, 0)
    rng = {"fit": (0, f), "cal": (f, f + c), "eval": (f + c, f + c + e)}[which]
    return _take(rows, rng[0], rng[1], f"{name} {which}")


def eval_pool(cfg: dict, name: str) -> List[dict]:
    n = cfg["pools"][name]
    if name in TASK_FILES:
        rows = _load(os.path.join(cfg["data_dir"], f"{TASK_FILES[name]}_test.jsonl"))
        return _take(rows, 0, n, f"{name} test")
    if name in REF_FILES:
        return ref_split(cfg, name, "eval")
    rows = _load(os.path.join(cfg["data_dir"], EVAL_ONLY[name]))
    return _take(rows, 0, n, name)


def eval_pools(cfg: dict) -> Dict[str, List[dict]]:
    # fixed order: the same prompts in the same batches at every stage (needed for the exactness check)
    return {name: eval_pool(cfg, name) for name in sorted(cfg["pools"])}


# Paper v3 (PLAN_PAPER3.md, step p3.7): Dolly requests that no pool of a non-ladder experiment uses. data/dolly.jsonl is
# shuffled once (seed 7), so these rows are drawn like the reference rows [0, 1700). The ladder tasks (IF, Concise,
# Concise-ctx) are built from rows 2000 onward and the stress passages come from rows 12000 onward, so the set is
# defined only for experiments without a ladder task.
FRESH_DOLLY = (2000, 12000)
LADDER_TASKS = ("ifc", "concise", "concisectx")


def dolly_fresh_rows(cfg: dict) -> List[dict]:
    """Held-out Dolly requests for a larger check of the certified level ([] for experiments with a ladder task)."""
    if any(t in LADDER_TASKS for t in cfg["stream"]) or "dolly" not in cfg["tripwire"]["cal_ref"]:
        return []
    a, b = FRESH_DOLLY
    return _take(_load(os.path.join(cfg["data_dir"], REF_FILES["dolly"])), a, b, "dolly fresh")


def wildchat_path(cfg: dict) -> str:
    return os.path.join(cfg["data_dir"], EVAL_ONLY["wildchat"])


def wildchat_rows(cfg: dict, n: int = WILDCHAT_N) -> List[dict]:
    """The first n WildChat requests (the same rows as a config's pool of that size)."""
    return _take(_load(wildchat_path(cfg)), 0, n, "wildchat")
