"""Harmful requests written by others, for the pre-registered evaluation (step p3.12, PLAN_PAPER3.md; no torch).

The stress families of twcore/stress.py rewrite AdvBench requests in forms we wrote. These sets come from public
benchmarks we did not write, and no gate, setting or component was chosen after seeing them:
  ext_hb      HarmBench standard behaviours: plain harmful requests
  ext_hb_ctx  HarmBench contextual behaviours: a context passage, then a harmful request about it (HarmBench's own
              layout, passage, a "---" line, request), the external counterpart of the passage forms
  ext_sr      StrongREJECT forbidden prompts, rows taken from AdvBench removed
Rows live in data/ext_*.jsonl ({"id", "prompt", "context", "source"}), written by scripts/prepare_external.py, which
also drops every request whose normalised text equals an AdvBench request. A missing file means the set is skipped.
Scoring is refusal_format_score, as for the stress families (refusal as for AdvBench, and task-format leak).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from .tasks import POOLS, Pool, refusal_format_score

EXTERNAL: Dict[str, str] = {"ext_hb": "ext_harmbench.jsonl", "ext_hb_ctx": "ext_harmbench_ctx.jsonl",
                            "ext_sr": "ext_strongreject.jsonl"}
LABEL = {"ext_hb": "HarmBench", "ext_hb_ctx": "HarmBench contextual", "ext_sr": "StrongREJECT"}


def norm(text: str) -> str:
    """lower case, letters and digits only: the key for dropping AdvBench duplicates"""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def external_path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, EXTERNAL[name])


def external_rows(cfg: dict, name: str) -> List[dict]:
    """The rows of one external set in file order ([] if the file is missing)."""
    path = external_path(cfg["data_dir"], name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def external_pools(cfg: dict) -> List[str]:
    """The external sets present in the config's data folder."""
    return [n for n in EXTERNAL if os.path.exists(external_path(cfg["data_dir"], n))]


def _plain(ex: dict) -> str:
    return ex["prompt"].strip()


def _with_context(ex: dict) -> str:
    return f"{ex['context'].strip()}\n\n---\n\n{ex['prompt'].strip()}"


for _name, _fn in (("ext_hb", _plain), ("ext_hb_ctx", _with_context), ("ext_sr", _plain)):
    POOLS[_name] = Pool(_name, _fn, refusal_format_score, source=_name)
