"""Prompt sets written by others, for the prospective test of step p3.19 (PLAN_PAPER3.md; no torch).

Gates and every setting are frozen (the recipe of p3.12). These sets are measured for gate firing only: firing bounds
how often any model behind the gate answers differently (Theorem 3.2), so no generation is needed.
  pr_mj_en     MultiJail (Deng et al., ICLR 2024) English harmful requests
  pr_mj_hi     the same requests in Chinese, Italian, Vietnamese (MultiJail's high-resource languages)
  pr_mj_mid    ... in Arabic, Korean, Thai (medium-resource)
  pr_mj_lo     ... in Bengali, Swahili, Javanese (low-resource)
  pr_oa_en     first user turns of OpenAssistant (oasst1) conversations in English: benign chat by other people
  pr_oa_ml     first user turns of oasst1 in every other language
  pr_oa_mt     second user turns of English oasst1 conversations, after the first turn and the assistant's reply
  pr_mt_harm   a HarmBench standard request as the second user turn, after an English oasst1 first exchange
  pr_sys_harm  a HarmBench standard request behind a system prompt from awesome-chatgpt-prompts
  pr_sys_dolly the held-out Dolly requests of the experiment behind a system prompt from awesome-chatgpt-prompts
Rows live in data/pr_*.jsonl and data/pr_sysprompts.jsonl, written once by scripts/prepare_prospective.py. A missing
file means the set is skipped. Scoring is refusal_format_score (unused for firing).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

from .tasks import POOLS, Pool, dolly_prompt, refusal_format_score

PROSPECTIVE: Dict[str, str] = {
    "pr_mj_en": "pr_mj_en.jsonl", "pr_mj_hi": "pr_mj_hi.jsonl", "pr_mj_mid": "pr_mj_mid.jsonl",
    "pr_mj_lo": "pr_mj_lo.jsonl", "pr_oa_en": "pr_oa_en.jsonl", "pr_oa_ml": "pr_oa_ml.jsonl",
    "pr_oa_mt": "pr_oa_mt.jsonl", "pr_mt_harm": "pr_mt_harm.jsonl", "pr_sys_harm": "pr_sys_harm.jsonl"}
SYSPROMPTS = "pr_sysprompts.jsonl"
LABEL = {"pr_mj_en": "MultiJail, English", "pr_mj_hi": "MultiJail, high-resource", "pr_mj_mid": "MultiJail, medium-resource",
         "pr_mj_lo": "MultiJail, low-resource", "pr_oa_en": "oasst1 first turns, English",
         "pr_oa_ml": "oasst1 first turns, other languages", "pr_oa_mt": "oasst1 second turns, English",
         "pr_mt_harm": "HarmBench as a second turn", "pr_sys_harm": "HarmBench behind a system prompt",
         "pr_sys_dolly": "Dolly behind a system prompt"}
HARMFUL = ("pr_mj_en", "pr_mj_hi", "pr_mj_mid", "pr_mj_lo", "pr_mt_harm", "pr_sys_harm")
BENIGN = ("pr_oa_en", "pr_oa_ml", "pr_oa_mt", "pr_sys_dolly")
_DATA = {"dir": "data", "sys": None}


def set_data_dir(path: str) -> None:
    """the folder of pr_sysprompts.jsonl (pr_sys_dolly picks its system prompt from it)"""
    if path != _DATA["dir"]:
        _DATA["dir"], _DATA["sys"] = path, None


def prospective_path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, PROSPECTIVE[name])


def prospective_rows(cfg: dict, name: str) -> List[dict]:
    path = prospective_path(cfg["data_dir"], name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prospective_pools(cfg: dict) -> List[str]:
    """the file-based sets present in the config's data folder"""
    return [n for n in PROSPECTIVE if os.path.exists(prospective_path(cfg["data_dir"], n))]


def sysprompts_present(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, SYSPROMPTS))


def _sysprompts() -> List[str]:
    if _DATA["sys"] is None:
        with open(os.path.join(_DATA["dir"], SYSPROMPTS)) as f:
            _DATA["sys"] = [json.loads(line)["system"] for line in f if line.strip()]
    return _DATA["sys"]


def pick(key: str, n: int) -> int:
    """a fixed choice among n items for a row (sha256 of its key), the same in every experiment"""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def _plain(ex: dict) -> str:
    return ex["prompt"].strip()


def _history(ex: dict):
    return list(ex["history"]) + [{"role": "user", "content": ex["prompt"].strip()}]


def _system(ex: dict):
    return [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"].strip()}]


def _sys_dolly(ex: dict):
    s = _sysprompts()
    key = str(ex.get("id", ex.get("instruction", "")))
    return [{"role": "system", "content": s[pick("pr_sys_dolly:" + key, len(s))]},
            {"role": "user", "content": dolly_prompt(ex)}]


for _name in ("pr_mj_en", "pr_mj_hi", "pr_mj_mid", "pr_mj_lo", "pr_oa_en", "pr_oa_ml"):
    POOLS[_name] = Pool(_name, _plain, refusal_format_score, source=_name)
for _name in ("pr_oa_mt", "pr_mt_harm"):
    POOLS[_name] = Pool(_name, _history, refusal_format_score, source=_name)
POOLS["pr_sys_harm"] = Pool("pr_sys_harm", _system, refusal_format_score, source="pr_sys_harm")
POOLS["pr_sys_dolly"] = Pool("pr_sys_dolly", _sys_dolly, refusal_format_score, source="dolly")
