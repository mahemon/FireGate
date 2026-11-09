"""Step p3.23 (Nov 5, PLAN_PAPER3.md): a wide harmful component for a task drawn from chat (Concise); no torch.

The chosen Concise gate samples harmful requests from AdvBench only, and fires on up to 32% of MultiJail (p3.19): a
harmful request written unlike AdvBench reads as chat. This step samples harmful requests in more forms and languages
as one more reference component ("harm_wide") and keeps a third of every source, and two new sets, out of fitting
and calibration.

Sources of the wide component (all already stored as features in every experiment of p3.12 / p3.19):
  ext_hb       HarmBench standard requests          (with their forms pr_mt_harm: as a second turn after an oasst1
                                                     exchange, and pr_sys_harm: behind a system prompt)
  ext_sr       StrongREJECT forbidden prompts
  pr_mj_*      MultiJail, English and nine other languages (a request and its translations are one group)
Each request group goes to one part, by sha256("p3.23:<group>") mod 3: 0 fit, 1 cal, 2 test. Fitting uses every row
of the fit part (every form and language). Calibration uses one row per group of the cal part (HarmBench and
StrongREJECT: the plain request; MultiJail: one language chosen by sha256), so its prompts are i.i.d. draws of
requests and Theorem 3.6 holds for the component. The test part is never fitted or calibrated.

Sets never fitted or calibrated, written once by scripts/prepare_widen.py (rows {"id", "prompt", "lang", "source"}):
  wd_aya       Aya red-teaming (Aakanksha et al., 2024): harmful prompts written by native speakers in eight
               languages, at most 300 per language
  wd_dna       Do-Not-Answer (Wang et al., 2024): questions a responsible model should not answer, English
Both drop every row whose normalised text equals a request of AdvBench, HarmBench, StrongREJECT or MultiJail (English).
Scoring is refusal_format_score. Refusal is an English string check, so the newly answered prompts of step p3.23 are
counted on English sets only; firing is measured in every language.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from .tasks import POOLS, Pool, refusal_format_score

WIDEN: Dict[str, str] = {"wd_aya": "wd_aya.jsonl", "wd_dna": "wd_dna.jsonl"}
LABEL = {"wd_aya": "Aya red-teaming", "wd_dna": "Do-Not-Answer"}
SOURCES = ("ext_hb", "pr_mt_harm", "pr_sys_harm", "ext_sr", "pr_mj_en", "pr_mj_hi", "pr_mj_mid", "pr_mj_lo")
CAL_POOLS = ("ext_hb", "ext_sr", "pr_mj_en", "pr_mj_hi", "pr_mj_mid", "pr_mj_lo")   # one row per group from these
PARTS = ("fit", "cal", "test")
SALT = "p3.23"


def widen_path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, WIDEN[name])


def widen_rows(cfg: dict, name: str) -> List[dict]:
    path = widen_path(cfg["data_dir"], name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def widen_pools(cfg: dict) -> List[str]:
    return [n for n in WIDEN if os.path.exists(widen_path(cfg["data_dir"], n))]


def group_of(pool: str, rid) -> Optional[str]:
    """the request a row belongs to: a HarmBench request and its two forms, a MultiJail request and its translations"""
    rid = str(rid)
    if pool == "ext_hb":
        return "hb:" + rid
    if pool == "pr_mt_harm" and rid.startswith("mt_"):
        return "hb:" + rid[3:]
    if pool == "pr_sys_harm" and rid.startswith("sys_"):
        return "hb:" + rid[4:]
    if pool == "ext_sr":
        return "sr:" + rid
    if pool.startswith("pr_mj_") and rid.startswith("mj"):
        return "mj:" + rid[2:].split("_")[0]
    return None


def part_of(group: str) -> str:
    return PARTS[int(hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest(), 16) % 3]


def pick(key: str, n: int) -> int:
    return int(hashlib.sha256(f"{SALT}-pick:{key}".encode()).hexdigest(), 16) % n


def in_test(pool: str, rid) -> bool:
    """True for a row of a source in the test part, and for every row of a set that is not a source"""
    g = group_of(pool, rid) if pool in SOURCES else None
    return g is None or part_of(g) == "test"


def generation_sets(cfg: dict, prospective_rows) -> Dict[str, List[dict]]:
    """English harmful rows for step p3.23's responses (scripts/eval_extra.py --what widen): the test part of
    MultiJail English and of the two HarmBench forms, every Do-Not-Answer row and the English Aya rows. The plain
    HarmBench and StrongREJECT rows have their responses from --what external."""
    out = {}
    for name in ("pr_mj_en", "pr_mt_harm", "pr_sys_harm"):
        rows = [r for r in prospective_rows(cfg, name) if in_test(name, r.get("id"))]
        if rows:
            out[name] = rows
    dna = widen_rows(cfg, "wd_dna")
    if dna:
        out["wd_dna"] = dna
    aya = [r for r in widen_rows(cfg, "wd_aya") if str(r.get("lang", "")).lower() in ("en", "eng", "english")]
    if aya:
        out["wd_aya_en"] = aya
    return out


def _plain(ex: dict) -> str:
    return ex["prompt"].strip()


for _name in ("wd_aya", "wd_dna", "wd_aya_en"):
    POOLS[_name] = Pool(_name, _plain, refusal_format_score, source=_name)
