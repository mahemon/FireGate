#!/usr/bin/env python
"""The two harmful sets that step p3.23 (Nov 5, PLAN_PAPER3.md) never fits or calibrates; run once on the login node
(needs the internet and the `datasets` / `huggingface_hub` packages), after scripts/prepare_prospective.py:

    python scripts/prepare_widen.py --data data
    python scripts/prepare_widen.py --data data --aya-jsonl DIR --dna-csv data_en.csv     # if downloads are blocked

Sources (public, not gated): Aya red-teaming (CohereForAI/aya_redteaming, Apache-2.0; files aya_<lang>.jsonl, one
per language, with a "prompt" field) and Do-Not-Answer (LibrAI/do-not-answer, split train, column "question"). Every
rule is fixed here, before any gate sees a row:
  both   rows whose normalised text (twcore/external.norm: lower case, Latin letters and digits) equals a request of
         AdvBench, HarmBench, StrongREJECT or MultiJail English are dropped (rows of another script, whose norm is
         empty, are never matched), and duplicates within a set (case-folded text, any script, spaces collapsed)
  wd_aya at most N_AYA rows per language, sampled with seed 1904 from the rows in file order; id aya_<lang>_<row>
  wd_dna every row; id dna<id>
Writes data/wd_aya.jsonl and data/wd_dna.jsonl ({"id", "prompt", "lang", "source", ...}) and prints the counts and the
first row of each. Record the counts in PLAN_PAPER3.md (step p3.23), then run
python scripts/prereg_record.py --step p3.23 before any gate sees the sets.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.external import norm  # noqa: E402

SEED = 1904
N_AYA = 300
AYA_FILE = re.compile(r"aya_([a-z]{3})\.jsonl$")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def same_key(text):
    """the key for duplicates within a set: case-folded, spaces collapsed, every script kept (Nov 5: norm() keeps
    only a-z and 0-9, so it made every Arabic, Hindi or Russian row a duplicate of the first)"""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def known_requests(data):
    """normalised texts of the sets the gate of p3.23 is fitted or calibrated on, and AdvBench"""
    out = set()
    for fn, keys in (("advbench.jsonl", ("prompt", "goal")), ("ext_harmbench.jsonl", ("prompt",)),
                     ("ext_strongreject.jsonl", ("prompt",)), ("pr_mj_en.jsonl", ("prompt",))):
        p = os.path.join(data, fn)
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing (scripts/prepare_data.py, prepare_external.py, prepare_prospective.py)")
        for r in read_jsonl(p):
            t = next((r[k] for k in keys if r.get(k)), "")
            if norm(t):
                out.add(norm(t))
    return out


def aya_files(args):
    """[(language code, rows)] from local files or the Hugging Face repository"""
    if args.aya_jsonl:
        paths = sorted(glob.glob(os.path.join(args.aya_jsonl, "aya_*.jsonl")))
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        names = sorted(f for f in list_repo_files(args.aya_repo, repo_type="dataset") if AYA_FILE.search(f))
        paths = [hf_hub_download(args.aya_repo, f, repo_type="dataset") for f in names]
    out = []
    for p in paths:
        m = AYA_FILE.search(os.path.basename(p))
        if m:
            out.append((m.group(1), read_jsonl(p)))
    if not out:
        raise SystemExit(f"Aya red-teaming: no aya_<lang>.jsonl files found ({args.aya_jsonl or args.aya_repo})")
    return out


def aya(files, known):
    rng = random.Random(SEED)
    rows, counts = [], {}
    for lang, rs in files:
        if rs and "prompt" not in rs[0]:
            raise SystemExit(f"Aya red-teaming aya_{lang}.jsonl: no 'prompt' field (fields: {sorted(rs[0])})")
        seen, keep = set(), []
        for i, r in enumerate(rs):
            t = (r.get("prompt") or "").strip()
            k, u = norm(t), same_key(t)
            if not t or (k and k in known) or u in seen:
                continue
            seen.add(u)
            keep.append((i, r, t))
        take = sorted(rng.sample(range(len(keep)), min(N_AYA, len(keep))))
        for j in take:
            i, r, t = keep[j]
            rows.append({"id": f"aya_{lang}_{i}", "prompt": t, "lang": lang, "source": "Aya red-teaming",
                         "harm_category": r.get("harm_category"), "global_or_local": r.get("global_or_local")})
        counts[lang] = (len(rs), len(keep), len(take))
    for lang, (n, k, t) in counts.items():
        print(f"Aya red-teaming {lang}: {n} rows, {k} after dropping duplicates and known requests, {t} kept")
    return rows


def dna(args, known):
    if args.dna_csv:
        with open(args.dna_csv, newline="", encoding="utf-8") as f:
            rs = list(csv.DictReader(f))
    else:
        from datasets import load_dataset
        rs = list(load_dataset("LibrAI/do-not-answer", split="train"))
    if rs and "question" not in rs[0]:
        raise SystemExit(f"Do-Not-Answer: no 'question' column (columns: {sorted(rs[0])})")
    rows, seen, dropped = [], set(), 0
    for i, r in enumerate(rs):
        t = (r.get("question") or "").strip()
        k, u = norm(t), same_key(t)
        if not t or (k and k in known) or u in seen:
            dropped += 1
            continue
        seen.add(u)
        rows.append({"id": f"dna{r.get('id', i)}", "prompt": t, "lang": "en", "source": "Do-Not-Answer",
                     "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm")})
    print(f"Do-Not-Answer: {len(rs)} rows, {dropped} dropped (empty, duplicate or a known request), {len(rows)} kept")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--aya-repo", default="CohereForAI/aya_redteaming")
    ap.add_argument("--aya-jsonl", default="", help="a folder holding the aya_<lang>.jsonl files")
    ap.add_argument("--dna-csv", default="", help="Do-Not-Answer's data_en.csv")
    args = ap.parse_args()
    known = known_requests(args.data)
    sets = {"wd_aya": aya(aya_files(args), known), "wd_dna": dna(args, known)}
    for name, rows in sets.items():
        if not rows:
            raise SystemExit(f"{name}: no rows; nothing written")
    for name, rows in sets.items():
        write_jsonl(os.path.join(args.data, f"{name}.jsonl"), rows)
        print(f"  {name}: {len(rows)} rows; first: {json.dumps(rows[0], ensure_ascii=False)[:220]}")
    print("record these counts in PLAN_PAPER3.md (p3.23), then: python scripts/prereg_record.py --step p3.23")


if __name__ == "__main__":
    main()
