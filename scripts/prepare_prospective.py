#!/usr/bin/env python
"""Prompt sets written by others for the prospective test (step p3.19, PLAN_PAPER3.md); run once on the login node
(needs the internet and the `datasets` package), after scripts/prepare_external.py:

    python scripts/prepare_prospective.py --data data
    python scripts/prepare_prospective.py --data data --multijail-csv MultiJail.csv --sysprompts-csv prompts.csv \
        --oasst-jsonl 2023-04-12_oasst_ready.messages.jsonl      # if the Hugging Face downloads are blocked

Sources (all public): MultiJail (DAMO-NLP-SG/MultiJail, MIT), OpenAssistant oasst1 (OpenAssistant/oasst1, Apache-2.0),
awesome-chatgpt-prompts (fka/awesome-chatgpt-prompts, CC0), and data/ext_harmbench.jsonl (HarmBench standard, written
by prepare_external.py). Every rule below is fixed here, before any gate sees a row:
  MultiJail   rows whose English text equals an AdvBench request (normalised) are dropped in every language;
              pr_mj_en: English; pr_mj_hi: zh, it, vi; pr_mj_mid: ar, ko, th; pr_mj_lo: bn, sw, jv
  oasst1      train and validation messages, not deleted, review_result not False. pr_oa_en / pr_oa_ml: 1,000
              first user turns each (English / every other language), sampled with seed 1904 from the messages sorted
              by id. pr_oa_mt: 1,000 English second user turns, each after its first user turn and the assistant reply
              it answers (the reply with the lowest rank). History messages are cut to 2,000 characters.
  sysprompts  awesome-chatgpt-prompts' "prompt" column, cut before the first "My first", kept if at least 40
              characters remain and no jailbreak wording appears (jailbreak, DAN, no restrictions, unfiltered,
              uncensored, ignore ... instructions/rules/guidelines), duplicates dropped: data/pr_sysprompts.jsonl
  pr_mt_harm  each HarmBench standard request after an English oasst1 first exchange (seed 1904, one per request)
  pr_sys_harm each HarmBench standard request behind a system prompt chosen by sha256("pr_sys_harm:<id>")
  pr_sys_dolly (no file) each held-out Dolly request of an experiment behind a system prompt chosen by
              sha256("pr_sys_dolly:<id>"), built in scripts/gate_features.py
The script prints the counts and the first row of each set; record the counts in PLAN_PAPER3.md, then run
python scripts/prereg_record.py --step p3.19 before any gate sees the sets.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from twcore.external import norm  # noqa: E402
from twcore.prospective import SYSPROMPTS, pick  # noqa: E402

SEED = 1904
N_OASST = 1000
HIST_CHARS = 2000
GROUPS = {"pr_mj_hi": ("zh", "it", "vi"), "pr_mj_mid": ("ar", "ko", "th"), "pr_mj_lo": ("bn", "sw", "jv")}
JAILBREAK = re.compile(r"jailbreak|\bDAN\b|no restrictions|unfiltered|uncensored|"
                       r"ignore (all |any |the |your )?(previous |prior )?(instructions|rules|guidelines)", re.I)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_hf(name, split=None):
    from datasets import load_dataset
    if split:
        return list(load_dataset(name, split=split))
    d = load_dataset(name)
    return [r for s in d for r in d[s]]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def multijail(rows, adv):
    langs = [lang for g in GROUPS.values() for lang in g]
    cols = set(rows[0]) if rows else set()
    missing = [c for c in ["en"] + langs if c not in cols]
    if missing:
        raise SystemExit(f"MultiJail: columns {missing} are missing (columns: {sorted(cols)})")
    out = {k: [] for k in ["pr_mj_en"] + list(GROUPS)}
    dropped = 0
    for i, r in enumerate(rows):
        rid = str(r.get("id", i))
        if norm(r["en"]) in adv:
            dropped += 1
            continue
        out["pr_mj_en"].append({"id": f"mj{rid}", "prompt": r["en"].strip(), "lang": "en", "source": "MultiJail"})
        for name, g in GROUPS.items():
            for lang in g:
                if (r.get(lang) or "").strip():
                    out[name].append({"id": f"mj{rid}_{lang}", "prompt": r[lang].strip(), "lang": lang,
                                      "source": "MultiJail"})
    print(f"MultiJail: {len(rows)} rows, {dropped} dropped as AdvBench duplicates")
    return out


def oasst(msgs):
    ok = [m for m in msgs if not m.get("deleted") and m.get("review_result") is not False and (m.get("text") or "").strip()]
    by_id = {m["message_id"]: m for m in ok}
    kids = {}
    for m in ok:
        if m.get("parent_id"):
            kids.setdefault(m["parent_id"], []).append(m)
    roots = sorted((m for m in ok if not m.get("parent_id") and m.get("role") == "prompter"), key=lambda m: m["message_id"])

    def best_reply(m):
        rs = [k for k in kids.get(m["message_id"], []) if k.get("role") == "assistant"]
        rs.sort(key=lambda k: (k.get("rank") if k.get("rank") is not None else 1e9, k["message_id"]))
        return rs[0] if rs else None

    rng = random.Random(SEED)
    en = [m for m in roots if m.get("lang") == "en"]
    ml = [m for m in roots if m.get("lang") and m.get("lang") != "en"]
    row = lambda m: {"id": m["message_id"], "prompt": m["text"].strip(), "lang": m["lang"], "source": "oasst1"}  # noqa: E731
    out = {"pr_oa_en": [row(m) for m in rng.sample(en, min(N_OASST, len(en)))],
           "pr_oa_ml": [row(m) for m in rng.sample(ml, min(N_OASST, len(ml)))]}
    second = []
    for m in en:
        a = best_reply(m)
        if a is None:
            continue
        turns = sorted((k for k in kids.get(a["message_id"], []) if k.get("role") == "prompter"), key=lambda k: k["message_id"])
        if turns:
            second.append((m, a, turns[0]))
    hist = lambda m, a: [{"role": "user", "content": m["text"].strip()[:HIST_CHARS]},  # noqa: E731
                         {"role": "assistant", "content": a["text"].strip()[:HIST_CHARS]}]
    out["pr_oa_mt"] = [{"id": t["message_id"], "history": hist(m, a), "prompt": t["text"].strip(), "lang": "en",
                        "source": "oasst1"} for m, a, t in rng.sample(second, min(N_OASST, len(second)))]
    exchanges = [(m, a) for m in en for a in [best_reply(m)] if a is not None]
    from collections import Counter
    print(f"oasst1: {len(ok)} messages kept, {len(en)} English and {len(ml)} other first turns "
          f"({dict(Counter(m['lang'] for m in ml).most_common(8))}), {len(second)} English second turns")
    return out, [hist(m, a) for m, a in exchanges]


def sysprompts(rows):
    out, seen = [], set()
    for r in rows:
        p = (r.get("prompt") or "").strip()
        i = p.find("My first")
        if i >= 0:
            p = p[:i].strip()
        if len(p) < 40 or JAILBREAK.search(p) or p in seen:
            continue
        seen.add(p)
        out.append({"id": f"sys{len(out)}", "system": p, "act": (r.get("act") or "").strip()})
    print(f"awesome-chatgpt-prompts: {len(rows)} rows, {len(out)} system prompts kept")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--multijail-csv", default="")
    ap.add_argument("--sysprompts-csv", default="")
    ap.add_argument("--oasst-jsonl", default="", help="oasst1 messages (.jsonl or .jsonl.gz)")
    args = ap.parse_args()
    adv_path = os.path.join(args.data, "advbench.jsonl")
    hb_path = os.path.join(args.data, "ext_harmbench.jsonl")
    for p in (adv_path, hb_path):
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing (scripts/prepare_data.py, scripts/prepare_external.py)")
    adv = {norm(r.get("prompt") or r.get("goal") or "") for r in read_jsonl(adv_path)}
    hb = read_jsonl(hb_path)

    mj_rows = read_csv(args.multijail_csv) if args.multijail_csv else load_hf("DAMO-NLP-SG/MultiJail", "train")
    sets = multijail(mj_rows, adv)
    oa_rows = read_jsonl(args.oasst_jsonl) if args.oasst_jsonl else load_hf("OpenAssistant/oasst1")
    oa, exchanges = oasst(oa_rows)
    sets.update(oa)
    sp_rows = read_csv(args.sysprompts_csv) if args.sysprompts_csv else load_hf("fka/awesome-chatgpt-prompts", "train")
    sp = sysprompts(sp_rows)
    if not sp or not exchanges:
        raise SystemExit("no system prompts or no oasst1 exchanges: nothing written")
    rng = random.Random(SEED)
    sets["pr_mt_harm"] = [{"id": f"mt_{r['id']}", "history": rng.choice(exchanges), "prompt": r["prompt"].strip(),
                           "source": "HarmBench+oasst1"} for r in hb]
    sets["pr_sys_harm"] = [{"id": f"sys_{r['id']}", "system": sp[pick(f"pr_sys_harm:{r['id']}", len(sp))]["system"],
                            "prompt": r["prompt"].strip(), "source": "HarmBench+awesome-chatgpt-prompts"} for r in hb]
    write_jsonl(os.path.join(args.data, SYSPROMPTS), sp)
    for name, rows in sets.items():
        write_jsonl(os.path.join(args.data, f"{name}.jsonl"), rows)
        first = json.dumps(rows[0], ensure_ascii=False)[:220] if rows else "(empty)"
        print(f"  {name}: {len(rows)} rows; first: {first}")
    print(f"  {SYSPROMPTS}: {len(sp)} rows; first: {sp[0]['system'][:160]}")
    print("record these counts in PLAN_PAPER3.md (p3.19), then: python scripts/prereg_record.py --step p3.19")


if __name__ == "__main__":
    main()
