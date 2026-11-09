#!/usr/bin/env python
"""External harmful-request sets for step p3.12 (PLAN_PAPER3.md); run once on the login node (needs the internet).

    python scripts/prepare_external.py --data data
    python scripts/prepare_external.py --data data --harmbench-csv path/to/harmbench_behaviors_text_all.csv \
        --strongreject-csv path/to/strongreject_dataset.csv          # if the downloads are blocked

Writes data/ext_harmbench.jsonl (standard behaviours), data/ext_harmbench_ctx.jsonl (contextual behaviours, with
their context passage) and data/ext_strongreject.jsonl (StrongREJECT, rows sourced from AdvBench removed). Every
request whose normalised text equals an AdvBench request (data/advbench.jsonl) is dropped, so no set overlaps the
pools the gates were fitted, calibrated or evaluated on. Sources are tried in order: a local CSV given on the command
line; otherwise the authors' GitHub CSV, then the Hugging Face copy (walledai/...). The script prints the counts and the first rows; check
them before step p3.12 runs, and record the counts in PLAN_PAPER3.md.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from twcore.external import EXTERNAL, norm  # noqa: E402

HB_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
SR_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_csv(local, url):
    """rows of a CSV from a local file or a URL, or None"""
    if local:
        with open(local, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as r:
            return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    except Exception as e:  # noqa: BLE001
        print(f"  {url}: {repr(e)[:160]}")
        return None


def load_hf(name, config):
    try:
        from datasets import load_dataset
        return list(load_dataset(name, config, split="train"))
    except Exception as e:  # noqa: BLE001
        print(f"  {name} ({config}): {repr(e)[:160]}")
        return None


def harmbench(local):
    """-> (standard rows, contextual rows), each {"prompt", "context", "source"}"""
    rows = read_csv(local, HB_URL)
    if rows is None and not local:
        std, ctx = load_hf("walledai/HarmBench", "standard"), load_hf("walledai/HarmBench", "contextual")
        if std is not None and ctx is not None:
            print("  HarmBench from the Hugging Face copy walledai/HarmBench")
            return ([{"prompt": r["prompt"], "context": "", "source": "harmbench/standard"} for r in std],
                    [{"prompt": r["prompt"], "context": r.get("context", ""), "source": "harmbench/contextual"}
                     for r in ctx])
    if rows is None:
        raise SystemExit("HarmBench: no source worked; download harmbench_behaviors_text_all.csv and pass --harmbench-csv")
    need = {"Behavior", "FunctionalCategory", "ContextString"}
    if not need <= set(rows[0]):
        raise SystemExit(f"HarmBench CSV: columns {sorted(rows[0])}, expected at least {sorted(need)}")
    std = [{"prompt": r["Behavior"], "context": "", "source": "harmbench/standard"}
           for r in rows if r["FunctionalCategory"].strip() == "standard"]
    ctx = [{"prompt": r["Behavior"], "context": r["ContextString"], "source": "harmbench/contextual"}
           for r in rows if r["FunctionalCategory"].strip() == "contextual" and r["ContextString"].strip()]
    return std, ctx


def strongreject(local):
    rows = read_csv(local, SR_URL)
    if rows is None and not local:
        hf = load_hf("walledai/StrongREJECT", None)
        if hf is not None:
            print("  StrongREJECT from the Hugging Face copy walledai/StrongREJECT")
            return [{"prompt": r["prompt"], "context": "", "source": f"strongreject/{r.get('source', '')}"} for r in hf]
    if rows is None:
        raise SystemExit("StrongREJECT: no source worked; download strongreject_dataset.csv and pass --strongreject-csv")
    if "forbidden_prompt" not in rows[0]:
        raise SystemExit(f"StrongREJECT CSV: columns {sorted(rows[0])}, expected forbidden_prompt")
    return [{"prompt": r["forbidden_prompt"], "context": "", "source": f"strongreject/{r.get('source', '')}"}
            for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--harmbench-csv", default="")
    ap.add_argument("--strongreject-csv", default="")
    args = ap.parse_args()
    adv_path = os.path.join(args.data, "advbench.jsonl")
    if not os.path.exists(adv_path):
        raise SystemExit(f"{adv_path} is missing (prepare_data.py first)")
    adv = {norm(json.loads(line)["prompt"]) for line in open(adv_path) if line.strip()}
    std, ctx = harmbench(args.harmbench_csv)
    sr = strongreject(args.strongreject_csv)
    sets = {"ext_hb": std, "ext_hb_ctx": ctx,
            "ext_sr": [r for r in sr if "advbench" not in r["source"].lower()]}
    for name, rows in sets.items():
        seen, out, dropped = set(), [], 0
        for r in rows:
            k = norm(r["prompt"]) + "|" + norm(r["context"])
            if norm(r["prompt"]) in adv or k in seen or not r["prompt"].strip():
                dropped += 1
                continue
            seen.add(k)
            out.append({"id": f"{name}_{len(out)}", **r})
        write_jsonl(os.path.join(args.data, EXTERNAL[name]), out)
        print(f"{name}: {len(out)} rows written to {args.data}/{EXTERNAL[name]} ({dropped} dropped: AdvBench, "
              f"duplicate or empty)")
        for r in out[:2]:
            print("   ", (r["context"][:80] + " ... --- ") if r["context"] else "", r["prompt"][:120])


if __name__ == "__main__":
    main()
