#!/usr/bin/env python
"""Download the model and datasets and write the JSONL files used by the runs. Run once on a node with
internet access (e.g. the Nova login node); compute jobs then run offline.

    python scripts/prepare_data.py --data data --models models
    python scripts/prepare_data.py --data data --skip-model          # datasets only
    python scripts/prepare_data.py --data data --only-synthetic      # Countdown / Knights and Knaves only

Files (all shuffled with fixed seeds where an order matters):
  gsm8k_{train,test}.jsonl      openai/gsm8k (main)
  countdown_{train,test}.jsonl  generated (6000 / 1000)
  kk_{train,test}.jsonl         generated (6000 / 1000)
  dolly.jsonl                   databricks/databricks-dolly-15k
  advbench.jsonl                AdvBench harmful behaviours (520)
  xstest.jsonl                  XSTest prompts with safe/unsafe labels (450)
  mmlu_eval.jsonl               2000 random cais/mmlu test questions
  general.jsonl                 task-like reference prompts: ARC, CommonsenseQA, OpenBookQA, BoolQ, Alpaca, MBPP
Nov 1 (PLAN_NOV.md), each only on request:
  wildchat_eval.jsonl           --only-wildchat: 1000 first user turns of English WildChat conversations
                                (allenai/WildChat-1M, read as a stream, about 0.2 GB of it), evaluation only
  code_{train,test}.jsonl       --only-code: KodCode-Light-RL-10K problems whose reference solution passes their tests
                                in the code task's runner (twcore/codetask.py); stdlib-only, easy and medium
Paper v3 (PLAN_PAPER3.md), only on request:
  squad2_{train,test}.jsonl     --only-squad2: rajpurkar/squad_v2 (CC BY-SA 4.0), contexts of at most 1500 characters,
                                half answerable and half not; 3600 training rows from its train split, 500 test rows
                                from its validation split (other Wikipedia articles)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.gen_countdown import make_instances as make_countdown  # noqa: E402
from twcore.gen_kk import make_instances as make_kk  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # an expired cached token breaks public downloads


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(rows)} rows)")


def load_hf(name, config=None, split="train"):
    from datasets import load_dataset
    try:
        return load_dataset(name, config, split=split, token=False)
    except TypeError:
        return load_dataset(name, config, split=split)


def fetch_url(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def gsm8k(data):
    for split in ("train", "test"):
        ds = load_hf("openai/gsm8k", "main", split)
        rows = [{"id": f"gsm8k_{split}_{i}", "question": ex["question"], "answer": ex["answer"]} for i, ex in enumerate(ds)]
        if split == "train":
            random.Random(11).shuffle(rows)
        write_jsonl(os.path.join(data, f"gsm8k_{split}.jsonl"), rows)


def synthetic(data):
    cd = make_countdown(7000, seed=1234)
    write_jsonl(os.path.join(data, "countdown_train.jsonl"), cd[:6000])
    write_jsonl(os.path.join(data, "countdown_test.jsonl"), cd[6000:])
    kk = make_kk(7000, seed=4321)
    write_jsonl(os.path.join(data, "kk_train.jsonl"), kk[:6000])
    write_jsonl(os.path.join(data, "kk_test.jsonl"), kk[6000:])


def dolly(data):
    ds = load_hf("databricks/databricks-dolly-15k", None, "train")
    rows = [{"id": f"dolly_{i}", "instruction": ex["instruction"], "context": ex.get("context", ""),
             "category": ex.get("category", "")} for i, ex in enumerate(ds)]
    random.Random(7).shuffle(rows)
    write_jsonl(os.path.join(data, "dolly.jsonl"), rows)


def advbench(data):
    prompts = None
    try:
        ds = load_hf("walledai/AdvBench", None, "train")
        prompts = [ex["prompt"] for ex in ds]
    except Exception as e:
        print("  walledai/AdvBench failed:", repr(e)[:200], "- trying GitHub")
        text = fetch_url("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
        prompts = [r["goal"] for r in csv.DictReader(io.StringIO(text))]
    rows = [{"id": f"adv_{i}", "prompt": p} for i, p in enumerate(prompts)]
    random.Random(3).shuffle(rows)
    write_jsonl(os.path.join(data, "advbench.jsonl"), rows)


def xstest(data):
    rows = None
    try:
        ds = load_hf("walledai/XSTest", None, "test")
        rows = [{"id": f"xs_{i}", "prompt": ex["prompt"], "label": ex["label"], "type": ex.get("type", "")}
                for i, ex in enumerate(ds)]
    except Exception as e:
        print("  walledai/XSTest failed:", repr(e)[:200], "- trying GitHub")
        for url in ("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
                    "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_v2_prompts.csv"):
            try:
                text = fetch_url(url)
                rd = list(csv.DictReader(io.StringIO(text)))
                rows = [{"id": f"xs_{i}", "prompt": r["prompt"], "label": r["label"], "type": r.get("type", "")}
                        for i, r in enumerate(rd)]
                break
            except Exception as e2:
                print("  ", url, "failed:", repr(e2)[:120])
    if not rows:
        raise SystemExit("could not obtain XSTest")
    for r in rows:
        r["label"] = "unsafe" if "unsafe" in str(r["label"]).lower() else "safe"
    write_jsonl(os.path.join(data, "xstest.jsonl"), rows)


def mmlu(data):
    ds = load_hf("cais/mmlu", "all", "test")
    idx = list(range(len(ds)))
    random.Random(5).shuffle(idx)
    rows = []
    for i in idx[:2000]:
        ex = ds[i]
        rows.append({"id": f"mmlu_{i}", "question": ex["question"], "choices": list(ex["choices"]),
                     "answer": int(ex["answer"]), "subject": ex["subject"]})
    write_jsonl(os.path.join(data, "mmlu_eval.jsonl"), rows)


def nq(data):
    """Natural Questions (open, closed book): 6000 training questions, 1000 test questions from validation."""
    tr = load_hf("google-research-datasets/nq_open", None, "train")
    va = load_hf("google-research-datasets/nq_open", None, "validation")
    rows = [{"id": f"nq_train_{i}", "question": ex["question"], "answers": list(ex["answer"])} for i, ex in enumerate(tr)]
    random.Random(17).shuffle(rows)
    write_jsonl(os.path.join(data, "nq_train.jsonl"), rows[:6000])
    rows = [{"id": f"nq_val_{i}", "question": ex["question"], "answers": list(ex["answer"])} for i, ex in enumerate(va)]
    random.Random(19).shuffle(rows)
    write_jsonl(os.path.join(data, "nq_test.jsonl"), rows[:1000])


def general(data, per_source=400):
    """Task-like prompts from several sources, in formats deliberately unlike MMLU's template, so that MMLU and
    XSTest stay held-out tests of how certificates transfer to distributions never used in calibration."""
    rng = random.Random(13)
    rows = []

    def take(name, items):
        items = [p for p in items if p and len(p) < 3000]
        rng.shuffle(items)
        items = items[:per_source]
        rows.extend({"id": f"gen_{name}_{i}", "prompt": p, "source": name} for i, p in enumerate(items))
        print(f"  general: {len(items)} prompts from {name}")

    def opts(ch):
        return list(zip(ch["label"], ch["text"]))

    sources = [
        ("arc", lambda: [f"{ex['question']}\n" + "\n".join(f"({l}) {t}" for l, t in opts(ex["choices"]))
                         + "\nWhich option is correct? Explain briefly."
                         for cfg in ("ARC-Challenge", "ARC-Easy") for ex in load_hf("allenai/ai2_arc", cfg, "train")]),
        ("commonsenseqa", lambda: [f"{ex['question']} Choose one: " + ", ".join(t for _, t in opts(ex["choices"])) + "."
                                   for ex in load_hf("tau/commonsense_qa", None, "train")]),
        ("openbookqa", lambda: [f"Question: {ex['question_stem']}\nOptions: " + "; ".join(f"{l}) {t}" for l, t in opts(ex["choices"]))
                                for ex in load_hf("allenai/openbookqa", "main", "train")]),
        ("boolq", lambda: [f"{ex['passage'][:800]}\n\nQuestion: {ex['question']}? Answer yes or no."
                           for ex in load_hf("google/boolq", None, "train")]),
        ("alpaca", lambda: [ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
                            for ex in load_hf("tatsu-lab/alpaca", None, "train")]),
        ("mbpp", lambda: [f"{ex['text']} Write the Python function." for ex in load_hf("google-research-datasets/mbpp", "full", "train")]),
    ]
    for name, fn in sources:
        try:
            take(name, fn())
        except Exception as e:  # one unavailable source is not fatal
            print(f"  general: {name} failed ({repr(e)[:160]}); skipped")
    if len(rows) < 1600:
        raise SystemExit(f"general reference set has only {len(rows)} prompts (need 1600)")
    rng.shuffle(rows)
    write_jsonl(os.path.join(data, "general.jsonl"), rows)


def wildchat(data, n=1000, seed=23):
    """First user turns of English, non-toxic, non-redacted WildChat conversations: deduplicated, 20-2000 characters,
    in the order of a seeded stream shuffle (fixed seed and buffer), so the file is the same on every rerun."""
    from datasets import load_dataset
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True, token=False)
    except TypeError:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    rows, seen, read = [], set(), 0
    for ex in ds:
        read += 1
        if ex.get("toxic") or ex.get("redacted") or ex.get("language") != "English":
            continue
        first = next((m for m in (ex.get("conversation") or []) if m.get("role") == "user"), None)
        text = ((first or {}).get("content") or "").strip()
        if not 20 <= len(text) <= 2000:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"wildchat_{ex.get('conversation_hash') or read}", "prompt": text})
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} WildChat requests found (need {n})")
    write_jsonl(os.path.join(data, "wildchat_eval.jsonl"), rows)
    with open(os.path.join(data, "wildchat_eval.meta.json"), "w") as f:
        json.dump({"source": "allenai/WildChat-1M", "license": "ODC-BY", "seed": seed, "buffer": 20000,
                   "conversations_read": read, "rows": len(rows)}, f, indent=1)


def squad2(data, n_train=3600, n_test=500, seed=31, max_ctx=1500):
    """SQuAD 2.0 (paper v3): question + passage in the layout of a Dolly request with context (twcore/squad2.py). Half
    of each split is answerable; contexts longer than max_ctx characters are left out, so the Dolly layout (which
    keeps the first 1500 characters of a context) shows the whole passage."""
    def rows(ds, pre):
        out = []
        for i, ex in enumerate(ds):
            ctx = (ex["context"] or "").strip()
            q = (ex["question"] or "").strip()
            if not q or not ctx or len(ctx) > max_ctx:
                continue
            ans = sorted({a.strip() for a in ex["answers"]["text"] if a and a.strip()})
            out.append({"id": f"{pre}_{i}", "instruction": q, "context": ctx, "answers": ans,
                        "answerable": bool(ans), "title": ex.get("title", "")})
        return out

    def balanced(rs, n, rng, what):
        a = [r for r in rs if r["answerable"]]
        u = [r for r in rs if not r["answerable"]]
        rng.shuffle(a)
        rng.shuffle(u)
        k = n // 2
        if len(a) < n - k or len(u) < k:
            raise SystemExit(f"squad2 {what}: {len(a)} answerable and {len(u)} unanswerable rows, need {n - k} and {k}")
        out = a[:n - k] + u[:k]
        rng.shuffle(out)
        return out

    rng = random.Random(seed)
    tr = rows(load_hf("rajpurkar/squad_v2", None, "train"), "sq2_train")
    va = rows(load_hf("rajpurkar/squad_v2", None, "validation"), "sq2_val")
    write_jsonl(os.path.join(data, "squad2_train.jsonl"), balanced(tr, n_train, rng, "train"))
    write_jsonl(os.path.join(data, "squad2_test.jsonl"), balanced(va, n_test, rng, "validation"))


def code(data, n_test=500, seed=29, workers=8):
    """KodCode-Light-RL-10K (KodCode/KodCode-Light-RL-10K, CC BY-NC 4.0): problems whose solution and tests import only
    the standard library, rated easy or medium, and whose reference solution passes its tests in the code task's own
    runner (so every kept test file works without pytest). Shuffled with a fixed seed; the first n_test rows are the
    test file, the rest the training file."""
    import ast
    from concurrent.futures import ThreadPoolExecutor
    from twcore.codetask import run_tests
    std = set(getattr(sys, "stdlib_module_names", ())) | {"solution", "pytest"}

    def stdlib_only(src):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if not node.level else ["solution"]
            else:
                continue
            if any(nm.split(".")[0] not in std for nm in names):
                return False
        return True

    ds = load_hf("KodCode/KodCode-Light-RL-10K", None, "train")
    cand, why = [], {}
    for ex in ds:
        info = ex.get("test_info") or []
        info = info[0] if isinstance(info, list) and info else (info if isinstance(info, dict) else {})
        q, sol, test = (ex.get("question") or "").strip(), ex.get("solution") or "", ex.get("test") or ""
        reason = None
        if not q or not sol or "from solution import" not in test:
            reason = "fields"
        elif str(ex.get("gpt_difficulty", "")).lower() not in ("easy", "medium"):
            reason = "hard"
        elif len(q) > 3000:
            reason = "long"
        elif not (stdlib_only(sol) and stdlib_only(test)):
            reason = "non-stdlib"
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        cand.append({"id": f"code_{ex.get('question_id')}", "question": q, "test": test, "solution": sol,
                     "entry": info.get("function_name") or "", "signature": (info.get("function_declaration") or "").strip(),
                     "difficulty": ex.get("gpt_difficulty"), "subset": ex.get("subset")})
    print(f"  code: {len(cand)} candidates; left out: {why}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda r: run_tests(r["solution"], r["test"]), cand))
    rows = [r for r, k in zip(cand, ok) if k]
    print(f"  code: {len(rows)} of {len(cand)} reference solutions pass in the runner")
    if len(rows) < n_test + 3000:
        raise SystemExit(f"only {len(rows)} usable code problems (need {n_test + 3000})")
    random.Random(seed).shuffle(rows)
    write_jsonl(os.path.join(data, "code_test.jsonl"), rows[:n_test])
    write_jsonl(os.path.join(data, "code_train.jsonl"), rows[n_test:])
    diff = {}
    for r in rows:
        diff[r["difficulty"]] = diff.get(r["difficulty"], 0) + 1
    with open(os.path.join(data, "code_meta.json"), "w") as f:
        json.dump({"source": "KodCode/KodCode-Light-RL-10K", "license": "CC BY-NC 4.0", "seed": seed,
                   "left_out": why, "candidates": len(cand), "reference_passes": len(rows), "test": n_test,
                   "train": len(rows) - n_test, "difficulty": diff}, f, indent=1)


# Llama-2 chat's own template (as in Meta's tokenizer_config.json); copies of the weights may come without it
LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = "
    "messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + "
    "'\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}")


def ensure_chat_template(local, repo):
    """Llama-2 chat copies without a chat template (e.g. NousResearch/Llama-2-13b-chat-hf) get Meta's template"""
    if "llama-2" not in repo.lower() or "chat" not in repo.lower():
        return
    p = os.path.join(local, "tokenizer_config.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        cfg = json.load(f)
    if cfg.get("chat_template"):
        return
    cfg["chat_template"] = LLAMA2_CHAT_TEMPLATE
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  added Llama-2 chat's template to", p)


def model(models_dir, repo):
    import glob
    from huggingface_hub import snapshot_download
    local = os.path.join(models_dir, repo.split("/")[-1])
    token = os.environ.get("HF_GATED_TOKEN") or False
    try:
        # public models are downloaded without a token (an expired HF_TOKEN in the environment would make them fail);
        # gated models (e.g. meta-llama/Llama-3.1-8B-Instruct) need HF_GATED_TOKEN=<a valid token>. Duplicate
        # weights in another format (Mistral's consolidated.safetensors, Llama's original/) are left out.
        snapshot_download(repo_id=repo, local_dir=local, token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"],
                          ignore_patterns=["consolidated*", "original/*", "pytorch_model.bin.index.json"])
        # weights only as .bin (deepseek-llm-7b-chat), or a conversion that was interrupted: (re)convert
        complete = (os.path.exists(os.path.join(local, "model.safetensors"))
                    or os.path.exists(os.path.join(local, "model.safetensors.index.json")))
        if not complete or glob.glob(os.path.join(local, "pytorch_model*.bin")):
            snapshot_download(repo_id=repo, local_dir=local, token=token,
                              allow_patterns=["pytorch_model*.bin", "pytorch_model.bin.index.json"])
            bin_to_safetensors(local)
    except TypeError:
        snapshot_download(repo_id=repo, local_dir=local)
    ensure_chat_template(local, repo)
    print("  model in", local)


def bin_to_safetensors(local):
    """Rewrite PyTorch .bin weight shards as .safetensors (recent transformers versions refuse to torch.load .bin
    files unless torch >= 2.6); the .bin files and their index are removed afterwards"""
    import glob
    import torch
    from safetensors.torch import save_file
    bins = sorted(glob.glob(os.path.join(local, "pytorch_model*.bin")))
    names = {}
    for b in bins:
        sd = {k: v.contiguous() for k, v in torch.load(b, map_location="cpu", weights_only=True).items()}
        out = os.path.basename(b).replace("pytorch_model", "model").replace(".bin", ".safetensors")
        try:
            save_file(sd, os.path.join(local, out), metadata={"format": "pt"})
        except RuntimeError:   # tensors that share memory (tied weights) are written as separate copies
            save_file({k: v.clone() for k, v in sd.items()}, os.path.join(local, out), metadata={"format": "pt"})
        names[os.path.basename(b)] = out
        del sd
        print(f"  {os.path.basename(b)} -> {out}")
    idx = os.path.join(local, "pytorch_model.bin.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            j = json.load(f)
        j["weight_map"] = {k: names.get(v, v) for k, v in j["weight_map"].items()}
        with open(os.path.join(local, "model.safetensors.index.json"), "w") as f:
            json.dump(j, f, indent=2)
        os.remove(idx)
    for b in bins:
        os.remove(b)


def trivia(data):
    """Step p3.26 (Nov 8, PLAN_PAPER3.md): TriviaQA (unfiltered, no context), a task no earlier step used. Closed-book
    trivia questions given as plain questions, scored like NQ (twcore/tasks.py): 6000 training questions, 1000 test
    questions from validation. Gold answers: the answer's value and every alias, duplicates removed."""
    def rows(ds, pre):
        out = []
        for i, ex in enumerate(ds):
            a = ex["answer"]
            golds = list(dict.fromkeys([a["value"]] + list(a.get("aliases") or [])))
            golds = [g for g in golds if g and g.strip()]
            if golds:
                out.append({"id": f"{pre}_{ex.get('question_id') or i}", "question": ex["question"], "answers": golds})
        return out
    tr = load_hf("mandarjoshi/trivia_qa", "unfiltered.nocontext", "train")
    va = load_hf("mandarjoshi/trivia_qa", "unfiltered.nocontext", "validation")
    r = rows(tr, "tq_train")
    random.Random(43).shuffle(r)
    write_jsonl(os.path.join(data, "trivia_train.jsonl"), r[:6000])
    r = rows(va, "tq_val")
    random.Random(47).shuffle(r)
    write_jsonl(os.path.join(data, "trivia_test.jsonl"), r[:1000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--model-repo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--only-synthetic", action="store_true")
    ap.add_argument("--only-general", action="store_true", help="only build data/general.jsonl")
    ap.add_argument("--only-nq", action="store_true", help="only build data/nq_{train,test}.jsonl")
    ap.add_argument("--only-wildchat", action="store_true", help="only build data/wildchat_eval.jsonl (PLAN_NOV.md)")
    ap.add_argument("--only-code", action="store_true", help="only build data/code_{train,test}.jsonl (PLAN_NOV.md)")
    ap.add_argument("--only-squad2", action="store_true", help="only build data/squad2_{train,test}.jsonl (paper v3)")
    ap.add_argument("--only-trivia", action="store_true", help="only build data/trivia_{train,test}.jsonl (step p3.26)")
    ap.add_argument("--only-model", action="store_true",
                    help="only download --model-repo into --models (e.g. Qwen/Qwen2.5-7B-Instruct)")
    args = ap.parse_args()
    if args.only_model:
        os.makedirs(args.models, exist_ok=True)
        print("model"); model(args.models, args.model_repo)
        return
    os.makedirs(args.data, exist_ok=True)
    if args.only_wildchat:
        print("wildchat"); wildchat(args.data)
        return
    if args.only_code:
        print("code"); code(args.data)
        return
    if args.only_squad2:
        print("squad2"); squad2(args.data)
        return
    if args.only_trivia:
        print("trivia"); trivia(args.data)
        return
    if args.only_nq:
        print("nq"); nq(args.data)
        return
    if args.only_general:
        print("general"); general(args.data)
        return
    print("synthetic tasks"); synthetic(args.data)
    if args.only_synthetic:
        return
    for name, fn in (("gsm8k", gsm8k), ("dolly", dolly), ("advbench", advbench), ("xstest", xstest), ("mmlu", mmlu),
                     ("general", general), ("nq", nq)):
        print(name)
        fn(args.data)
    if not args.skip_model:
        os.makedirs(args.models, exist_ok=True)
        print("model"); model(args.models, args.model_repo)
    print("done")


if __name__ == "__main__":
    main()
