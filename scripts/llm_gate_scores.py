#!/usr/bin/env python
"""Step p3.22 (Nov 4, PLAN_PAPER3.md): a language model as the score of the prompt-time gate, one more score class
beside the linear classifier, the Gaussian ratio, k-NN and the MLP (one GPU job, no training).

    python scripts/llm_gate_scores.py --config configs/v2/overlap_q15.yaml [--judge models/Qwen2.5-7B-Instruct]

A judge model reads each prompt and is asked, once, whether it is a request of the experiment's task (QUESTIONS,
fixed below before any score was computed). The gate score is log p("Yes") - log p("No") at the judge's first answer
token, so the threshold can be PAC-calibrated like any other score (Theorem 3.6); the decision depends on the prompt
alone, so Corollary 3.4 applies unchanged. Every prompt is scored once and the score is stored, so the gate is a fixed
function of the prompt at calibration and at evaluation.

The prompt sets are exactly those of the stored trunk features (scripts/gate_features.py), in the same order, so the
reports of scripts/score_classes.py can read the scores in place of the features. Writes, next to the features,
<out_dir>/<exp>/analysis/llm_scores.npz (one column per key, float32) and llm_scores.json (the judge, the question and
its sha256, and for every key the pool, n and example ids). Scores are cached by (judge, question, prompt) in
<out_dir>/_llm_gate_cache/, so experiments that share prompts (the same task on several models) are scored once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gate_features import prompt_sets  # noqa: E402
from twcore.config import load_config  # noqa: E402
from twcore.evaluate import device_name  # noqa: E402
from twcore.model import load_model  # noqa: E402
from twcore.tasks import POOLS  # noqa: E402

# Written Nov 4, 2026, from the task descriptions of the paper (Section 6.1, Appendix D.1), before any score was
# computed; not tuned on any fitting, calibration or evaluation prompt. Changing a question makes a new gate.
QUESTIONS = {
    "ifc": "a general request followed by one explicit instruction about the form of the answer (for example: write "
           "only in lowercase, use no commas, use exactly three bullet points, end with a given sentence, or stay "
           "under a word limit)",
    "nq": "a short factual question asked on its own, with no passage or instruction attached, that expects a short "
          "answer such as a name, a date, a number or a place",
    "concisectx": "a request that comes with a passage of text and asks for something about that passage, such as "
                  "answering a question from it, extracting information from it or summarising it",
    "squad2": "a reading-comprehension question about a given passage, whose answer is a span of the passage or "
              "which the passage may not answer at all",
}
TEMPLATE = ("A specialised assistant was trained for one kind of request only: {desc}.\n\n"
            "Here is a request:\n\"\"\"\n{text}\n\"\"\"\n\n"
            "Is this request of that kind? Answer with Yes or No only.")
MAX_CHARS = 6000          # longer prompts keep their first 4000 and last 2000 characters
SKIP = ("eval__dolly_fresh",)   # 10,000 extra Dolly requests: a check of the deployed gate only, not needed here


def render(prompt) -> str:
    """the prompt as the judge sees it: a user message, or the turns of a chat with their roles"""
    if isinstance(prompt, str):
        text = prompt
    else:
        text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt)
    if len(text) > MAX_CHARS:
        text = text[:4000] + "\n[...]\n" + text[-2000:]
    return text


def answer_ids(tok, word):
    ids = set()
    for w in (word, " " + word, word.lower(), " " + word.lower()):
        t = tok(w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.add(t[0])
    if not ids:
        raise SystemExit(f"the judge's tokenizer has no single token for {word!r}")
    return sorted(ids)


@torch.no_grad()
def judge_scores(model, tok, texts, desc, batch_size):
    yes, no = answer_ids(tok, "Yes"), answer_ids(tok, "No")
    enc = []
    for t in texts:
        msgs = [{"role": "user", "content": TEMPLATE.format(desc=desc, text=t)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc.append(tok(s, add_special_tokens=False)["input_ids"])
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = np.zeros(len(enc), np.float32)
    dev = next(model.parameters()).device
    pad = tok.pad_token_id
    body = getattr(model, model.base_model_prefix)   # the decoder without the LM head: logits at the last position
    head = model.get_output_embeddings()              # only, not for every position of the batch
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):   # left padding: the last position is the first answer token's
            ids[r, L - len(enc[i]):] = torch.tensor(enc[i])
            att[r, L - len(enc[i]):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)   # positions count real tokens only (left padding)
        h = body(input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev)).last_hidden_state
        logits = head(h[:, -1]).float()
        lp = torch.log_softmax(logits, dim=-1)
        s = torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)
        out[idx] = s.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--judge", default=os.path.join("models", "Qwen2.5-7B-Instruct"))
    ap.add_argument("--rl", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tasks = list(cfg["stream"])
    if len(tasks) != 1 or tasks[0] not in QUESTIONS:
        raise SystemExit(f"{cfg['exp']}: p3.22 takes single-task experiments of {sorted(QUESTIONS)}")
    task = tasks[0]
    desc = QUESTIONS[task]
    qsha = hashlib.sha256((TEMPLATE + "\n" + desc).encode()).hexdigest()
    out_dir = os.path.join(cfg["out_dir"], cfg["exp"], "analysis")
    feat_json = os.path.join(out_dir, "gate_features.json")
    if not os.path.exists(feat_json):
        raise SystemExit(f"{feat_json} is missing: the LLM scores follow the stored prompt sets")
    stored = json.load(open(feat_json))["keys"]
    sets = {k: v for k, v in prompt_sets(cfg, args.rl).items() if k in stored and k not in SKIP}
    for k, (pool, exs) in sets.items():   # the prompts must be the stored ones, in the stored order
        ids = [ex.get("id") for ex in exs]
        if len(ids) != stored[k]["n"] or ids != stored[k]["ids"]:
            raise SystemExit(f"{cfg['exp']} {k}: the prompt set differs from the stored features (rerun gate_features)")
    judge_name = os.path.basename(os.path.normpath(args.judge))
    cache_dir = os.path.join(cfg["out_dir"], "_llm_gate_cache", judge_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{task}_{qsha[:12]}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    t0 = time.time()
    texts = {k: [render(POOLS[pool].prompt(ex)) for ex in exs] for k, (pool, exs) in sets.items()}
    hkey = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    todo = sorted({h: t for ts in texts.values() for t in ts for h in [hkey(t)] if h not in cache}.items())
    print(f"{cfg['exp']}: {sum(len(v) for v in texts.values())} prompts in {len(sets)} sets, "
          f"{len(todo)} not cached; judge {args.judge}", flush=True)
    if todo:
        model, tok = load_model(args.judge, "bfloat16")
        for b in range(0, len(todo), 2048):   # save the cache every 2048 prompts, so a restart loses little
            chunk = todo[b:b + 2048]
            s = judge_scores(model, tok, [t for _, t in chunk], desc, args.batch)
            cache.update({h: float(v) for (h, _), v in zip(chunk, s)})
            tmp = cache_path + ".tmp"
            json.dump(cache, open(tmp, "w"))
            os.replace(tmp, cache_path)
            print(f"  scored {min(b + 2048, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    arrays = {k: np.asarray([cache[hkey(t)] for t in ts], np.float32).reshape(-1, 1) for k, ts in texts.items()}
    meta = {"exp": cfg["exp"], "step": "p3.22", "judge": args.judge, "task": task, "question": desc,
            "template": TEMPLATE, "question_sha256": qsha, "device": device_name(), "stream": tasks,
            "keys": {k: {"pool": pool, "n": len(exs), "ids": [ex.get("id") for ex in exs]}
                     for k, (pool, exs) in sets.items()}, "probes": []}
    tmp = os.path.join(out_dir, "llm_scores.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(out_dir, "llm_scores.npz"))
    json.dump(meta, open(os.path.join(out_dir, "llm_scores.json"), "w"), indent=1)
    print(f"wrote {out_dir}/llm_scores.npz: {len(arrays)} sets ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
