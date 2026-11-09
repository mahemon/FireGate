"""Greedy evaluation of one stage on every held-out pool, with online tripwire records.

Writes <stage_dir>/eval/<pool>.jsonl (one row per prompt) and summary.json, then a DONE marker. A pool whose
file exists is skipped, so an interrupted evaluation resumes at the next pool. Multi-GPU: generation is split
across ranks, rank 0 writes the files.

The stage-0 evaluation (the base model) is the same for every method and seed of an experiment, so it is cached
under <out_dir>/_base_eval/<key>/ and copied, keyed by model, pools, decoding settings and GPU model (greedy bf16
outputs are reproducible on one GPU model, not across GPU models).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Callable, Dict, List

import torch

from . import dist
from .data import eval_pools
from .generate import generate
from .io_utils import ensure_dir, read_jsonl, save_json, write_jsonl
from .tasks import POOLS


def pool_summary(name: str, rows: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {"n": len(rows)}
    if not rows:
        return out
    keys = rows[0]["m"].keys()
    for k in keys:
        out[k] = sum(r["m"][k] for r in rows) / len(rows)
    if name == "xstest":
        for lab in ("safe", "unsafe"):
            sel = [r for r in rows if r.get("label") == lab]
            if sel:
                out[f"refused_{lab}"] = sum(r["m"]["refused"] for r in sel) / len(sel)
    tids = sorted({t for r in rows for t in r.get("fire", {})})
    for t in tids:
        out[f"fire_{t}"] = sum(r["fire"].get(t) is not None for r in rows) / len(rows)
    if tids:
        out["fire_any"] = sum(any(r["fire"].get(t) is not None for t in tids) for r in rows) / len(rows)
    out["mean_len"] = sum(len(r["out"]) for r in rows) / len(rows)
    return out


def device_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"


def eval_rows(model, tok, state, cfg: dict, name: str, exs: List[dict], encode, stop, *, greedy: bool = True,
              temperature: float = 1.0, seed: int = 0, top_p: float = 1.0) -> List[dict]:
    """One row per prompt of pool `name` (all ranks call this together; every rank gets the rows)."""
    g = cfg["gen"]
    prompts = [encode(POOLS[name].prompt(ex)) for ex in exs]
    res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop, max_new_tokens=g["max_new_tokens"],
                   batch_size=g["eval_batch_size"], greedy=greedy, temperature=temperature, seed=seed, record=True,
                   top_p=top_p)
    dev = device_name()
    texts = [tok.decode(r["out"], skip_special_tokens=True) for r in res]
    ms = POOLS[name].scores(exs, texts)   # the pool's own score, all responses at once (same values)
    rows = []
    for i, (ex, r, text, m) in enumerate(zip(exs, res, texts, ms)):
        row = {"i": i, "id": ex["id"], "plen": r["plen"], "out": r["out"], "text": text,
               "m": m, "fire": r["fire"], "maxs": r["maxs"], "dev": dev}
        if "label" in ex:
            row["label"] = ex.get("label")
        rows.append(row)
    return rows


def evaluate_stage(model, tok, state, cfg: dict, stage_dir: str, encode: Callable[[str], List[int]],
                   stop: set, log, stop_file: str | None) -> dict:
    edir = os.path.join(stage_dir, "eval")
    if dist.is_main():
        ensure_dir(edir)
    dist.barrier()
    done = os.path.join(edir, "DONE")
    summary_path = os.path.join(edir, "summary.json")
    if dist.exists(done):
        return {}
    summary = {"device": device_name(), "tripwires": sorted(state.tripwires), "pools": {}}
    for name, exs in eval_pools(cfg).items():
        path = os.path.join(edir, f"{name}.jsonl")
        if dist.exists(path):
            rows = read_jsonl(path) if dist.is_main() else []
        else:
            dist.check_stop(stop_file)
            t0 = time.time()
            rows = eval_rows(model, tok, state, cfg, name, exs, encode, stop)
            if dist.is_main():
                write_jsonl(path, rows)
            dist.barrier()
            log(f"eval {os.path.basename(stage_dir)}/{name}", secs=round(time.time() - t0), **{
                k: round(v, 4) for k, v in pool_summary(name, rows).items() if k != "n"})
        if dist.is_main():
            summary["pools"][name] = pool_summary(name, rows)
    if dist.is_main():
        save_json(summary_path, summary)
        with open(done, "w") as f:
            f.write("ok\n")
    dist.barrier()
    return summary


# ----------------------------------------------------------------------------- shared stage-0 evaluation
def base_cache_dir(cfg: dict) -> str:
    """Key: everything that determines the stage-0 rows: model, dtype, decoding settings, GPU model, software
    versions and the exact evaluation prompts (which depend on the data files and on the reference split sizes)."""
    import transformers
    g = cfg["gen"]
    prompts = hashlib.sha1()
    for name, exs in eval_pools(cfg).items():
        for ex in exs:
            prompts.update(json.dumps([name, ex.get("id"), POOLS[name].prompt(ex)], sort_keys=True).encode())
    key = {"model": os.path.normpath(cfg["model_path"]), "dtype": cfg["dtype"], "prompts": prompts.hexdigest(),
           "max_new_tokens": g["max_new_tokens"], "max_prompt_tokens": g["max_prompt_tokens"],
           "eval_batch_size": g["eval_batch_size"], "code_version": cfg.get("code_version", 1), "device": device_name(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__}
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(cfg["out_dir"], "_base_eval", h)


def restore_base_eval(cfg: dict, stage_dir: str, log) -> bool:
    """Copy a finished stage-0 evaluation from the cache (same model, pools, decoding and GPU model)."""
    src = os.path.join(base_cache_dir(cfg), "eval")
    dst = os.path.join(stage_dir, "eval")
    ok = dist.exists(os.path.join(src, "DONE")) and not dist.exists(os.path.join(dst, "DONE"))
    if ok and dist.is_main():
        tmp = dst + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
        log("stage0: base-model evaluation copied from the cache", cache=src)
    dist.barrier()
    return ok


def store_base_eval(cfg: dict, stage_dir: str) -> None:
    """Publish this run's stage-0 evaluation to the cache (first writer wins)."""
    if not dist.is_main():
        return
    src = os.path.join(stage_dir, "eval")
    cdir = base_cache_dir(cfg)
    dst = os.path.join(cdir, "eval")
    if not os.path.exists(os.path.join(src, "DONE")) or os.path.exists(os.path.join(dst, "DONE")):
        return
    ensure_dir(cdir)
    import socket
    tmp = f"{dst}.tmp-{socket.gethostname()}-{os.getpid()}"
    try:
        shutil.copytree(src, tmp)
        os.rename(tmp, dst)
    except OSError:  # another run published it first (the cache is an optimization; never fail the run)
        shutil.rmtree(tmp, ignore_errors=True)


def stored_sequences(stage_dir: str, pool: str, encode, exs) -> List[List[int]]:
    """prompt + response token ids of a stored evaluation (for post-hoc scoring of later tripwires)."""
    rows = read_jsonl(os.path.join(stage_dir, "eval", f"{pool}.jsonl"))
    rows.sort(key=lambda r: r["i"])
    seqs = []
    for r, ex in zip(rows, exs):
        p = encode(POOLS[pool].prompt(ex))
        if len(p) != r["plen"]:
            raise RuntimeError(f"prompt encoding changed for {pool} row {r['i']}")
        seqs.append(p + r["out"])
    return seqs
