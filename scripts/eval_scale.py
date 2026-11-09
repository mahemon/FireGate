#!/usr/bin/env python
"""Adapter-scaling baseline (WiSE-FT style, evaluation only): a finished ungated run (default Seq-GRPO) evaluated
with every adapter's update multiplied by alpha. alpha < 1 trades new-task gain for less drift, so the ungated
method gets its own trade-off curve to compare with Tripwire's, at the cost of evaluations only.

    python scripts/eval_scale.py --config configs/v2/overlap_q15.yaml --set seed=0 --base seq --alpha 0.5

Writes a run folder <exp>/s<seed>/scale_a<alpha> (scale_<base>_a<alpha> for another base run; method "scale", base
and alpha in config.json) with the same
stage evaluations as a trained run, so the tables and reports treat it like any other method. Resumable per pool.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from twcore import dist  # noqa: E402
from twcore.config import load_config, run_dir as make_run_dir  # noqa: E402
from twcore.evaluate import device_name, evaluate_stage, restore_base_eval  # noqa: E402
from twcore.gating import GateState, add_adapter, install, load_adapter_state_dict, scale_adapter  # noqa: E402
from twcore.io_utils import Logger, StopRequested, ensure_dir, load_json, save_json  # noqa: E402
from twcore.model import load_model, make_encoder, stop_ids, template_prefix_len  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--base", default="seq", help="finished ungated run to scale (seq, kl, refkl, ...)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--allow-device-change", action="store_true")
    ap.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    dist.init()
    cfg0 = load_config(args.config, args.set)
    src = make_run_dir(cfg0, args.base)
    if not dist.exists(os.path.join(src, "DONE")):
        raise SystemExit(f"{src} is not finished")
    info = load_json(os.path.join(src, "config.json"))
    if info["method"] in ("tripwire", "tripwire_prompt", "router"):
        raise SystemExit("scaling is a baseline for ungated runs")
    cfg = info["config"]
    base = info["method"] + (f"_{info['tag']}" if info.get("tag") else "")
    tag = f"a{args.alpha:.2f}" if base == "seq" else f"{base}_a{args.alpha:.2f}"   # scale_a0.50, scale_kl_a0.50
    dst = make_run_dir(cfg, "scale", tag)
    verdict = "ok"
    if dist.is_main():
        ensure_dir(dst)
        cj = os.path.join(dst, "config.json")
        old = load_json(cj) if os.path.exists(cj) else None
        if old is not None and (old.get("base_run"), old.get("settings_key")) != (base, info.get("settings_key")):
            verdict = f"{dst} holds a scaled {old.get('base_run')} run with other settings; move it away"
        elif old is not None and old.get("device") != device_name() and not args.allow_device_change:
            verdict = (f"{dst} was started on {old.get('device')} but this job has {device_name()}; use that GPU "
                       "model (GPU_TYPE) or pass --allow-device-change")
        elif old is None:
            save_json(cj, {"method": "scale", "tag": tag, "base_method": info["method"], "base_run": base,
                           "alpha": args.alpha, "config": cfg, "device": device_name(),
                           "settings_key": info.get("settings_key")})
    verdict = dist.broadcast(verdict)
    if verdict != "ok":
        if dist.is_main():
            print("ERROR:", verdict, flush=True)
        dist.shutdown()
        sys.exit(2)
    log = Logger(os.path.join(dst, "log.jsonl"), prefix=f"scale {args.base} x{args.alpha} ")
    if dist.exists(os.path.join(dst, "DONE")):
        log("already done")
        dist.finish()
        return
    model, tok = load_model(cfg["model_path"], cfg["dtype"], device=dist.device())
    state = GateState()
    install(model, state, cfg["gate_layer"], cfg["lora"]["targets"])
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    stop = stop_ids(model, tok)
    state.gate_from = template_prefix_len(encode)
    stop_file = args.stop_file or os.environ.get("TRIPWIRE_STOP_FILE")
    try:
        s0 = os.path.join(dst, "stage0")
        restore_base_eval(cfg, s0, log)
        evaluate_stage(model, tok, state, cfg, s0, encode, stop, log, stop_file)
        for k in range(1, len(cfg["stream"]) + 1):
            tid = str(k)
            add_adapter(model, tid, cfg["lora"]["r"], cfg["lora"]["alpha"], bool(cfg["lora"].get("dense", False)))
            load_adapter_state_dict(model, tid, torch.load(os.path.join(src, f"stage{k}", "adapter.pt"),
                                                           map_location="cpu", weights_only=False))
            scale_adapter(model, tid, args.alpha)
            sdir = os.path.join(dst, f"stage{k}")
            evaluate_stage(model, tok, state, cfg, sdir, encode, stop, log, stop_file)
            if dist.is_main():
                with open(os.path.join(sdir, "DONE"), "w") as f:
                    f.write("ok\n")
            dist.barrier()
        if dist.is_main():
            with open(os.path.join(dst, "DONE"), "w") as f:
                f.write("ok\n")
        log("scaled evaluation finished")
        dist.finish()
    except StopRequested:
        log("stop requested; rerun to continue")
        dist.shutdown()
        sys.exit(3)


if __name__ == "__main__":
    main()
