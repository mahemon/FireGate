#!/usr/bin/env python
"""Router baseline, evaluation only: Tripwire's trained adapters and probes, but each adapter is switched on by
one uncalibrated decision per prompt (probe score at the last prompt token > 0, i.e. p > 0.5) instead of a
calibrated tripwire. Same adapters, so any difference from Tripwire comes from the gate alone; no certificate.

    python scripts/eval_router.py --config configs/stream_qwen15_v2.yaml
    (reads <out_dir>/<exp>/tripwire, writes <out_dir>/<exp>/router)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from twcore.config import load_config  # noqa: E402
from twcore.evaluate import evaluate_stage  # noqa: E402
from twcore.gating import GateState, add_adapter, install, load_adapter_state_dict, tripwire_from_state  # noqa: E402
from twcore.io_utils import Logger, StopRequested, ensure_dir  # noqa: E402
from twcore.model import load_model, make_encoder, stop_ids, template_prefix_len  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.0, help="probe logit threshold (0 = probability 0.5)")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    exp_dir = os.path.join(cfg["out_dir"], cfg["exp"])
    src = os.path.join(exp_dir, "tripwire")
    dst = ensure_dir(os.path.join(exp_dir, "router"))
    log = Logger(os.path.join(dst, "log.jsonl"), prefix="router ")
    if not os.path.exists(os.path.join(src, "DONE")):
        raise SystemExit(f"{src} is not finished; the router reuses its adapters")
    # stage 0 is the base model: identical to Tripwire's stage 0
    if not os.path.exists(os.path.join(dst, "stage0", "eval", "DONE")):
        shutil.copytree(os.path.join(src, "stage0", "eval"), os.path.join(dst, "stage0", "eval"), dirs_exist_ok=True)

    model, tok = load_model(cfg["model_path"], cfg["dtype"])
    state = GateState()
    install(model, state, cfg["gate_layer"], cfg["lora"]["targets"])
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    stop = stop_ids(model, tok)
    state.gate_from = template_prefix_len(encode)
    state.last_prompt_only = True
    log("router: one decision per prompt at the last prompt token", threshold=args.threshold)
    try:
        for k, task in enumerate(cfg["stream"], start=1):
            tid = str(k)
            sd = os.path.join(src, f"stage{k}")
            tw = tripwire_from_state(torch.load(os.path.join(sd, "tripwire", "tripwire.pt"), map_location="cpu",
                                                weights_only=False), next(model.parameters()).device)
            tw.kappa.fill_(args.threshold)
            add_adapter(model, tid, cfg["lora"]["r"], cfg["lora"]["alpha"])
            load_adapter_state_dict(model, tid, torch.load(os.path.join(sd, "adapter.pt"), map_location="cpu",
                                                           weights_only=False))
            state.tripwires[tid] = tw
            evaluate_stage(model, tok, state, cfg, ensure_dir(os.path.join(dst, f"stage{k}")), encode, stop, log,
                           args.stop_file)
            with open(os.path.join(dst, f"stage{k}", "DONE"), "w") as f:
                f.write("ok\n")
        with open(os.path.join(dst, "DONE"), "w") as f:
            f.write("ok\n")
        log("router evaluation finished")
    except StopRequested:
        log("stop requested; rerun to continue")
        sys.exit(3)


if __name__ == "__main__":
    main()
