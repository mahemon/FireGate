#!/usr/bin/env python
"""Run one method over the whole task stream (resumable, one or several GPUs).

    python scripts/run_stream.py --config configs/v2/stream_q15.yaml --method tripwire
    python scripts/run_stream.py --config configs/v2/overlap_q15.yaml --method kl --tag b0.2 --set grpo.kl_beta=0.2
    deepspeed --master_port 29511 scripts/run_stream.py --config ... --method tripwire        (all visible GPUs)
    torchrun --nproc_per_node 4 --master_port 29511 scripts/run_stream.py --config ... --method tripwire

Methods: tripwire (ours, anytime tripwire), tripwire_prompt (ours, calibrated decision at the last prompt token),
router (uncalibrated decision at the last prompt token, probe logit > tripwire.threshold), seq (ungated LoRA per
task), kl (+KL to the previous model on the task's samples), replay (+earlier-task prompts), refkl (+KL to the
previous model on the unlabeled reference prompts), olora (+orthogonality to earlier adapters), nullspace (adapter
blind to the input directions of the reference prompts), ewc / mas / l2 (+penalty on the weight change weighted by
the previous model's Fisher, MAS importance or uniformly, from the same reference prompts), agem (A-GEM projection
of the RL gradient against the reference prompts' KL gradient), safemix (paper v3: harmful reference requests with a
refusal reward mixed into every batch). Paper v3: tripwire.protect=[advbench,general] restricts the tripwire's (and
Ref-KL's) reference components to the ones protected.
Outputs: <out_dir>/<exp>[/s<seed>]/<method>[_<tag>]/stage{0..K}/ (eval/, tripwire/, train/, adapter.pt, DONE).
v7: lora.dense=true trains a full-rank update of every target projection instead of LoRA (its training checkpoint is
deleted once adapter.pt is written); grpo.objective=sft trains with SFT on rejection-sampled data (twcore/grpo.py).
Exit code 3 means a stop was requested (time limit); run the same command again to continue from the last
checkpoint. A run refuses to continue on a different GPU model than it started on (greedy bf16 outputs are
reproducible on one GPU model only); pass --allow-device-change to override.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from twcore import dist  # noqa: E402
from twcore.config import deep_update, load_config, run_dir as make_run_dir, settings_key  # noqa: E402
from twcore.data import ref_split, task_split  # noqa: E402
from twcore.evaluate import device_name, evaluate_stage, restore_base_eval, store_base_eval  # noqa: E402
from twcore.gating import (GateState, add_adapter, adapter_state_dict, install,  # noqa: E402
                             load_adapter_state_dict)
from twcore.grpo import GATED, train_stage  # noqa: E402
from twcore.io_utils import Logger, StopRequested, ensure_dir, load_json, save_json  # noqa: E402
from twcore.model import load_model, make_encoder, stop_ids, template_prefix_len  # noqa: E402
from twcore.stage import build_tripwire, reference_prompts  # noqa: E402

METHODS = ("tripwire", "tripwire_prompt", "router", "seq", "kl", "replay", "refkl", "olora", "nullspace", "ewc", "mas",
           "l2", "agem", "safemix")
# method -> tripwire settings it implies
METHOD_TRIPWIRE = {"tripwire_prompt": {"mode": "prompt", "calibrate": True},
                   "router": {"mode": "prompt", "calibrate": False}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--tag", default="", help="suffix of the run folder, for sweeps (e.g. b0.2 -> kl_b0.2)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like grpo.steps=10")
    ap.add_argument("--stop-file", default=None, help="if this file appears, save and exit with code 3")
    ap.add_argument("--force", action="store_true", help="continue even if the settings changed")
    ap.add_argument("--allow-device-change", action="store_true")
    ap.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)  # set by the deepspeed launcher
    args = ap.parse_args()

    dist.init()
    cfg = load_config(args.config, args.set)
    if args.method in METHOD_TRIPWIRE:
        cfg = deep_update(cfg, {"tripwire": METHOD_TRIPWIRE[args.method]})
    run_dir = make_run_dir(cfg, args.method, args.tag)
    key = settings_key(cfg)
    cfg_path = os.path.join(run_dir, "config.json")

    # start-up checks on rank 0; every rank follows its verdict
    verdict, lock = "ok", None
    if dist.is_main():
        ensure_dir(run_dir)
        old = load_json(cfg_path) if os.path.exists(cfg_path) else None
        if old is None and any(d.startswith("stage") for d in os.listdir(run_dir)):
            verdict = (f"{run_dir} holds results without config.json (e.g. an evaluation-only router); "
                       "move it away or use another --tag")
        elif old is not None and old.get("settings_key") != key and not args.force:
            verdict = (f"{run_dir} was started with other settings ({old.get('settings_key')} != {key}); "
                       "use another exp name / --tag, or --force")
        elif old is not None and old.get("device") not in (None, device_name()) and not args.allow_device_change:
            verdict = (f"{run_dir} was started on {old.get('device')} but this job has {device_name()}; submit with "
                       f"GPU_TYPE pinned to that GPU model, or pass --allow-device-change")
        if verdict == "ok":
            lock = open(os.path.join(run_dir, ".lock"), "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                verdict = f"another process is running {run_dir}; not starting a second one"
            except OSError as e:  # file systems without flock support: run without the lock
                print(f"warning: no file lock on {run_dir} ({e}); make sure only one job runs this method", flush=True)
        if verdict == "ok":
            device = old.get("device") if old else None
            save_json(cfg_path, {"settings_key": key, "method": args.method, "tag": args.tag, "config": cfg,
                                 "device": device or device_name(), "world_size": dist.world()})
    if verdict == "ok" and not dist.same_on_all_ranks(device_name()):
        verdict = "the ranks of this job have different GPU models; request one GPU type (GPU_TYPE)"
    verdict = dist.broadcast(verdict)
    if verdict != "ok":
        if dist.is_main():
            print("ERROR:", verdict, flush=True)
        dist.shutdown()
        sys.exit(2)
    stop_file = args.stop_file or os.environ.get("TRIPWIRE_STOP_FILE")
    log = Logger(os.path.join(run_dir, "log.jsonl"), prefix=f"{args.method}{'_' + args.tag if args.tag else ''} ")
    if dist.exists(os.path.join(run_dir, "DONE")):
        log("already done")
        dist.finish()
        return

    torch.manual_seed(int(cfg["seed"]))
    model, tok = load_model(cfg["model_path"], cfg["dtype"], device=dist.device())
    log("model loaded", gpu=device_name(), world_size=dist.world(), layers=len(model.model.layers), settings_key=key)
    if dist.world() > 1:
        log("ranks: " + dist.describe())
    state = GateState()
    install(model, state, cfg["gate_layer"], cfg["lora"]["targets"])
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    stop = stop_ids(model, tok)
    state.gate_from = template_prefix_len(encode)
    gated = args.method in GATED
    state.last_prompt_only = gated and cfg["tripwire"].get("mode", "anytime") == "prompt"
    activation = cfg["tripwire"].get("activation", "decision")
    if activation not in ("decision", "sequence") or (activation == "sequence" and not state.last_prompt_only):
        raise SystemExit(f"tripwire.activation={activation}: 'sequence' needs a prompt-time gate "
                         "(tripwire_prompt or router); otherwise use 'decision'")
    state.whole_sequence = activation == "sequence"
    log("tripwires skip the shared chat-template prefix", tokens=state.gate_from,
        prompt_time_gate=state.last_prompt_only, whole_sequence=state.whole_sequence)

    try:
        s0 = os.path.join(run_dir, "stage0")
        restore_base_eval(cfg, s0, log)
        evaluate_stage(model, tok, state, cfg, s0, encode, stop, log, stop_file)
        store_base_eval(cfg, s0)
        for k, task in enumerate(cfg["stream"], start=1):
            sdir = os.path.join(run_dir, f"stage{k}")
            if dist.is_main():
                ensure_dir(sdir)
            dist.barrier()
            prev_dir = os.path.join(run_dir, f"stage{k - 1}")
            tid = str(k)
            prev = list(cfg["stream"][:k - 1])
            tw = None
            if gated:
                tw = build_tripwire(model, tok, state, cfg, k, task, prev, sdir, prev_dir, encode, stop, log, stop_file)
            dense = bool(cfg["lora"].get("dense", False))   # v7: full-rank update instead of LoRA
            dist.broadcast_params(add_adapter(model, tid, cfg["lora"]["r"], cfg["lora"]["alpha"], dense))
            if tw is not None:
                state.tripwires[tid] = tw  # the adapter is gated from now on
            adapter_file = os.path.join(sdir, "adapter.pt")
            if dist.exists(adapter_file):
                load_adapter_state_dict(model, tid, torch.load(adapter_file, map_location="cpu", weights_only=False))
            else:
                main_set = task_split(cfg, task, "rl")
                replay = {t: task_split(cfg, t, "rl") for t in prev} if (args.method == "replay" and prev) else None
                if args.method == "safemix":   # paper v3: harmful reference requests (fitting and calibration rows,
                    # never the evaluation rows) with a refusal reward
                    replay = {"advsafe": ref_split(cfg, "advbench", "fit") + ref_split(cfg, "advbench", "cal")}
                refs = (reference_prompts(cfg, prev)
                        if args.method in ("refkl", "nullspace", "ewc", "mas", "l2", "agem") else None)
                log(f"stage{k}: training {task}", method=args.method, prompts=len(main_set))
                train_stage(model, tok, state, cfg, args.method, tid, task, main_set, replay, encode, stop,
                            os.path.join(sdir, "train"), log, stop_file, refs=refs,
                            prev_tids=[str(i) for i in range(1, k)])
                if dist.is_main():
                    tmp = adapter_file + ".tmp"
                    torch.save(adapter_state_dict(model, tid), tmp)
                    os.replace(tmp, adapter_file)
                    ck = os.path.join(sdir, "train", "ckpt.pt")
                    if dense and os.path.exists(ck):   # about 12 GB at 1.5B; adapter.pt now holds the stage
                        os.remove(ck)
                dist.barrier()
            evaluate_stage(model, tok, state, cfg, sdir, encode, stop, log, stop_file)
            if dist.is_main():
                with open(os.path.join(sdir, "DONE"), "w") as f:
                    f.write("ok\n")
            dist.barrier()
        if dist.is_main():
            with open(os.path.join(run_dir, "DONE"), "w") as f:
                f.write("ok\n")
        log("stream finished")
        dist.finish()
    except StopRequested:
        log("stop requested: progress saved; rerun the same command to continue")
        dist.shutdown()
        sys.exit(3)


if __name__ == "__main__":
    main()
