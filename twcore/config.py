"""Configuration: defaults, YAML loading, dotted overrides, settings key (no torch)."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List

DEFAULTS: Dict[str, Any] = {
    # 2: per-batch sampling seeds, multi-GPU, new methods (changes the settings key, so a v1 run directory is
    # never resumed with this code)
    "code_version": 2,
    "exp": "stream_qwen15",
    # true: runs live in <out_dir>/<exp>/s<seed>/<method>[_<tag>] (one folder per seed)
    "seed_dirs": False,
    "model_path": "models/Qwen2.5-1.5B-Instruct",
    "data_dir": "data",
    "out_dir": "runs",
    "dtype": "bfloat16",
    "seed": 0,
    "stream": ["gsm8k", "countdown", "kk"],
    # blocks 0..gate_layer-1 form the frozen trunk; the tripwire reads the input of block gate_layer;
    # adapters live in blocks gate_layer..L-1 (0-indexed). 8 of 28 for Qwen2.5-1.5B.
    "gate_layer": 8,
    "lora": {"r": 16, "alpha": 32,
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             "dense": False},   # v7: True = a full-rank update of every target projection (full fine-tuning of them)
    "gen": {"max_new_tokens": 512, "max_prompt_tokens": 1024,
            "eval_batch_size": 128, "rollout_batch_size": 128, "feature_batch_size": 32},
    "grpo": {"steps": 150, "prompts_per_step": 16, "group_size": 8, "lr": 5.0e-5, "warmup_steps": 10,
             "grad_clip": 1.0, "micro_batch": 8, "temperature": 1.0, "ckpt_every": 25,
             "kl_beta": 0.05, "replay_frac": 0.25, "weight_decay": 0.0,
             # refkl: beta * KL(pi_old || pi) on pi_old's samples for the reference prompts (LwF-style),
             # refkl_batch reference sequences per step
             "refkl_beta": 1.0, "refkl_batch": 32, "refkl_temperature": 1.0, "refkl_max_prompts": 1024,
             "refkl_samples": 1,     # samples of the previous model per reference prompt (v5)
             # olora: lambda * sum ||A_new A_prev^T||_F^2 over wrapped projections (O-LoRA)
             "olora_lambda": 0.5,
             # nullspace: block the input directions holding `nullspace_energy` of the reference inputs' energy
             # (at most nullspace_max_rank per projection), estimated on nullspace_max_seqs reference sequences
             "nullspace_energy": 0.95, "nullspace_max_rank": 256, "nullspace_max_seqs": 512,
             "nullspace_refresh": 0,  # > 0: recompute the subspaces on the current model every that many steps (v5)
             # ewc: lambda * 0.5 * sum F dW^2, K-FAC diagonal Fisher of the previous model on ewc_max_seqs of its own
             # samples for the reference prompts (the same samples as refkl), ewc_batch sequences per backward pass
             "ewc_lambda": 1.0, "ewc_max_seqs": 256, "ewc_batch": 4,
             # mas / l2: the same penalty with the MAS or a uniform importance map, rescaled to the Fisher's mass
             "mas_lambda": 1.0, "l2_lambda": 1.0,
             # v7: objective "sft" = supervised fine-tuning on rejection-sampled data (sft_samples per prompt from the
             # model before the stage, the first accepted one kept), sft_batch sequences per step
             "objective": "grpo", "sft_samples": 4, "sft_batch": 64, "sft_temperature": 1.0, "sft_max_prompts": 2400,
             # PLAN_SFT.md: sft_source self (as above) / gold (the dataset's reference response, tasks.GOLD) /
             # teacher (samples of the model sft_teacher, accepted by the reward)
             "sft_source": "self", "sft_teacher": ""},
    # reference sets: Dolly / AdvBench rows are [fit | calibration | evaluation]; task sets come from `splits`
    # mode: "anytime" (fires at any prompt or response position) or "prompt" (one decision at the last prompt
    # token); calibrate: false uses `threshold` on the probe logit instead of PAC calibration (router baseline);
    # activation (prompt mode only): "decision" = the adapter acts from the last prompt token on, "sequence" = a fired
    # decision switches the adapter on for the whole sequence, prompt included (same certificate)
    "tripwire": {"mode": "anytime", "calibrate": True, "threshold": 0.0, "activation": "decision",
                 "eps": 0.03, "delta": 0.05, "positions_per_traj": 48, "l2": 1.0e-4, "lbfgs_iters": 200,
                 "fit_ref": {"dolly": 600, "advbench": 100},
                 "cal_ref": {"dolly": 700, "advbench": 170}},
    # held-out evaluation pools and their sizes
    "pools": {"gsm8k": 500, "countdown": 500, "kk": 500, "mmlu": 500, "advbench": 250,
              "xstest": 450, "dolly": 400},
    # index layout of each task's training split
    "splits": {"rl": [0, 2400], "fit_new": [2400, 3000], "fit_old": [3000, 3300], "cal_old": [3300, 3600]},
    # evaluation-only robustness checks (scripts/eval_extra.py)
    "extra": {"sample_temperature": 0.7, "sample_seeds": [11, 12]},
}

# keys that do not change results (excluded from the settings key)
_NON_RESULT_KEYS = {"out_dir", "exp", "extra"}


def run_dir(cfg: dict, method: str, tag: str = "") -> str:
    """<out_dir>/<exp>[/s<seed>]/<method>[_<tag>]"""
    import os
    parts = [cfg["out_dir"], cfg["exp"]]
    if cfg.get("seed_dirs"):
        parts.append(f"s{int(cfg['seed'])}")
    parts.append(method + (f"_{tag}" if tag else ""))
    return os.path.join(*parts)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s: str):
    try:
        return json.loads(s)
    except Exception:
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        return s


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like a.b=c: {ov}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path: str | None, overrides: List[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise SystemExit("pyyaml is needed for YAML configs (pip install pyyaml) or use a .json config") from e
            user = yaml.safe_load(text) or {}
        else:
            user = json.loads(text)
        cfg = deep_update(cfg, user)
    return apply_overrides(cfg, overrides or [])


# settings added after runs had started (v6 -> v7): left out of the key while they keep their default value, so
# runs started earlier keep their key and continue
_LATE_DEFAULTS = {("grpo", "nullspace_energy"): 0.95, ("grpo", "nullspace_max_rank"): 256,
                  ("grpo", "nullspace_max_seqs"): 512,
                  ("grpo", "ewc_lambda"): 1.0, ("grpo", "ewc_max_seqs"): 256, ("grpo", "ewc_batch"): 4,
                  ("grpo", "mas_lambda"): 1.0, ("grpo", "l2_lambda"): 1.0,
                  ("tripwire", "activation"): "decision",
                  ("grpo", "refkl_samples"): 1, ("grpo", "nullspace_refresh"): 0,   # v5
                  ("lora", "dense"): False, ("grpo", "objective"): "grpo", ("grpo", "sft_samples"): 4,   # v7
                  ("grpo", "sft_batch"): 64, ("grpo", "sft_temperature"): 1.0, ("grpo", "sft_max_prompts"): 2400,
                  ("grpo", "sft_source"): "self", ("grpo", "sft_teacher"): ""}   # PLAN_SFT.md


def settings_key(cfg: dict) -> str:
    clean = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items() if k not in _NON_RESULT_KEYS}
    for (sec, key), default in _LATE_DEFAULTS.items():
        if isinstance(clean.get(sec), dict) and clean[sec].get(key) == default:
            del clean[sec][key]
    blob = json.dumps(clean, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
