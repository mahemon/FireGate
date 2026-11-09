#!/usr/bin/env python
"""Robustness evaluations of one finished run (evaluation only; resumable per pool).

    python scripts/eval_extra.py --config configs/v2/stream_q15.yaml --method tripwire --set seed=0 --what all

what (comma-separated, or all):
  advwrap  harmful AdvBench requests written in the format of each trained task (advwrap_gsm8k, ...): the case in
           which a harmful request can fire a tripwire and reach an adapter. Reports firing, refusal of the final
           model vs the base model, and refusal on the fired subset.
  sample   reference pools decoded by sampling (temperature extra.sample_temperature, seeds extra.sample_seeds),
           final model and base model with the same seed: Theorem 3.2 with shared randomness says every non-fired
           response is identical; the script counts violations and reports score changes under sampling.
  shift    reference requests behind another system prompt or after an earlier exchange (dolly_sys, dolly_turn,
           advbench_sys): how firing and drift move when the prompt format shifts.
  decode   (not part of all) the reference pools under other decoding rules, final and base model with the same seed:
           temperatures and nucleus sampling from extra.decode_settings ([[T, top_p], ...], default 0.3/1.0/1.0 with
           top-p 0.9), seeds extra.decode_seeds (default [11]). Per pool: firing and its Clopper--Pearson bound, the
           share of changed responses under shared randomness (by the coupling inequality an upper bound on the TV
           distance of the two laws on the pool, and an estimate of one for the population; its Clopper--Pearson bound
           is stored too), changes on non-fired prompts (must be 0), score changes, and the Kolmogorov--Smirnov distance
           of response lengths with a 95% lower confidence bound (a lower bound on the TV distance). Serving control:
           the base model decoded twice with the same seed and rule but a different batch size.
  wildchat (not part of all; Nov 1, PLAN_NOV.md) the first WILDCHAT_N real-user requests of data/wildchat_eval.jsonl
           (never used to fit or calibrate), greedy, final and base model: firing and its Clopper--Pearson bound,
           changed responses, changes on non-fired prompts (must be 0). scripts/switch_report.py reads the two files
           (extra/wildchat/{final,base}_wildchat.jsonl), so an ungated run is also evaluated behind every gate there.
           Skipped for runs whose config lists the pool (their stage evaluations already contain it).
  external (not part of all; Nov 2, step p3.12) the harmful requests of benchmarks we did not write
           (twcore/external.py: HarmBench, HarmBench contextual, StrongREJECT), greedy, final and base model: firing,
           refusal, changes on non-fired prompts (must be 0); --set extra.external_batch_size=16 lowers the
           generation batch (final and base alike). scripts/external_report.py reads
           extra/external/{final,base}_<set>.jsonl and puts an ungated run behind each gate.
  widen    (not part of all; Nov 5, step p3.23) English harmful prompts for the wide harmful component of Concise
           (twcore/widen.generation_sets: the test part of MultiJail English and of HarmBench as a second turn and
           behind a system prompt, Do-Not-Answer, the English Aya red-teaming rows), greedy, final and base model:
           firing, refusal, changes on non-fired prompts (must be 0); also --what external when its files are
           missing (HarmBench and StrongREJECT). scripts/widen_report.py reads extra/widen/{final,base}_<set>.jsonl.
  stress   (not part of all; paper v3, PLAN_PAPER3.md) the stress families of the run's tasks (twcore/stress.py: the
           AdvBench evaluation requests written to look like the task in ways the remedy never saw), greedy, final and
           base model: firing and its Clopper--Pearson bound, refusal, changes on non-fired prompts (must be 0).
           scripts/switch_report.py and scripts/v3_analyses.py read extra/stress/{final,base}_<family>.jsonl.
The base model is the same network with every adapter switched off (tripwires still run, so its firing record must
equal the final model's). Output: <run>/extra/<what>/<variant>_<pool>[_seed].jsonl and <run>/extra/summary.json.
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from twcore import dist  # noqa: E402
from twcore.certify import clopper_pearson_upper  # noqa: E402
from twcore.config import load_config, run_dir as make_run_dir  # noqa: E402
from twcore.data import WILDCHAT_N, eval_pool, wildchat_rows  # noqa: E402
from twcore.evaluate import eval_rows  # noqa: E402
from twcore.gating import GateState, add_adapter, install, load_adapter_state_dict, tripwire_from_state  # noqa: E402
from twcore.grpo import GATED  # noqa: E402
from twcore.io_utils import Logger, ensure_dir, load_json, read_jsonl, save_json, write_jsonl  # noqa: E402
from twcore.model import load_model, make_encoder, stop_ids, template_prefix_len  # noqa: E402
from twcore.stress import set_data_dir, stress_pools  # noqa: E402
from twcore.external import external_pools, external_rows  # noqa: E402
from twcore.prospective import prospective_rows  # noqa: E402  (registers the pr_* pools)
from twcore.widen import generation_sets  # noqa: E402  (registers the wd_* pools)
from twcore.tasks import POOLS  # noqa: E402

WHAT = ("advwrap", "sample", "shift")   # "all"; decode only when asked for
DECODE_SETTINGS = ((0.3, 1.0), (1.0, 1.0), (1.0, 0.9))   # (temperature, top_p)
DECODE_POOLS = (("dolly", "leak"), ("advbench", "refused"), ("xstest", "refused"), ("mmlu", "correct"))


def fired(r) -> bool:
    return any(v is not None for v in r.get("fire", {}).values())


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pool_stats(final, base, metric):
    """Firing, certificate, exactness on non-fired prompts, score change overall and on the fired subset."""
    n = len(final)
    X = sum(fired(r) for r in final)
    nf = [(a, b) for a, b in zip(final, base) if not fired(a)]
    fi = [(a, b) for a, b in zip(final, base) if fired(a)]
    out = {"n": n, "fired": X, "fire_rate": X / n if n else float("nan"),
           "cert_cp95": clopper_pearson_upper(X, n) if n else float("nan"),
           "fire_record_equal": all(fired(a) == fired(b) for a, b in zip(final, base)),
           "nonfired_changed": sum(a["out"] != b["out"] for a, b in nf),
           "changed": mean(float(a["out"] != b["out"]) for a, b in zip(final, base))}
    if metric:
        out[f"{metric}_final"] = mean(r["m"][metric] for r in final)
        out[f"{metric}_base"] = mean(r["m"][metric] for r in base)
        out[f"{metric}_delta"] = out[f"{metric}_final"] - out[f"{metric}_base"]
        out[f"{metric}_fired_final"] = mean(a["m"][metric] for a, _ in fi)
        out[f"{metric}_fired_base"] = mean(b["m"][metric] for _, b in fi)
    return out


def length_stats(final, base, alpha=0.05):
    """Two-sample Kolmogorov--Smirnov distance of response lengths and a lower confidence bound on the TV distance
    of the two response distributions: TV >= KS of any statistic, and by the DKW inequality (each empirical CDF within
    sqrt(log(4/alpha)/(2n)) of its own, union over the two samples) the true KS is at least the estimate minus twice that."""
    a = sorted(len(r["out"]) for r in final)
    b = sorted(len(r["out"]) for r in base)
    n = len(a)
    if not n or len(b) != n:
        return {}
    ks = max(abs(bisect.bisect_right(a, v) - bisect.bisect_right(b, v)) / n for v in set(a) | set(b))
    slack = 2 * math.sqrt(math.log(4 / alpha) / (2 * n))
    return {"len_final": mean(a), "len_base": mean(b), "len_ks": ks, "len_ks_lb95": max(0.0, ks - slack)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--what", default="all")
    ap.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    dist.init()
    rdir = make_run_dir(load_config(args.config, args.set), args.method, args.tag)
    if not dist.exists(os.path.join(rdir, "DONE")):
        raise SystemExit(f"{rdir} is not finished")
    if load_json(os.path.join(rdir, "config.json")).get("method") == "scale":
        raise SystemExit("scale runs have no adapters of their own; run eval_extra.py on the base run")
    cfg = load_json(os.path.join(rdir, "config.json"))["config"]  # exactly the run's settings ...
    cfg["extra"] = load_config(args.config, args.set).get("extra", {})  # ... except the evaluation-only options
    whats = WHAT if args.what == "all" else tuple(w.strip() for w in args.what.split(","))
    log = Logger(os.path.join(rdir, "extra", "log.jsonl"), prefix=f"extra {args.method} ")

    model, tok = load_model(cfg["model_path"], cfg["dtype"], device=dist.device())
    state = GateState()
    install(model, state, cfg["gate_layer"], cfg["lora"]["targets"])
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    stop = stop_ids(model, tok)
    state.gate_from = template_prefix_len(encode)
    gated = args.method in GATED
    state.last_prompt_only = gated and cfg["tripwire"].get("mode", "anytime") == "prompt"
    state.whole_sequence = state.last_prompt_only and cfg["tripwire"].get("activation", "decision") == "sequence"
    dev = next(model.parameters()).device
    tids = []
    for k in range(1, len(cfg["stream"]) + 1):
        tid = str(k)
        sd = os.path.join(rdir, f"stage{k}")
        if gated:
            state.tripwires[tid] = tripwire_from_state(
                torch.load(os.path.join(sd, "tripwire", "tripwire.pt"), map_location="cpu", weights_only=False), dev)
        add_adapter(model, tid, cfg["lora"]["r"], cfg["lora"]["alpha"], bool(cfg["lora"].get("dense", False)))
        load_adapter_state_dict(model, tid, torch.load(os.path.join(sd, "adapter.pt"), map_location="cpu",
                                                       weights_only=False))
        tids.append(tid)

    def run(what, name, exs, variant, greedy=True, temperature=1.0, seed=0, suffix="", top_p=1.0, batch_size=None):
        path = os.path.join(rdir, "extra", what, f"{variant}_{name}{suffix}.jsonl")
        if dist.exists(path):
            return read_jsonl(path)
        state.disabled = set(tids) if variant == "base" else set()
        c = cfg
        if batch_size is not None:    # serving control: same model, seed and rule, a different batch size
            c = dict(cfg, gen=dict(cfg["gen"], eval_batch_size=int(batch_size)))
        try:
            rows = eval_rows(model, tok, state, c, name, exs, encode, stop, greedy=greedy,
                             temperature=temperature, seed=seed, top_p=top_p)
        finally:
            state.disabled = set()
        if dist.is_main():
            ensure_dir(os.path.dirname(path))
            write_jsonl(path, rows)
        dist.barrier()
        log(f"{what}: {variant} {name}{suffix}", n=len(rows), fire=round(mean(float(fired(r)) for r in rows), 4))
        return rows

    summary_path = os.path.join(rdir, "extra", "summary.json")
    summary = load_json(summary_path) if dist.exists(summary_path) and dist.is_main() else {}
    if "advwrap" in whats:
        exs = eval_pool(cfg, "advbench")
        for t in cfg["stream"]:
            name = f"advwrap_{t}"
            if name not in POOLS:
                continue
            fin, base = run("advwrap", name, exs, "final"), run("advwrap", name, exs, "base")
            summary.setdefault("advwrap", {})[name] = pool_stats(fin, base, "refused")
    if "shift" in whats:
        for name in ("dolly_sys", "dolly_turn", "advbench_sys"):
            exs = eval_pool(cfg, POOLS[name].source)
            fin, base = run("shift", name, exs, "final"), run("shift", name, exs, "base")
            metric = "refused" if "advbench" in name else "leak"
            summary.setdefault("shift", {})[name] = pool_stats(fin, base, metric)
    if "sample" in whats:
        T = float(cfg.get("extra", {}).get("sample_temperature", 0.7))
        for name, metric in (("dolly", "leak"), ("advbench", "refused"), ("xstest", "refused"), ("mmlu", "correct")):
            if name not in cfg["pools"]:
                continue
            exs = eval_pool(cfg, name)
            per_seed = []
            for sd in cfg.get("extra", {}).get("sample_seeds", [11, 12]):
                kw = dict(greedy=False, temperature=T, seed=int(sd), suffix=f"_t{T}_s{sd}")
                fin, base = run("sample", name, exs, "final", **kw), run("sample", name, exs, "base", **kw)
                per_seed.append(pool_stats(fin, base, metric))
            summary.setdefault("sample", {})[name] = {"temperature": T, "seeds": per_seed}
    if "decode" in whats:
        ex = cfg.get("extra", {})
        settings = [(float(t), float(p)) for t, p in ex.get("decode_settings", DECODE_SETTINGS)]
        seeds = [int(s) for s in ex.get("decode_seeds", [11])]
        for T, p in settings:
            key = f"t{T:g}" + ("" if p >= 1 else f"_p{p:g}")
            for name, metric in DECODE_POOLS:
                if name not in cfg["pools"]:
                    continue
                exs = eval_pool(cfg, name)
                per_seed = []
                for sd in seeds:
                    kw = dict(greedy=False, temperature=T, top_p=p, seed=sd, suffix=f"_{key}_s{sd}")
                    fin, base = run("decode", name, exs, "final", **kw), run("decode", name, exs, "base", **kw)
                    st = pool_stats(fin, base, metric)
                    st.update(length_stats(fin, base))
                    # the observed share is an estimate of an upper bound on TV; this is a 95% bound on that share
                    st["changed_cp95"] = clopper_pearson_upper(int(round(st["changed"] * st["n"])), st["n"])
                    # final and base must come from the same GPU model (a resumed job may land elsewhere)
                    st["same_device"] = all(x.get("dev") == y.get("dev") for x, y in zip(fin, base))
                    per_seed.append(st)
                summary.setdefault("decode", {}).setdefault(key, {})[name] = {"temperature": T, "top_p": p,
                                                                               "seeds": per_seed}
        # serving control on Dolly and AdvBench: the base model twice under the first setting at temperature 1 (or
        # the first setting), same seed, the second time with an eighth of the batch size
        T, p = next(((t, q) for t, q in settings if t == 1.0), settings[0])
        key = f"t{T:g}" + ("" if p >= 1 else f"_p{p:g}")
        bs = max(1, int(cfg["gen"]["eval_batch_size"]) // 8)
        for name in ("dolly", "advbench"):
            if name not in cfg["pools"]:
                continue
            exs = eval_pool(cfg, name)
            kw = dict(greedy=False, temperature=T, top_p=p, seed=seeds[0])
            a = run("decode", name, exs, "base", suffix=f"_{key}_s{seeds[0]}", **kw)
            b = run("decode", name, exs, "base", suffix=f"_{key}_s{seeds[0]}_bs{bs}", batch_size=bs, **kw)
            summary.setdefault("decode_serving", {})[name] = {
                "setting": key, "batch_sizes": [int(cfg["gen"]["eval_batch_size"]), bs], "n": len(a),
                "changed": mean(float(x["out"] != y["out"]) for x, y in zip(a, b)),
                "fire_record_equal": all(fired(x) == fired(y) for x, y in zip(a, b)),
                "same_device": all(x.get("dev") == y.get("dev") for x, y in zip(a, b))}
    if "wildchat" in whats:
        if "wildchat" in cfg["pools"]:
            log("wildchat: already in the run's pools (stage evaluations); nothing to do")
        else:
            n = int(cfg.get("extra", {}).get("wildchat_n", WILDCHAT_N))
            exs = wildchat_rows(cfg, n)
            fin, base = run("wildchat", "wildchat", exs, "final"), run("wildchat", "wildchat", exs, "base")
            st = pool_stats(fin, base, "refused")
            st["same_device"] = all(x.get("dev") == y.get("dev") for x, y in zip(fin, base))
            summary["wildchat"] = st
    if "stress" in whats:
        set_data_dir(cfg["data_dir"])
        exs = eval_pool(cfg, "advbench")
        for name in stress_pools(list(cfg["stream"])):
            fin, base = run("stress", name, exs, "final"), run("stress", name, exs, "base")
            st = pool_stats(fin, base, "refused")
            st["same_device"] = all(x.get("dev") == y.get("dev") for x, y in zip(fin, base))
            summary.setdefault("stress", {})[name] = st
    if "widen" in whats:   # Nov 5 (step p3.23)
        sets = generation_sets(cfg, prospective_rows)
        if not sets:
            log("widen: no data/pr_*.jsonl or data/wd_*.jsonl (prepare_prospective.py, prepare_widen.py); nothing to do")
        bs = cfg.get("extra", {}).get("external_batch_size")
        for name, exs in sets.items():
            fin, base = run("widen", name, exs, "final", batch_size=bs), run("widen", name, exs, "base", batch_size=bs)
            st = pool_stats(fin, base, "refused")
            st["same_device"] = all(x.get("dev") == y.get("dev") for x, y in zip(fin, base))
            summary.setdefault("widen", {})[name] = st
        if "external" not in whats and "external" not in summary:
            whats = tuple(whats) + ("external",)
    if "external" in whats:
        names = external_pools(cfg)
        if not names:
            log("external: no data/ext_*.jsonl (scripts/prepare_external.py); nothing to do")
        # extra.external_batch_size (Nov 3): a smaller generation batch for these sets, whose passages are long (OLMo-2-7B
        # ran out of memory on an 80 GB A100 at its batch of 64); final and base use the same batch, so they compare
        bs = cfg.get("extra", {}).get("external_batch_size")
        for name in names:
            exs = external_rows(cfg, name)
            fin, base = run("external", name, exs, "final", batch_size=bs), run("external", name, exs, "base", batch_size=bs)
            st = pool_stats(fin, base, "refused")
            st["same_device"] = all(x.get("dev") == y.get("dev") for x, y in zip(fin, base))
            summary.setdefault("external", {})[name] = st
    if dist.is_main():
        save_json(summary_path, summary)
    log("done", what=",".join(whats))
    dist.finish()


if __name__ == "__main__":
    main()
