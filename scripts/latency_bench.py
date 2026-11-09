#!/usr/bin/env python
"""Inference cost of the gate (one GPU, no training): latency, throughput and memory of the base model, the gated
adapters of a K-stage stream, a full-rank gated update, and the prompt-time switch between two whole models.

    python scripts/latency_bench.py --config configs/v2/overlap_q15.yaml --name Small [--out paper/v2_results/latency]

Every configuration generates (greedy) the same prompts: half from the config's first task pool, half Dolly.
  base       the model as loaded
  wrapped    the target projections of blocks >= gate_layer wrapped, no adapter
  prompt K   K LoRA adapters (the config's rank and targets), each behind its own prompt-time tripwire (scored at the
             last prompt token; the decision holds for the whole sequence): a K-stage stream, K = 1, 2, 5, 10, 20
  anytime K  the same with tripwires scored at every position (the anytime tripwire), K = 1 and the largest K
  dense      one full-rank update of every wrapped projection behind a prompt-time tripwire, firing on no prompt
             and on every prompt (a full-rank update is skipped in a forward pass where its tripwire fired nowhere)
  switch     Corollary 3.5: the gate reads the old model's block-gate_layer input at the last prompt token (a forward
             pass through the first gate_layer blocks, stopped there); prompts on which it fires go to a second
             whole model, the others to the old one. The gate fires on half the prompts.
Adapters and updates are zero (LoRA B = 0, D = 0) and the tripwires fire on every prompt unless said otherwise, so every
configuration computes the full gated forward pass and must give the base model's outputs token for token (checked):
the times differ only by the gate's computation. Two settings: batched (the config's evaluation batch size, --n
prompts, --max-new tokens) and one prompt at a time (--n1 prompts, --max-new1 tokens). Times are medians of
--repeats runs after a warm-up. Writes <out>/latency_<name>.json, latency_<name>_table.tex and
latency_<name>_numbers.tex (macros \\lat<name>...).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from twcore.config import load_config  # noqa: E402
from twcore.data import eval_pool  # noqa: E402
from twcore.evaluate import device_name  # noqa: E402
from twcore.gating import GateState, Tripwire, add_adapter, install, wrappers_of  # noqa: E402
from twcore.generate import _left_pad, generate  # noqa: E402
from twcore.model import load_model, make_encoder, stop_ids, template_prefix_len  # noqa: E402
from twcore.tasks import POOLS  # noqa: E402

GB = 1024 ** 3
WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty"}   # macro names


class _Stop(Exception):
    pass


def sync():
    torch.cuda.synchronize()


def random_tripwire(d, device, seed, kappa):
    g = torch.Generator().manual_seed(seed)
    tw = Tripwire(d)
    with torch.no_grad():
        tw.a.copy_(torch.randn(d, generator=g) / d ** 0.5)
        tw.kappa.fill_(kappa)
    return tw.to(device)


@torch.no_grad()
def gate_pass(model, tw, prompts, pad_id, batch_size, layer):
    """the switch's gate: the old model's forward pass through blocks 0 .. layer-1 on each prompt, stopped at block
    `layer`, scored at the last prompt token (left padding: the last column)"""
    dev = next(model.parameters()).device
    out = []

    def hook(mod, args, kwargs):
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        out.append(tw.score(h[:, -1]).float().cpu())
        raise _Stop

    hd = model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for b0 in range(0, len(prompts), batch_size):
            ids, attn = _left_pad(prompts[b0:b0 + batch_size], pad_id, dev)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            try:
                model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            except _Stop:
                pass
    finally:
        hd.remove()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="", help="letters only: macro prefix \\lat<name>...")
    ap.add_argument("--out", default="paper/v2_results/latency")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n1", type=int, default=16)
    ap.add_argument("--max-new1", type=int, default=64)
    ap.add_argument("--ks", default="1,2,5,10,20")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma list of configurations to leave out (dense, switch, anytime)")
    args = ap.parse_args()
    if args.name and not args.name.isalpha():
        raise SystemExit("--name must be letters only")
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    cfg = load_config(args.config, [])
    os.makedirs(args.out, exist_ok=True)
    model, tok = load_model(cfg["model_path"], cfg["dtype"])
    dev = next(model.parameters()).device
    L, d = int(cfg["gate_layer"]), int(model.config.hidden_size)
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    stop = stop_ids(model, tok)
    task = cfg["stream"][0]
    half = args.n // 2
    texs = eval_pool(cfg, task)[:half]
    exs = [(task, e) for e in texs] + [("dolly", e) for e in eval_pool(cfg, "dolly")[:args.n - len(texs)]]
    prompts = [encode(POOLS[p].prompt(e)) for p, e in exs]
    nt = len(texs)
    single = prompts[:args.n1 // 2] + prompts[nt:nt + args.n1 - args.n1 // 2]
    bs = int(cfg["gen"]["eval_batch_size"])
    settings = {"batched": (prompts, bs, args.max_new), "single": (single, 1, args.max_new1)}
    res = {"model": cfg["model_path"], "gpu": device_name(), "gate_layer": L, "layers": len(model.model.layers),
           "lora": cfg["lora"], "n": len(prompts), "n1": len(single), "max_new": args.max_new,
           "max_new1": args.max_new1, "batch_size": bs, "weights_gb": sum(p.numel() * p.element_size()
                                                                       for p in model.parameters()) / GB,
           "configs": []}
    print(f"{res['model']} on {res['gpu']}: {len(prompts)} prompts, batch {bs}, {args.max_new} new tokens", flush=True)
    ref = {}

    def run(name, gen_fn, extra=None, refs=None):
        """gen_fn(prompts, batch_size, max_new) -> generation results; times every setting. refs: {setting: outputs}
        to compare with instead of the base model's (the switch batches its two halves apart)"""
        row = {"config": name, **(extra or {})}
        torch.cuda.reset_peak_memory_stats()
        static = torch.cuda.memory_allocated() / GB
        for sname, (ps, b, mn) in settings.items():
            ts, outs = [], None
            gen_fn(ps[:min(len(ps), b)], b, 8)   # warm-up
            for _ in range(args.repeats):
                sync()
                t0 = time.perf_counter()
                outs = gen_fn(ps, b, mn)
                sync()
                ts.append(time.perf_counter() - t0)
            toks = sum(len(r["out"]) for r in outs)
            seqs = [r["out"] for r in outs]
            if name == "base":
                ref[sname] = seqs
            same = sum(a == b_ for a, b_ in zip(seqs, (refs or ref)[sname]))
            row[sname] = {"secs": statistics.median(ts), "secs_all": ts, "tokens": toks,
                          "tok_per_s": toks / statistics.median(ts), "ms_per_prompt": 1000 * statistics.median(ts) / len(ps),
                          "identical": same, "of": len(ps)}
        row["static_gb"] = static
        row["peak_gb"] = torch.cuda.max_memory_allocated() / GB
        b0 = res["configs"][0] if res["configs"] else None
        rel = "" if b0 is None else f" (x{row['batched']['secs'] / b0['batched']['secs']:.3f} batched, " \
                                    f"x{row['single']['secs'] / b0['single']['secs']:.3f} single)"
        print(f"  {name}: batched {row['batched']['secs']:.1f} s, {row['batched']['tok_per_s']:.0f} tok/s; single "
              f"{row['single']['ms_per_prompt']:.0f} ms/prompt; peak {row['peak_gb']:.1f} GB{rel}; identical "
              f"{row['batched']['identical']}/{row['batched']['of']}, {row['single']['identical']}/{row['single']['of']}",
              flush=True)
        if row["batched"]["identical"] != row["batched"]["of"] or row["single"]["identical"] != row["single"]["of"]:
            print(f"  WARNING: {name} does not reproduce the base model's outputs", flush=True)
        res["configs"].append(row)
        return row

    def plain(m):
        st = GateState()
        return lambda ps, b, mn: generate(m, st, ps, pad_id=tok.pad_token_id, stop=stop, max_new_tokens=mn,
                                          batch_size=b, distributed=False)

    run("base", plain(model))

    # ---- the switch between two whole models (before the base model is wrapped)
    def switch_section():
        other = copy.deepcopy(model)   # the "new" model: the same weights, so the outputs can be checked
        tw = random_tripwire(d, dev, 1, 0.0)
        sc = gate_pass(model, tw, prompts, tok.pad_token_id, bs, L)
        with torch.no_grad():
            tw.kappa.fill_(float(sc.median()))   # fires on half the prompts
        ga, gb = plain(model), plain(other)

        def switch(ps, b, mn, both_old=False):
            fire = (gate_pass(model, tw, ps, tok.pad_token_id, b, L) > tw.kappa.item()).tolist()
            i1 = [i for i, f in enumerate(fire) if f]
            i0 = [i for i, f in enumerate(fire) if not f]
            out = [None] * len(ps)
            for idx, g in ((i0, ga), (i1, ga if both_old else gb)):
                if idx:
                    for i, r in zip(idx, g([ps[i] for i in idx], b, mn)):
                        out[i] = r
            return out
        # the switch batches its two halves apart, so its reference is the old model on the same two halves (bf16
        # results depend on the batch); the second model is a copy, so the outputs must be equal
        refs = {sn: [r["out"] for r in switch(ps, b, mn, both_old=True)] for sn, (ps, b, mn) in settings.items()}
        row = run("switch", switch, extra={"fires": float((sc > tw.kappa.item()).float().mean())}, refs=refs)
        ts = []
        for _ in range(args.repeats):   # the gate alone, batched
            sync()
            t0 = time.perf_counter()
            gate_pass(model, tw, prompts, tok.pad_token_id, bs, L)
            sync()
            ts.append(time.perf_counter() - t0)
        row["gate_secs"] = statistics.median(ts)
        row["gate_share"] = row["gate_secs"] / res["configs"][0]["batched"]["secs"]
        print(f"  switch gate alone: {1000 * row['gate_secs'] / len(prompts):.2f} ms per prompt "
              f"({100 * row['gate_share']:.2f}% of the base model's batched time)", flush=True)

    if "switch" not in skip:
        switch_section()
        gc.collect()
        torch.cuda.empty_cache()

    # ---- gated adapters
    state = GateState()
    install(model, state, L, cfg["lora"]["targets"])
    state.gate_from = template_prefix_len(encode)

    def gated(prompt_time):
        def g(ps, b, mn):
            state.last_prompt_only = prompt_time
            state.whole_sequence = prompt_time
            return generate(model, state, ps, pad_id=tok.pad_token_id, stop=stop, max_new_tokens=mn, batch_size=b,
                            distributed=False)
        return g

    run("wrapped", gated(True))
    ks = sorted(int(k) for k in args.ks.split(",") if k.strip())
    if any(k not in WORDS for k in ks):
        raise SystemExit(f"--ks: values from {sorted(WORDS)}")
    tws, n_lora = {}, 0
    for K in ks:
        while len(tws) < K:
            tid = f"s{len(tws) + 1}"
            ps = add_adapter(model, tid, int(cfg["lora"]["r"]), float(cfg["lora"]["alpha"]))
            n_lora = sum(p.numel() for p in ps)
            tws[tid] = random_tripwire(d, dev, 100 + len(tws), float("-inf"))   # fires on every prompt
        state.tripwires = dict(tws)
        run(f"prompt {K}", gated(True), extra={"K": K, "adapter_params": n_lora,
                                               "adapter_gb": K * n_lora * 4 / GB})
        if "anytime" not in skip and K in (ks[0], ks[-1]):
            run(f"anytime {K}", gated(False), extra={"K": K, "adapter_params": n_lora,
                                                      "adapter_gb": K * n_lora * 4 / GB})

    # ---- one full-rank update (blocks >= gate_layer), the LoRA adapters switched off
    if "dense" not in skip:
        state.tripwires = {}
        for _, w in wrappers_of(model):   # free the LoRA adapters, so the dense rows' memory is their own
            for t in tws:
                del w.lora_A[t], w.lora_B[t]
                w.scale.pop(t, None)
        gc.collect()
        torch.cuda.empty_cache()
        ps = add_adapter(model, "dense", 0, 1.0, dense=True)
        n_dense = sum(p.numel() for p in ps)
        # "fires on all" first: it builds the bf16 copy of the update, which then stays in memory for both rows
        for kappa, what in ((float("-inf"), "dense, fires on all"), (float("inf"), "dense, fires on none")):
            state.tripwires = {"dense": random_tripwire(d, dev, 7, kappa)}
            run(what, gated(True), extra={"K": 1, "adapter_params": n_dense, "adapter_gb": n_dense * 4 / GB})

    tag = args.name.lower() or "x"
    with open(os.path.join(args.out, f"latency_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    base = res["configs"][0]
    lines = [r"\begin{tabular}{l r r r r r}", r"\toprule",
             r"Configuration & Time (batched) & Tokens/s & Time (one prompt) & Peak GB & Outputs equal \\", r"\midrule"]
    nums = {}
    pre = "lat" + args.name
    for r in res["configs"]:
        rb = r["batched"]["secs"] / base["batched"]["secs"]
        rs = r["single"]["secs"] / base["single"]["secs"]
        eq = "yes" if all(r[s]["identical"] == r[s]["of"] for s in settings) else "no"
        lines.append(f"{r['config']} & $\\times${rb:.3f} & {r['batched']['tok_per_s']:.0f} & $\\times${rs:.3f} & "
                     f"{r['peak_gb']:.1f} & {eq} \\\\")
        key = "".join(w.capitalize() for w in r["config"].replace(",", "").split() if w.isalpha())
        num = "".join(WORDS.get(int(w), "") for w in r["config"].split() if w.isdigit())
        k = pre + key + num
        if k.isalpha():
            nums[k + "Rel"] = f"{rb:.3f}"
            nums[k + "RelOne"] = f"{rs:.3f}"
            nums[k + "Peak"] = f"{r['peak_gb']:.1f}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, f"latency_{tag}_table.tex"), "w") as fh:
        fh.write(f"% {res['model']} on {res['gpu']}; {res['n']} prompts, batch {bs}, {args.max_new} new tokens; "
                 f"one prompt at a time: {res['n1']} prompts, {args.max_new1} new tokens\n" + "\n".join(lines) + "\n")
    sw = [r for r in res["configs"] if r["config"] == "switch"]
    if sw:
        nums[pre + "GateMs"] = f"{1000 * sw[0]['gate_secs'] / len(prompts):.2f}"
        nums[pre + "GateShare"] = f"{100 * sw[0]['gate_share']:.2f}"
    nums[pre + "BaseGb"] = f"{res['weights_gb']:.1f}"
    with open(os.path.join(args.out, f"latency_{tag}_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/latency_bench.py\n")
        for k in sorted(nums):
            if k.isalpha():
                fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")
    print(f"wrote {args.out}/latency_{tag}*", flush=True)


if __name__ == "__main__":
    main()
