"""Model loading, prompt encoding and stopping tokens."""
from __future__ import annotations

import json
import os
from typing import Callable, List, Set, Union

import torch

Message = dict
PromptLike = Union[str, List[Message]]
FIXED_DATE = "26 Jul 2024"   # Llama 3's own default "Today Date" (PLAN_SCALE.md); other templates ignore it


def pick_device() -> torch.device:
    """cuda:LOCAL_RANK (one GPU per process in multi-GPU runs), or the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")


def load_model(path: str, dtype: str = "bfloat16", device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": tdtype} if (major, minor) >= (4, 56) else {"torch_dtype": tdtype}
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="sdpa", **kw)
    if next(model.parameters()).dtype != tdtype:  # some versions ignore torch_dtype
        model = model.to(tdtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def stop_ids(model, tok) -> Set[int]:
    ids = set()
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(int(x) for x in e)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return ids


def template_prefix_len(encode: Callable[[str], List[int]]) -> int:
    """Number of leading prompt tokens that are identical for every user message (system prompt, role tags)."""
    samples = [encode(t) for t in ("Hello there.", "1234 + 5 = ?", "Zebra crossing", "(1) knights; (2) knaves", "\u00e9t\u00e9")]
    n = min(len(s) for s in samples)
    P = 0
    while P < n and all(s[P] == samples[0][P] for s in samples):
        P += 1
    return P


def make_encoder(tok, max_prompt_tokens: int) -> Callable[[PromptLike], List[int]]:
    """Prompt -> token ids of the chat-formatted prompt (ends with the assistant header).
    A prompt is a user message (str) or a list of chat messages ({"role", "content"}; system / earlier turns)."""
    cache = {}

    def encode(prompt: PromptLike) -> List[int]:
        key = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        if key in cache:
            return list(cache[key])
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if getattr(tok, "chat_template", None):
            try:   # date_string: a fixed date for templates that print one (Llama 3), so a prompt never changes
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
            except Exception:
                # templates without a system role (Gemma-2 raises "System role not supported"): the system text
                # opens the first user message instead; templates that accept a system message never get here
                if not (len(msgs) > 1 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"):
                    raise
                merged = [{"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}] + msgs[2:]
                s = tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True, date_string=FIXED_DATE)
        else:  # tiny test tokenizers without a template
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) > max_prompt_tokens:  # keep the template head and the end of the prompt
            ids = ids[:64] + ids[-(max_prompt_tokens - 64):]
        cache[key] = list(ids)
        return list(ids)

    return encode
