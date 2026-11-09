"""Tripwire-gated LoRA (Definition 3.1 of the paper).

* Blocks 0..gate_layer-1 are the frozen trunk. A forward pre-hook on block `gate_layer` reads the residual
  stream u_t entering that block, scores it with every tripwire, and turns firings into latched masks
  m_t = 1[the tripwire fired at some valid position <= t of this sequence].
* LoRA adapters live only in blocks gate_layer..L-1. A gated adapter adds its update at position t only where
  m_t is true, via torch.where, so masked positions return the base value bit for bit.
* Two modes: "full" (a whole sequence in one pass; masks are a cumulative OR over positions) and
  "incremental" (decoding with a KV cache; one latch bit per sequence and tripwire is carried across steps).
* Prompt-time gates (last_prompt_only) with whole_sequence (tripwire.activation = "sequence"): the decision at the
  last prompt token switches the adapter on for every position of the sequence, prompt included. The decision
  depends on the prompt only, so a prompt on which it does not fire still runs the old model bit for bit and the
  certificate is unchanged; a fired prompt runs the fully adapted model (as an ungated adapter would).

Exactness argument (Lemma A.1): before the first firing no adapter of that tripwire is applied at any position,
and the trunk is shared, so every hidden state equals the old model's.

Full-rank updates (v7, lora.dense): an adapter can also be a dense update D (out x in, float32, zero at the start) of
the wrapped projection, y = W x + [m] D x: full fine-tuning of every adapted linear layer, gated exactly like a LoRA
adapter (the argument above needs only that the update is switched off where m is false).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class Tripwire(nn.Module):
    """score(u) = a . ((u / rms(u) - mu) / sigma) + b ; fires when score > kappa (all float32 buffers)."""

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.register_buffer("a", torch.zeros(d))
        self.register_buffer("b", torch.zeros(()))
        self.register_buffer("kappa", torch.tensor(math.inf))

    @staticmethod
    def normalize(h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)

    def score(self, h: torch.Tensor) -> torch.Tensor:
        z = (self.normalize(h) - self.mu) / self.sigma
        return z @ self.a + self.b


class GateState:
    """Shared between the tripwire hook (writer) and the gated LoRA modules (readers)."""

    def __init__(self):
        self.tripwires: Dict[str, Tripwire] = {}  # tripwires of gated adapters (adapter id -> tripwire)
        self.shadow: Dict[str, Tripwire] = {}     # scored and recorded only (calibration), no adapter
        self.masks: Dict[str, torch.Tensor] = {}  # adapter id -> bool [B, T] for the current forward pass
        self.latch: Dict[str, torch.Tensor] = {}  # adapter id -> bool [B] (incremental mode)
        self.mode: str = "full"
        self.valid: Optional[torch.Tensor] = None  # bool [B, T]: positions that count (no padding / finished)
        self.disabled: set = set()                 # adapter ids skipped entirely (reference policy)
        self.recording: bool = False
        self.rec: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        # positions 0..gate_from-1 are the chat-template prefix shared by every prompt; the likelihood ratio is
        # exactly 1 there, so tripwires never look at them (callers fold this into `valid`)
        self.gate_from: int = 0
        # router baseline (evaluation only): gates are read once, at the last prompt token, and latch for the
        # whole response; no other position is a gate position
        self.last_prompt_only: bool = False
        # prompt-time gates only: a fired decision switches the adapter on for the whole sequence, prompt included
        self.whole_sequence: bool = False
        # full-rank adapters (v7): whether any position of the current forward pass fired, per tripwire, so that a
        # dense update nobody would receive is not computed (one host sync per forward pass, only when tracked)
        self.track_active: bool = False
        self.active: Dict[str, bool] = {}

    def begin(self, mode: str, record: bool = False) -> None:
        if mode not in ("full", "incremental"):
            raise ValueError(mode)
        self.mode = mode
        self.masks, self.latch, self.valid, self.active = {}, {}, None, {}
        self.recording = record
        self.rec = {t: [] for t in list(self.tripwires) + list(self.shadow)} if record else {}

    def end(self) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        rec = self.rec
        self.masks, self.latch, self.valid, self.recording, self.rec, self.active = {}, {}, None, False, {}, {}
        return rec

    def has_scoring(self) -> bool:
        return bool(self.tripwires) or bool(self.shadow)


def make_tripwire_hook(state: GateState):
    def hook(module, args, kwargs):
        if not state.has_scoring():
            return None
        h = args[0] if len(args) > 0 else kwargs["hidden_states"]
        B, T = h.shape[0], h.shape[1]
        valid = state.valid
        if valid is None:
            raise RuntimeError("GateState.valid must be set before a forward pass when tripwires exist")
        if tuple(valid.shape) != (B, T):
            raise RuntimeError(f"valid mask {tuple(valid.shape)} does not match hidden states {(B, T)}")
        hd = h.detach()
        for tid, tw in list(state.tripwires.items()) + list(state.shadow.items()):
            s = tw.score(hd)  # [B, T] float32
            fire = (s > tw.kappa) & valid
            if state.whole_sequence and state.last_prompt_only:
                # the only gate position of a sequence is its last prompt token: its decision holds for every
                # position of the sequence (prompt included) and, when decoding, for every later step
                on = fire.any(dim=1)
                if state.mode == "incremental":
                    prev = state.latch.get(tid)
                    if prev is not None:
                        on = on | prev
                    state.latch[tid] = on.clone()
                m = on[:, None].expand(B, T)
            else:
                fired_so_far = fire.long().cumsum(dim=1) > 0
                if state.mode == "incremental":
                    prev = state.latch.get(tid)
                    if prev is None:
                        prev = torch.zeros(B, dtype=torch.bool, device=h.device)
                    m = fired_so_far | prev[:, None]
                    state.latch[tid] = m[:, -1].clone()
                else:
                    m = fired_so_far
            if tid in state.tripwires:
                state.masks[tid] = m
                if state.track_active:
                    state.active[tid] = bool(m.any())
            if state.recording:
                state.rec[tid].append((s, valid))
        return None

    return hook


class GatedLoRALinear(nn.Module):
    """y = W x + sum_k [m_k] * (alpha/r) B_k A_k x, with float32 adapter weights."""

    def __init__(self, base: nn.Linear, state: GateState):
        super().__init__()
        self.base = base
        self.state = state
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scale: Dict[str, float] = {}
        # null-space baseline: adapter tid sees only the part of its input orthogonal to span(U) (U: [in, k])
        self.proj: Dict[str, torch.Tensor] = {}
        self.dense: set = set()   # adapter ids with a full-rank update D (stored in lora_A, no lora_B)
        self._dcache: Dict[str, tuple] = {}   # tid -> (version of D, dtype, D in the model's dtype), no-grad passes

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self):  # some code paths read .weight of projections
        return self.base.weight

    def add_adapter(self, tid: str, r: int, alpha: float, dense: bool = False) -> List[nn.Parameter]:
        dev = self.base.weight.device
        if dense:   # full-rank update, zero at the start (the model is unchanged until training moves it)
            D = nn.Parameter(torch.zeros(self.out_features, self.in_features, dtype=torch.float32, device=dev),
                             requires_grad=False)
            self.lora_A[tid] = D
            self.scale[tid] = 1.0
            self.dense.add(tid)
            self.state.track_active = True
            return [D]
        A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32, device=dev), requires_grad=False)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        Bm = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32, device=dev), requires_grad=False)
        self.lora_A[tid] = A
        self.lora_B[tid] = Bm
        self.scale[tid] = float(alpha) / float(r)
        return [A, Bm]

    def _dense_weight(self, tid: str, dtype) -> torch.Tensor:
        """the dense update (projected when a null-space projection is set) in the model's dtype: differentiable when
        gradients are on, else a copy cached until D or its projection changes"""
        D, U = self.lora_A[tid], self.proj.get(tid)
        if torch.is_grad_enabled() and D.requires_grad:
            return (D if U is None else D - (D @ U) @ U.t()).to(dtype)
        uv = None if U is None else U._version
        c = self._dcache.get(tid)   # (D version, dtype, the projection object itself, its version, weight)
        if c is None or c[0] != D._version or c[1] != dtype or c[2] is not U or c[3] != uv:
            with torch.no_grad():
                W = D if U is None else D - (D @ U) @ U.t()
                c = (D._version, dtype, U, uv, W.to(dtype))
            self._dcache[tid] = c
        return c[4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for tid in self.lora_A.keys():
            if tid in self.state.disabled:
                continue
            if tid in self.dense:
                if (tid in self.state.tripwires and not self.state.active.get(tid, True)
                        and not (torch.is_grad_enabled() and self.lora_A[tid].requires_grad)):
                    continue   # no position fired: the masked update below would leave y unchanged anyway (not
                    #            skipped when training, so that the loss always depends on D: zero gradient, no error)
                d = x @ self._dense_weight(tid, x.dtype).t()
                d = (d * self.scale[tid]).to(y.dtype)
            else:
                A, Bm = self.lora_A[tid], self.lora_B[tid]
                xa = x.to(A.dtype)
                U = self.proj.get(tid)
                if U is not None:  # A (x - U U^T x) = (A - A U U^T) x: project A once instead of every token
                    A = A - (A @ U) @ U.t()
                d = (xa @ A.t()) @ Bm.t()
                d = (d * self.scale[tid]).to(y.dtype)
            if tid in self.state.tripwires:
                m = self.state.masks.get(tid)
                if m is None or tuple(m.shape) != tuple(y.shape[:2]):
                    raise RuntimeError(f"no mask of shape {tuple(y.shape[:2])} for gated adapter {tid}")
                y = torch.where(m.unsqueeze(-1), y + d, y)
            else:
                y = y + d
        return y


def decoder_layers(model) -> nn.ModuleList:
    return model.model.layers


def install(model, state: GateState, gate_layer: int, targets: List[str]) -> List[GatedLoRALinear]:
    """Wrap target projections of blocks >= gate_layer and register the tripwire hook on block gate_layer."""
    layers = decoder_layers(model)
    if not (0 < gate_layer < len(layers)):
        raise ValueError(f"gate_layer must be in 1..{len(layers) - 1}")
    wrappers = []
    for i in range(gate_layer, len(layers)):
        layer = layers[i]
        for name, mod in list(layer.named_modules()):
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear) and not isinstance(mod, GatedLoRALinear):
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                w = GatedLoRALinear(mod, state)
                setattr(parent, leaf, w)
                wrappers.append(w)
    if not wrappers:
        raise RuntimeError(f"no target modules {targets} found in blocks >= {gate_layer}")
    handle = layers[gate_layer].register_forward_pre_hook(make_tripwire_hook(state), with_kwargs=True)
    model._tripwire_hook_handle = handle
    return wrappers


def wrappers_of(model) -> List[Tuple[str, GatedLoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, GatedLoRALinear)]


def add_adapter(model, tid: str, r: int, alpha: float, dense: bool = False) -> List[nn.Parameter]:
    """a LoRA adapter (rank r) or, with dense=True, a full-rank update of every wrapped projection"""
    params = []
    for _, w in wrappers_of(model):
        if tid in w.lora_A:
            raise RuntimeError(f"adapter {tid} exists already")
        params += w.add_adapter(tid, r, alpha, dense)
    return params


def is_dense(model, tid: str) -> bool:
    return any(tid in w.dense for _, w in wrappers_of(model))


def adapter_params(model, tid: str) -> List[nn.Parameter]:
    ps = []
    for _, w in wrappers_of(model):
        ps += [w.lora_A[tid]] + ([w.lora_B[tid]] if tid in w.lora_B else [])
    return ps


def adapter_state_dict(model, tid: str) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, w in wrappers_of(model):
        sd[f"{name}.A"] = w.lora_A[tid].detach().cpu().clone()
        if tid in w.lora_B:
            sd[f"{name}.B"] = w.lora_B[tid].detach().cpu().clone()
        if tid in w.proj:
            sd[f"{name}.U"] = w.proj[tid].detach().cpu().clone()
    return sd


def load_adapter_state_dict(model, tid: str, sd: Dict[str, torch.Tensor]) -> None:
    for name, w in wrappers_of(model):
        with torch.no_grad():
            w.lora_A[tid].copy_(sd[f"{name}.A"].to(w.lora_A[tid].device))
            if tid in w.lora_B:
                w.lora_B[tid].copy_(sd[f"{name}.B"].to(w.lora_B[tid].device))
        if f"{name}.U" in sd:
            w.proj[tid] = sd[f"{name}.U"].to(device=w.lora_A[tid].device, dtype=torch.float32)


def set_projections(model, tid: str, subspaces: Dict[str, torch.Tensor]) -> None:
    """Null-space baseline: adapter tid of each wrapped projection ignores the input directions in subspaces[name]."""
    for name, w in wrappers_of(model):
        if name in subspaces and subspaces[name].numel():
            w.proj[tid] = subspaces[name].to(device=w.lora_A[tid].device, dtype=torch.float32)


def scale_adapter(model, tid: str, factor: float) -> None:
    """Multiply adapter tid's update by `factor` (adapter-scaling baseline; evaluation only)."""
    for _, w in wrappers_of(model):
        w.scale[tid] *= float(factor)


def tripwire_from_state(sd: Dict[str, torch.Tensor], device) -> Tripwire:
    tw = Tripwire(sd["mu"].numel())
    tw.load_state_dict(sd)
    return tw.to(device)
