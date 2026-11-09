"""Optional multi-GPU support on one node.

Launch a script with several processes, one per GPU:
    deepspeed --master_port 29511 scripts/run_stream.py ...          (uses the GPUs in CUDA_VISIBLE_DEVICES)
    torchrun --nproc_per_node 4 --master_port 29511 scripts/run_stream.py ...
With one process (plain `python ...`) every helper below is a no-op, so single-GPU behaviour is unchanged.

Design: every rank holds the full frozen model (LoRA adapters are tiny, so ZeRO sharding would buy nothing). Work is
split by fixed batches: generation batch b goes to rank b % world, and results are gathered so that every rank holds
the same rollouts, scores and files. Gradients of the trainable adapter are summed across ranks, so all ranks take
the same optimizer step. Only rank 0 writes files and logs; every rank reads them after a barrier. Decisions that
depend on the file system (does a result exist? was a stop requested?) are reduced across ranks so that all ranks
always take the same branch and call the same collectives.
"""
from __future__ import annotations

import os
from typing import Any, List, Sequence

import torch

from .io_utils import StopRequested, stop_requested

_STATE = {"init": False}


def init() -> None:
    """Initialise torch.distributed if the script was started by a multi-process launcher."""
    if _STATE["init"]:
        return
    _STATE["init"] = True
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        import deepspeed  # noqa: F401  (only its process-group setup is used)
        deepspeed.init_distributed(dist_backend=backend)
    except ImportError:
        torch.distributed.init_process_group(backend)


def active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized() and \
        torch.distributed.get_world_size() > 1


def world() -> int:
    return torch.distributed.get_world_size() if active() else 1


def rank() -> int:
    return torch.distributed.get_rank() if active() else 0


def is_main() -> bool:
    return rank() == 0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")


def barrier() -> None:
    if active():
        torch.distributed.barrier()


def any_true(flag: bool) -> bool:
    """True on every rank if it is true on at least one rank."""
    if not active():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return bool(t.item())


def all_true(flag: bool) -> bool:
    return not any_true(not flag)


def check_stop(stop_file) -> None:
    """Raise StopRequested on every rank as soon as any rank sees the stop file."""
    if any_true(stop_requested(stop_file)):
        raise StopRequested(stop_file)


def exists(path: str) -> bool:
    """File-system test that every rank agrees on (rank 0 decides)."""
    return bool(broadcast(os.path.exists(path)))


def broadcast(obj: Any) -> Any:
    if not active():
        return obj
    box = [obj if is_main() else None]
    torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def all_gather(obj: Any) -> List[Any]:
    if not active():
        return [obj]
    out: List[Any] = [None] * world()
    torch.distributed.all_gather_object(out, obj)
    return out


def sum_scalars(values: Sequence[float]) -> List[float]:
    if not active():
        return [float(v) for v in values]
    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=device())
    torch.distributed.all_reduce(t)
    return t.tolist()


def all_reduce_grads(params: Sequence[torch.nn.Parameter]) -> None:
    """Sum the gradients of `params` across ranks. A parameter that has no gradient on any rank keeps grad None
    (as in a single-GPU run, so AdamW skips it); one that has a gradient on some rank gets the sum."""
    if not active():
        return
    has = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=device())
    torch.distributed.all_reduce(has, op=torch.distributed.ReduceOp.MAX)
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    flat = torch._utils._flatten_dense_tensors(grads)
    torch.distributed.all_reduce(flat)
    for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(s)
    for p, h in zip(params, has.tolist()):
        if not h:
            p.grad = None


def broadcast_params(params: Sequence[torch.nn.Parameter]) -> None:
    """Make every rank hold rank 0's values (e.g. a freshly initialized adapter)."""
    if not active():
        return
    for p in params:
        torch.distributed.broadcast(p.data, src=0)


def same_on_all_ranks(value: str) -> bool:
    return all(v == value for v in all_gather(value))


def describe() -> str:
    """One line per rank: rank, CUDA device index and name (logged at start-up to catch device mapping errors)."""
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        me = f"rank {rank()}: cuda:{i} {p.name} uuid={getattr(p, 'uuid', '?')}"
    else:
        me = f"rank {rank()}: cpu"
    return " | ".join(all_gather(me))


def finish() -> None:
    if active():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def shutdown() -> None:
    """Close the process group without a barrier (all ranks are exiting on the same error or stop)."""
    if active():
        torch.distributed.destroy_process_group()
