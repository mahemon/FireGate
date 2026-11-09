"""Small I/O helpers: atomic writes, JSONL, logging, stop requests (no torch)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Iterable, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(path: str, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_default))


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _default(o):
    try:
        import numpy as np  # noqa
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    return str(o)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, default=_default) + "\n" for r in rows))


class Logger:
    """Prints to stdout and appends JSON lines to a log file."""

    def __init__(self, path: str | None = None, prefix: str = ""):
        self.path = path
        self.prefix = prefix
        self.t0 = time.time()
        if path:
            ensure_dir(os.path.dirname(os.path.abspath(path)))

    def __call__(self, msg: str, **kv):
        if os.environ.get("RANK", "0") != "0":  # multi-GPU runs: only rank 0 logs
            return
        el = time.time() - self.t0
        line = f"[{self.prefix}{el / 3600:6.2f}h] {msg}"
        if kv:
            line += " " + " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"t": round(el, 1), "msg": msg, **kv}, default=_default) + "\n")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


class StopRequested(Exception):
    """Raised when the job should save and exit (time limit or a stop file)."""


def stop_requested(stop_file: str | None) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


def check_stop(stop_file: str | None) -> None:
    if stop_requested(stop_file):
        raise StopRequested(stop_file)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
