"""Code task (Nov 1, PLAN_NOV.md; no torch): write a Python function, rewarded when the problem's unit tests pass.

Examples (data/code_{train,test}.jsonl, written by scripts/prepare_data.py --only-code) have the fields
  id, question (the problem statement), test (pytest-style test code that imports from `solution`),
  entry (the function the tests call; may be empty), signature (its signature line; may be empty).

A response's last ```python block (else its last ``` block, else the whole response if it defines a function) is
saved as solution.py next to the test file in a fresh temporary folder, and the tests run in a separate Python process
(python -I: no user site, no PYTHONPATH) with a wall-clock limit, CPU, memory and file-size limits, stdin closed and
file-system / process functions disabled before the tests import the solution (as in the HumanEval harness, Chen et
al. 2021). This keeps accidents of generated code from touching anything outside the folder; it is not a security
sandbox against deliberately hostile code. pytest itself is not needed: a small stand-in provides pytest.raises,
pytest.approx, pytest.mark.parametrize and module-level fixtures, and the runner calls every test function.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .tasks import POOLS, Pool

TIMEOUT_S = float(os.environ.get("TW_CODE_TIMEOUT", "6"))     # wall clock per response (all its tests)
MEM_BYTES = int(os.environ.get("TW_CODE_MEM_MB", "2048")) * 2 ** 20
WORKERS = int(os.environ.get("TW_CODE_WORKERS", str(max(1, min(16, (os.cpu_count() or 2) - 1)))))

_BLOCK = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)
_PY = re.compile(r"```[ \t]*(?:python|py|python3)[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> Optional[str]:
    """The last ```python block, else the last ``` block, else the whole text if it defines a function."""
    m = _PY.findall(text)
    if m:
        return m[-1]
    m = _BLOCK.findall(text)
    if m:
        return m[-1]
    unclosed = re.search(r"```[ \t]*(?:python|py|python3)[ \t]*\n(.*)$", text, re.S | re.I)   # cut at the length limit
    if unclosed and "def " in unclosed.group(1):
        return unclosed.group(1)
    return text if re.search(r"^\s*def \w+\(", text, re.M) else None


def code_prompt(ex: dict) -> str:
    q = ex["question"].strip()
    sig = (ex.get("signature") or "").strip()
    entry = (ex.get("entry") or "").strip()
    tail = "\n\nWrite a Python solution and put the complete code in a single ```python code block."
    if sig:
        tail += f" Use this function signature: `{sig}`."
    elif entry:
        tail += f" The function must be named `{entry}`."
    return q + tail


RUNNER = r'''
import builtins, importlib.util, inspect, itertools, math, os, re, resource, sys, types
d, mem, cpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for _lim, _v in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu), (resource.RLIMIT_FSIZE, 16 * 2 ** 20),
                 (resource.RLIMIT_CORE, 0)):
    try:
        resource.setrlimit(_lim, (_v, _v))
    except (ValueError, OSError):
        pass
sys.path.insert(0, d)
os.chdir(d)


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match, self.value = exc, match, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("DID NOT RAISE")
        if not issubclass(et, self.exc):
            return False
        if self.match is not None and not re.search(self.match, str(ev)):
            raise AssertionError("pattern not found")
        self.value = ev
        return True


def raises(exc, *args, match=None, **kw):
    if args and callable(args[0]):
        with _Raises(exc, match):
            args[0](*args[1:], **kw)
        return None
    return _Raises(exc, match)


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.e, self.rel, self.abs = expected, 1e-6 if rel is None else rel, 1e-12 if abs is None else abs

    def _eq(self, a, e):
        if isinstance(e, dict):
            return isinstance(a, dict) and a.keys() == e.keys() and all(self._eq(a[k], e[k]) for k in e)
        if isinstance(e, (list, tuple)):
            return isinstance(a, (list, tuple)) and len(a) == len(e) and all(self._eq(x, y) for x, y in zip(a, e))
        try:
            if isinstance(e, float) and math.isnan(e):
                return isinstance(a, float) and math.isnan(a)
            return abs(a - e) <= max(self.rel * abs(e), self.abs)
        except TypeError:
            return a == e

    def __eq__(self, other):
        return self._eq(other, self.e)

    __hash__ = None


class _Mark:
    def parametrize(self, names, values, ids=None, **kw):
        def deco(f):
            f.__dict__.setdefault("_tw_params", []).append((names, list(values)))
            return f
        return deco

    def __getattr__(self, name):   # skip, slow, timeout, ...: no effect
        def deco(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda f: f
        return deco


def fixture(*a, **k):
    def mark(f):
        f._tw_fixture = True
        return f
    if len(a) == 1 and callable(a[0]):
        return mark(a[0])
    return mark


pt = types.ModuleType("pytest")
pt.raises, pt.approx, pt.mark, pt.fixture = raises, _Approx, _Mark(), fixture
sys.modules["pytest"] = pt

# file-system and process functions off before the solution is imported (cf. the HumanEval harness)
import shutil, subprocess
for _n in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
           "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
           "chroot", "lchown", "chdir", "execv", "execve", "execl", "execvp", "spawnv", "popen"):
    if hasattr(os, _n):
        setattr(os, _n, None)
shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = subprocess.run = subprocess.call = subprocess.check_output = subprocess.check_call = None
builtins.exit = builtins.quit = None
for _m in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[_m] = None

spec = importlib.util.spec_from_file_location("test_solution", os.path.join(d, "test_solution.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fixtures = {k: v for k, v in vars(mod).items() if callable(v) and getattr(v, "_tw_fixture", False)}


def call(f, given):
    kw = dict(given)
    for p in inspect.signature(f).parameters.values():
        if p.name in kw or p.default is not inspect.Parameter.empty:
            continue
        if p.name not in fixtures:
            raise RuntimeError("unsupported fixture " + p.name)
        kw[p.name] = call(fixtures[p.name], {})
    return f(**kw)


n = 0
for name, f in list(vars(mod).items()):
    if not (name.startswith("test") and inspect.isfunction(f)) or getattr(f, "_tw_fixture", False):
        continue
    sets = []
    for names, values in reversed(getattr(f, "_tw_params", [])):
        ns = [x.strip() for x in names.split(",")] if isinstance(names, str) else list(names)
        sets.append([dict(zip(ns, v if len(ns) > 1 else (v,))) for v in values])
    for combo in itertools.product(*sets) if sets else [()]:
        given = {}
        for c in combo:
            given.update(c)
        call(f, given)
        n += 1
sys.exit(0 if n > 0 else 3)
'''


def run_tests(code: str, test: str, timeout: float = TIMEOUT_S) -> bool:
    """True if every test function of `test` passes against `code` (saved as solution.py) within the limits."""
    d = tempfile.mkdtemp(prefix="twcode_")
    try:
        for name, text in (("solution.py", code), ("test_solution.py", test), ("_tw_runner.py", RUNNER)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "HOME": d, "TMPDIR": d}
        # the limits are set by the runner itself (no preexec_fn: it is unsafe with the threads of code_score_many)
        p = subprocess.Popen([sys.executable, "-I", "-B", os.path.join(d, "_tw_runner.py"), d, str(MEM_BYTES),
                              str(int(timeout) + 1)], cwd=d, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            return p.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 9)
            except OSError:
                pass
            p.wait()
            return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def code_score(ex: dict, text: str) -> Dict[str, int]:
    code = extract_code(text)
    if code is None:
        return {"correct": 0, "format": 0}
    return {"correct": int(run_tests(code, ex["test"])), "format": 1}


def code_score_many(exs: List[dict], texts: List[str]) -> List[Dict[str, int]]:
    """code_score of many responses, their tests run in parallel processes (same results, in order)."""
    if len(texts) <= 1 or WORKERS <= 1:
        return [code_score(e, t) for e, t in zip(exs, texts)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(code_score, exs, texts))


POOLS["code"] = Pool("code", code_prompt, code_score, is_task=True, score_many=code_score_many)
