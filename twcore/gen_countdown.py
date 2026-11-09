"""Procedural Countdown instances (as in TinyZero): 3-4 numbers in [1, 99], an integer target in [1, 999]
reachable with +, -, *, / using every number exactly once with non-negative integer intermediate values."""
from __future__ import annotations

import random
from fractions import Fraction
from typing import List, Optional, Tuple

OPS = ["+", "-", "*", "/"]


def _combine(a: Fraction, b: Fraction, op: str) -> Optional[Fraction]:
    if op == "+":
        return a + b
    if op == "-":
        return a - b if a >= b else None
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a / b
    raise ValueError(op)


def random_target(nums: List[int], rng: random.Random) -> Optional[Tuple[int, str]]:
    """Combine all numbers in a random order and tree shape; return (target, expression) or None."""
    items = [(Fraction(n), str(n)) for n in nums]
    rng.shuffle(items)
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        (a, ea), (b, eb) = items[i], items[j]
        op = rng.choice(OPS)
        v = _combine(a, b, op)
        if v is None:
            return None
        rest = [it for t, it in enumerate(items) if t not in (i, j)]
        items = rest + [(v, f"({ea} {op} {eb})")]
    val, expr = items[0]
    if val.denominator != 1 or not (1 <= val <= 999):
        return None
    return int(val), expr


def make_instances(n: int, seed: int, start_id: int = 0) -> List[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        k = rng.choice([3, 4])
        nums = [rng.randint(1, 99) for _ in range(k)]
        res = random_target(nums, rng)
        if res is None:
            continue
        target, expr = res
        if target in nums:  # trivial instances
            continue
        key = (tuple(sorted(nums)), target)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"cd{start_id + len(out)}", "nums": nums, "target": target, "ref_expr": expr})
    return out
