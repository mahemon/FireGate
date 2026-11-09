"""Procedural Knights-and-Knaves puzzles with a unique solution (2-4 inhabitants).

Knights always tell the truth, knaves always lie. Each inhabitant makes one statement; a puzzle is kept only
if exactly one knight/knave assignment is consistent with all statements.
"""
from __future__ import annotations

import itertools
import random
from typing import List, Sequence, Tuple

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Isla", "Jack", "Kate",
         "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tara", "Uma", "Victor",
         "Wendy", "Zoe"]
ROLE = {True: "knight", False: "knave"}


def _truth(st: tuple, a: Sequence[bool]) -> bool:
    kind = st[0]
    if kind == "is":
        _, j, r = st
        return a[j] == r
    if kind in ("and", "or", "imp", "iff"):
        _, j, rj, k, rk = st
        p, q = a[j] == rj, a[k] == rk
        return {"and": p and q, "or": p or q, "imp": (not p) or q, "iff": p == q}[kind]
    if kind == "same":
        _, j, k = st
        return a[j] == a[k]
    if kind == "diff":
        _, j, k = st
        return a[j] != a[k]
    raise ValueError(kind)


def _render(st: tuple, speaker: int, names: Sequence[str]) -> str:
    def who(j, r):
        if j == speaker:
            return f"I am a {ROLE[r]}"
        return f"{names[j]} is a {ROLE[r]}"

    kind = st[0]
    if kind == "is":
        return who(st[1], st[2])
    if kind == "and":
        return f"{who(st[1], st[2])} and {who(st[3], st[4])}"
    if kind == "or":
        return f"{who(st[1], st[2])} or {who(st[3], st[4])}"
    if kind == "imp":
        return f"if {who(st[1], st[2])}, then {who(st[3], st[4])}"
    if kind == "iff":
        return f"{who(st[1], st[2])} if and only if {who(st[3], st[4])}"
    if kind == "same":
        a = "I" if st[1] == speaker else names[st[1]]
        b = "I" if st[2] == speaker else names[st[2]]
        if a == "I" or b == "I":
            other = b if a == "I" else a
            return f"{other} and I are the same type"
        return f"{a} and {b} are the same type"
    if kind == "diff":
        a = "I" if st[1] == speaker else names[st[1]]
        b = "I" if st[2] == speaker else names[st[2]]
        if a == "I" or b == "I":
            other = b if a == "I" else a
            return f"{other} and I are different types"
        return f"{a} and {b} are different types"
    raise ValueError(kind)


def _random_statement(rng: random.Random, n: int, speaker: int) -> tuple:
    kind = rng.choice(["is", "is", "and", "or", "imp", "iff", "same", "diff"])
    if kind == "is":
        j = rng.choice([x for x in range(n) if x != speaker] or [speaker])
        return ("is", j, rng.random() < 0.5)
    j, k = rng.sample(range(n), 2)
    if kind in ("same", "diff"):
        return (kind, j, k)
    return (kind, j, rng.random() < 0.5, k, rng.random() < 0.5)


def solve(statements: List[tuple], n: int) -> List[Tuple[bool, ...]]:
    sols = []
    for a in itertools.product([True, False], repeat=n):
        if all(a[i] == _truth(statements[i], a) for i in range(n)):
            sols.append(a)
    return sols


def make_instances(n_puzzles: int, seed: int, start_id: int = 0, sizes=(2, 3, 4), weights=(0.3, 0.45, 0.25)) -> List[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n_puzzles:
        n = rng.choices(list(sizes), weights=list(weights))[0]
        names = rng.sample(NAMES, n)
        sts = [_random_statement(rng, n, i) for i in range(n)]
        sols = solve(sts, n)
        if len(sols) != 1:
            continue
        key = (tuple(names), tuple(sts))
        if key in seen:
            continue
        seen.add(key)
        lines = [f'{names[i]} says: "{_render(sts[i], i, names)[0].upper() + _render(sts[i], i, names)[1:]}."'
                 for i in range(n)]
        out.append({"id": f"kk{start_id + len(out)}", "names": names, "solution": list(sols[0]),
                    "statements": lines})
    return out
