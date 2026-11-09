"""4x4 Sudoku puzzles with a unique solution (no torch). Used as the fifth task of the longer stream.

A full grid is a random symmetry of one base grid (digit relabelling, row and column swaps inside bands and stacks,
band and stack swaps, transpose); cells are then removed in random order while the solution stays unique, until
the puzzle has the requested number of blanks.
"""
from __future__ import annotations

import random
from typing import List, Optional

BASE = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]


def is_valid(grid: List[List[int]]) -> bool:
    want = {1, 2, 3, 4}
    rows = all(set(r) == want for r in grid)
    cols = all({grid[r][c] for r in range(4)} == want for c in range(4))
    boxes = all({grid[r][c] for r in (br, br + 1) for c in (bc, bc + 1)} == want for br in (0, 2) for bc in (0, 2))
    return rows and cols and boxes


def count_solutions(puzzle: List[List[int]], limit: int = 2) -> int:
    g = [r[:] for r in puzzle]
    cells = [(r, c) for r in range(4) for c in range(4) if g[r][c] == 0]

    def ok(r, c, v):
        if v in g[r] or any(g[i][c] == v for i in range(4)):
            return False
        br, bc = r - r % 2, c - c % 2
        return all(g[i][j] != v for i in (br, br + 1) for j in (bc, bc + 1))

    def rec(k):
        if k == len(cells):
            return 1
        r, c = cells[k]
        n = 0
        for v in (1, 2, 3, 4):
            if ok(r, c, v):
                g[r][c] = v
                n += rec(k + 1)
                g[r][c] = 0
                if n >= limit:
                    break
        return n

    return rec(0)


def random_grid(rng: random.Random) -> List[List[int]]:
    perm = [0] + rng.sample([1, 2, 3, 4], 4)
    g = [[perm[v] for v in row] for row in BASE]
    bands = rng.sample([0, 1], 2)
    rows = [b * 2 + i for b in bands for i in rng.sample([0, 1], 2)]
    g = [g[r] for r in rows]
    stacks = rng.sample([0, 1], 2)
    cols = [s * 2 + j for s in stacks for j in rng.sample([0, 1], 2)]
    g = [[row[c] for c in cols] for row in g]
    if rng.random() < 0.5:
        g = [list(r) for r in zip(*g)]
    assert is_valid(g)
    return g


def make_puzzle(rng: random.Random, blanks: int) -> Optional[dict]:
    sol = random_grid(rng)
    puz = [r[:] for r in sol]
    removed = 0
    for idx in rng.sample(range(16), 16):
        if removed == blanks:
            break
        r, c = divmod(idx, 4)
        v, puz[r][c] = puz[r][c], 0
        if count_solutions(puz) == 1:
            removed += 1
        else:
            puz[r][c] = v
    if removed != blanks:
        return None
    return {"puzzle": "".join(str(v) for r in puz for v in r), "solution": "".join(str(v) for r in sol for v in r),
            "blanks": blanks}


def make_instances(n: int, seed: int, min_blanks: int = 6, max_blanks: int = 11) -> List[dict]:
    """n distinct puzzles, blanks spread evenly over [min_blanks, max_blanks]."""
    rng = random.Random(seed)
    out, seen = [], set()
    levels = list(range(min_blanks, max_blanks + 1))
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 200 * n:
            raise RuntimeError(f"could not make {n} distinct puzzles")
        p = make_puzzle(rng, levels[len(out) % len(levels)])
        if p is None or p["puzzle"] in seen:
            continue
        seen.add(p["puzzle"])
        p["id"] = f"sudoku_{seed}_{len(out)}"
        out.append(p)
    return out
