"""The two extra tasks of the 5-task stream (no torch): GSM8K -> Countdown -> Knights & Knaves -> MATH -> Sudoku.

  math    competition problems (Hendrycks MATH, levels 1-4) whose answer is a number (integer, decimal or fraction),
          asked with the GSM8K instruction ("... put your final answer within \\boxed{}"). It overlaps an earlier
          task on purpose: its tripwire must stay below eps on stored GSM8K prompts, which share the instruction.
  sudoku  4x4 Sudoku with a unique solution (twcore/gen_sudoku.py), answered inside <answer> </answer> like
          Countdown and Knights & Knaves.

Data: scripts/prepare_stream5.py writes data/math_{train,test}.jsonl and data/sudoku_{train,test}.jsonl.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Optional

from .tasks import POOLS, Pool, gsm8k_prompt, last_answer_block, last_boxed

# ----------------------------------------------------------------------------- MATH
_TEXT = re.compile(r"\\(?:text|textbf|mbox|mathrm)\{[^{}]*\}")
_INT = re.compile(r"-?\d+")
_DEC = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)")
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_FRAC = re.compile(r"(-?)\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}")
_FRAC_SHORT = re.compile(r"(-?)\\[dt]?frac(\d)(\d)")
_SLASH = re.compile(r"(-?\d+)/(-?\d+)")


def math_number(s: Optional[str]) -> Optional[Fraction]:
    """A MATH answer as an exact number, or None if it is not a plain number (lists, intervals, radicals, pi,
    expressions, mixed numbers, ...). Units in \\text{}, dollar and percent signs and degree marks are dropped."""
    if s is None:
        return None
    t = _TEXT.sub("", s)
    for a in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\$", "$", "\\%", "%", "^\\circ", "^{\\circ}",
              "{}", " "):
        t = t.replace(a, "")
    t = t.strip().rstrip(".")
    while len(t) >= 2 and t[0] == "{" and t[-1] == "}":
        t = t[1:-1]
    if t.startswith("+"):
        t = t[1:]
    if _THOUSANDS.fullmatch(t):
        t = t.replace(",", "")
    try:
        if _DEC.fullmatch(t):
            return Fraction(t)
        m = _FRAC.fullmatch(t) or _FRAC_SHORT.fullmatch(t) or _SLASH.fullmatch(t)
        if m is None:
            return None
        if len(m.groups()) == 3:
            sign, a, b = m.groups()
        else:
            sign, (a, b) = "", m.groups()
        if int(b) == 0:
            return None
        v = Fraction(int(a), int(b))
        return -v if sign == "-" else v
    except (ValueError, ZeroDivisionError):
        return None


def math_gold(ex: dict) -> Fraction:
    return Fraction(ex["answer"])


def math_prompt(ex: dict) -> str:
    return gsm8k_prompt({"question": ex["problem"]})


def math_score(ex: dict, text: str) -> Dict[str, int]:
    box = last_boxed(text)
    pred = math_number(box)
    return {"correct": int(pred is not None and pred == math_gold(ex)), "format": int(box is not None)}


# ----------------------------------------------------------------------------- Sudoku
def sudoku_grid_text(puzzle: str) -> str:
    rows = []
    for r in range(4):
        cells = ["." if ch == "0" else ch for ch in puzzle[4 * r:4 * r + 4]]
        rows.append(f"{cells[0]} {cells[1]} | {cells[2]} {cells[3]}")
        if r == 1:
            rows.append("----+----")
    return "\n".join(rows)


def sudoku_prompt(ex: dict) -> str:
    return ("Solve this 4x4 Sudoku. Fill every empty cell (shown as .) with a digit from 1 to 4 so that each row, "
            "each column and each of the four 2x2 boxes contains 1, 2, 3 and 4 exactly once.\n\n"
            + sudoku_grid_text(ex["puzzle"]) +
            "\n\nThink step by step, then give the completed grid inside <answer> </answer> tags as four lines of "
            "four digits, one line per row, for example <answer>\n1234\n3412\n2143\n4321\n</answer>")


def sudoku_parse(ans: str) -> Optional[str]:
    """The 16 digits of a grid written as 4 lines (any separators), or as exactly 16 digits in a row."""
    digits = re.findall(r"[1-4]", ans)
    rows: List[str] = []
    for line in ans.splitlines():
        d = re.findall(r"\d", line)
        if len(d) >= 4:
            rows.append("".join(d[-4:]))      # "Row 1: 1 2 3 4" -> 1234
    if len(rows) == 4:
        return "".join(rows)
    if len(re.findall(r"\d", ans)) == 16 and len(digits) == 16:
        return "".join(digits)
    return None


def sudoku_score(ex: dict, text: str) -> Dict[str, int]:
    ans = last_answer_block(text)
    if ans is None:
        return {"correct": 0, "format": 0}
    grid = sudoku_parse(ans)
    return {"correct": int(grid == ex["solution"]), "format": 1}


POOLS["math"] = Pool("math", math_prompt, math_score, is_task=True)
POOLS["sudoku"] = Pool("sudoku", sudoku_prompt, sudoku_score, is_task=True)
