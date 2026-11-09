"""SQuAD 2.0 as a new task that overlaps the reference naturally (paper v3, PLAN_PAPER3.md; no torch).

A deployed assistant is updated to answer questions about a passage and to say so when the passage does not contain
the answer (a hallucination fix for retrieval-augmented answers). The prompt has the layout of a Dolly request with a
context passage (question, blank line, passage), so the new prompts look like the closed-QA and information-extraction
part of the Dolly reference, and like the BoolQ prompts among the task-like reference prompts: the overlap is part of
the task, not built in. The reward:
  answerable question     1 if a gold answer appears (normalized, word-bounded) in the first 300 characters and the
                          response does not abstain
  unanswerable question   1 if the response abstains (says the passage does not give the answer)
Data: scripts/prepare_data.py --only-squad2 writes data/squad2_{train,test}.jsonl (half answerable, contexts of at most
1500 characters, so the Dolly layout keeps the whole passage).
"""
from __future__ import annotations

import re
from typing import Dict

from .tasks import POOLS, Pool, _norm_answer, dolly_prompt

# phrases that say the passage does not give the answer (a refusal to help does not count)
_ABSTAIN = re.compile(
    r"\b(?:does not|doesn't|did not|didn't|do not|don't)\s+(?:explicitly\s+|directly\s+)?(?:say|mention|specify|state|"
    r"provide|contain|include|indicate|give|tell|discuss|address|describe|answer)\b"
    r"|\bnot\s+(?:explicitly\s+|directly\s+)?(?:mentioned|stated|specified|provided|given|indicated|addressed|"
    r"discussed|described|included|answered)\b"
    r"|\bno\s+(?:information|mention|details|data|indication)\s+(?:about|on|of|regarding|is|in|that)\b"
    r"|\b(?:not enough|insufficient)\s+(?:information|context|details)\b"
    r"|\b(?:cannot|can't|can not|unable to|impossible to|not possible to)\s+(?:be\s+)?(?:answer|answered|determine|"
    r"determined|tell|told|find|found|infer|inferred|know|known)\b"
    r"|\bunanswerable\b|\bno answer\b"
    r"|\b(?:passage|text|context|article|paragraph)\s+(?:is silent|says nothing|has no)\b")


def abstains(text: str) -> bool:
    """True if the response says the passage does not give the answer (first 400 characters)."""
    return bool(_ABSTAIN.search(text[:400].lower().replace("’", "'")))


def squad2_score(ex: dict, text: str) -> Dict[str, int]:
    ab = abstains(text)
    if ex["answerable"]:
        head = " " + _norm_answer(text[:300]) + " "
        hit = any(n and (" " + n + " ") in head for n in (_norm_answer(a) for a in ex["answers"]))
        ok = hit and not ab
    else:
        ok = ab
    return {"correct": int(ok), "abstain": int(ab), "answerable": int(bool(ex["answerable"]))}


POOLS["squad2"] = Pool("squad2", dolly_prompt, squad2_score, is_task=True)
