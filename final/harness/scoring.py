"""Move validity scorer — pass/fail signal for benchmark, test harness, gameplay.

Lifted out of LLMAgent._parse_response so all three harnesses share one
implementation. Pure functions, no agent state.
"""

import re
from typing import Optional

MOVE_PATTERN = re.compile(r"\d+[\-x]\d+")


def parse_move(response: str, legal_moves) -> Optional[object]:
    """Extract a legal move from text. Returns the move object or None.

    Scans the response for any token matching `\\d+[\\-x]\\d+` and returns
    the first one that exactly matches a legal move's str() form.
    """
    legal_strs = {str(m): m for m in legal_moves}
    for match in MOVE_PATTERN.findall(response.strip()):
        if match in legal_strs:
            return legal_strs[match]
    return None


def is_legal(response: str, legal_moves) -> bool:
    return parse_move(response, legal_moves) is not None


def score_response(response: str, legal_moves) -> dict:
    """Per-turn scoring record for the test harness."""
    move = parse_move(response, legal_moves)
    return {
        "raw": response,
        "parsed": str(move) if move is not None else None,
        "legal": move is not None,
    }
