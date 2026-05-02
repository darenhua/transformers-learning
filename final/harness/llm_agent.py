"""LLMAgent — model-swap point.

Differs from models/llm_agent.py in two ways:
  1. Prompt format is injected via prompt_fn / retry_prompt_fn (so SFT-minimal
     and verbose-API variants share the same agent class).
  2. legal_move_rate is computed correctly (per-turn, not per-attempt).

Captures per-turn introspection records so the test harness can render them.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from draughts import BaseAgent, Board

from harness.scoring import parse_move
from harness.prompts import (
    build_minimal_prompt,
    build_minimal_retry_prompt,
    build_verbose_prompt,
    build_verbose_retry_prompt,
)


@dataclass
class TurnRecord:
    fen_before: str
    prompts: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    parsed: Optional[str] = None
    legal_first_try: bool = False
    fell_back_to_random: bool = False
    chosen_move: Optional[str] = None


class LLMAgent(BaseAgent):
    """Wraps any text-generation callable as a checkers BaseAgent.

    Parameters
    ----------
    generate_fn : callable(prompt: str) -> str
    prompt_fn : callable(board) -> str OR callable(board, legal_strs) -> str
        Builds the initial prompt. Defaults to the minimal SFT format.
    retry_prompt_fn : callable(board, bad_response) or callable(board, legal_strs, bad_response)
        Builds the retry prompt. Defaults to the minimal retry.
    max_retries : int
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        prompt_fn: Optional[Callable] = None,
        retry_prompt_fn: Optional[Callable] = None,
        max_retries: int = 3,
        record_turns: bool = False,
    ):
        self.generate_fn = generate_fn
        self.prompt_fn = prompt_fn or build_minimal_prompt
        self.retry_prompt_fn = retry_prompt_fn or build_minimal_retry_prompt
        self.max_retries = max_retries
        self.record_turns = record_turns

        # Per-turn introspection (only populated when record_turns=True)
        self.turns: list[TurnRecord] = []

        # Aggregate stats
        self.num_turns = 0
        self.first_try_legal = 0
        self.total_fallbacks = 0

    def select_move(self, board: Board):
        legal_moves = list(board.legal_moves)
        legal_strs = [str(m) for m in legal_moves]

        record = TurnRecord(fen_before=board.fen) if self.record_turns else None
        self.num_turns += 1

        prompt = self._build(self.prompt_fn, board, legal_strs)

        for attempt in range(self.max_retries):
            if record is not None:
                record.prompts.append(prompt)

            response = self.generate_fn(prompt)
            if record is not None:
                record.responses.append(response)

            move = parse_move(response, legal_moves)

            if move is not None:
                if attempt == 0:
                    self.first_try_legal += 1
                if record is not None:
                    record.parsed = str(move)
                    record.legal_first_try = (attempt == 0)
                    record.chosen_move = str(move)
                    self.turns.append(record)
                return move

            prompt = self._build_retry(self.retry_prompt_fn, board, legal_strs, response)

        # All retries exhausted — random fallback
        self.total_fallbacks += 1
        fallback = random.choice(legal_moves)
        if record is not None:
            record.fell_back_to_random = True
            record.chosen_move = str(fallback)
            self.turns.append(record)
        return fallback

    @staticmethod
    def _build(fn, board, legal_strs):
        # Support both (board) and (board, legal_strs) signatures
        try:
            return fn(board, legal_strs)
        except TypeError:
            return fn(board)

    @staticmethod
    def _build_retry(fn, board, legal_strs, bad_response):
        try:
            return fn(board, legal_strs, bad_response)
        except TypeError:
            return fn(board, bad_response)

    def legal_move_rate(self) -> float:
        """Fraction of turns where the first attempt was a legal move."""
        if self.num_turns == 0:
            return 0.0
        return self.first_try_legal / self.num_turns

    def fallback_rate(self) -> float:
        """Fraction of turns where all retries failed and we picked random."""
        if self.num_turns == 0:
            return 0.0
        return self.total_fallbacks / self.num_turns

    def reset_stats(self):
        self.num_turns = 0
        self.first_try_legal = 0
        self.total_fallbacks = 0
        self.turns = []


def make_verbose_llm_agent(generate_fn, max_retries: int = 3, record_turns: bool = False) -> LLMAgent:
    """Convenience: LLMAgent with the verbose API-style prompt."""
    return LLMAgent(
        generate_fn=generate_fn,
        prompt_fn=build_verbose_prompt,
        retry_prompt_fn=build_verbose_retry_prompt,
        max_retries=max_retries,
        record_turns=record_turns,
    )
