import re
import random
from draughts import Board, BaseAgent


class LLMAgent(BaseAgent):
    """Wraps any text-generation callable as a checkers BaseAgent.

    Parameters
    ----------
    generate_fn : callable(prompt: str) -> str
        A function that takes a text prompt and returns the model's response.
        This keeps the agent model-agnostic — plug in HuggingFace pipeline,
        OpenAI API, or anything else.
    max_retries : int
        How many times to re-prompt the model on an illegal/unparseable move
        before falling back to a random legal move.
    """

    def __init__(self, generate_fn, max_retries: int = 3):
        self.generate_fn = generate_fn
        self.max_retries = max_retries

        # Tracking stats for legal-move-rate metric
        self.total_attempts = 0
        self.first_try_legal = 0
        self.total_fallbacks = 0

    def select_move(self, board: Board):
        legal_moves = board.legal_moves
        legal_strs = [str(m) for m in legal_moves]
        prompt = self._build_prompt(board, legal_strs)

        for attempt in range(self.max_retries):
            self.total_attempts += 1
            response = self.generate_fn(prompt)
            move = self._parse_response(response, legal_moves)

            if move is not None:
                if attempt == 0:
                    self.first_try_legal += 1
                return move

            # Re-prompt with feedback about the failed attempt
            prompt = self._build_retry_prompt(board, legal_strs, response)

        # All retries exhausted — fall back to random legal move
        self.total_fallbacks += 1
        return random.choice(legal_moves)

    def legal_move_rate(self) -> float:
        """Fraction of turns where the first attempt was a legal move."""
        if self.total_attempts == 0:
            return 0.0
        return self.first_try_legal / (self.first_try_legal + (self.total_attempts - self.first_try_legal) // max(1, self.max_retries))

    def reset_stats(self):
        self.total_attempts = 0
        self.first_try_legal = 0
        self.total_fallbacks = 0

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def _build_prompt(self, board: Board, legal_strs: list[str]) -> str:
        return (
            "You are playing American checkers (8x8). "
            "You must reply with ONLY a move in the format shown below (e.g. '22-18' or '15x6').\n\n"
            f"Current board:\n{board}\n\n"
            f"FEN: {board.fen}\n\n"
            f"Legal moves: {', '.join(legal_strs)}\n\n"
            "Your move:"
        )

    def _build_retry_prompt(self, board: Board, legal_strs: list[str], bad_response: str) -> str:
        return (
            "You are playing American checkers (8x8). "
            f"Your previous response '{bad_response}' was not a valid move.\n\n"
            f"Current board:\n{board}\n\n"
            f"Legal moves: {', '.join(legal_strs)}\n\n"
            "Reply with ONLY one of the legal moves listed above (e.g. '22-18').\n\n"
            "Your move:"
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, legal_moves) -> object | None:
        """Try to extract a legal move from the model's text output."""
        response = response.strip()

        # Try to find a move pattern like "32-28" or "23x14" anywhere in the response
        pattern = r"\d+[\-x]\d+"
        matches = re.findall(pattern, response)

        legal_strs = {str(m): m for m in legal_moves}

        for match in matches:
            if match in legal_strs:
                return legal_strs[match]

        return None
