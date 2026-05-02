"""Stateless baseline agents — consolidated into one module.

Lives here (not in three separate notebooks) because:
- Each is a tiny class with no UI and no expensive state.
- Marimo's module autoreloader reruns dependent cells on edit, so iterating
  on them feels notebook-native without the @app.cell boilerplate.

LLMAgent has its own module (harness/llm_agent.py) because it's larger and
holds more state.

Adding a new baseline? Add the class here, then register a factory in
harness/registry.py. That's it.
"""

import math
import random

from draughts import BaseAgent, Color


class RandomAgent(BaseAgent):
    """Picks a uniformly random legal move."""

    def select_move(self, board):
        return random.choice(list(board.legal_moves))


class GreedyAgent(BaseAgent):
    """Maximizes immediate capture chain length. Cheap, occasionally clever
    in tactical positions, blind everywhere else."""

    def select_move(self, board):
        return max(board.legal_moves, key=lambda m: len(m.captured_list))


class MinimaxAgent(BaseAgent):
    """Negamax with alpha-beta pruning. Material eval: men=1, kings=3.

    The built-in py-draughts AlphaBetaEngine only supports 10x10 boards, so
    this exists to give an 8x8 (American checkers) baseline.

    Typical depths:
        depth=3 — weak baseline, fast
        depth=8 — strong baseline, several seconds per move
    """

    def __init__(self, depth: int = 5):
        self.depth = depth

    def select_move(self, board):
        _, best_move = self._negamax(board, self.depth, -math.inf, math.inf)
        return best_move

    def _negamax(self, board, depth, alpha, beta):
        if depth == 0 or board.game_over:
            return self._evaluate(board), None

        best_score = -math.inf
        best_move = None

        for move in board.legal_moves:
            child = board.copy()
            child.push(move)
            score, _ = self._negamax(child, depth - 1, -beta, -alpha)
            score = -score

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_move

    def _evaluate(self, board):
        feats = board.features()
        white_score = feats.white_men + 3 * feats.white_kings
        black_score = feats.black_men + 3 * feats.black_kings
        if board.turn == Color.WHITE:
            return white_score - black_score
        return black_score - white_score
