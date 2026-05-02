"""Minimax agent that works with any py-draughts board variant (including AmericanBoard).

The built-in AlphaBetaEngine only supports 10x10 standard boards, so we implement
a simple negamax with alpha-beta pruning here using the BaseAgent interface.
"""

import math
from draughts import BaseAgent, Color


class MinimaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning.

    Typical usage:
        shallow = MinimaxAgent(depth=3)   # weak baseline
        strong  = MinimaxAgent(depth=8)   # strong baseline
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
        """Material-based evaluation from the perspective of the current player."""
        feats = board.features()
        white_score = feats.white_men + 3 * feats.white_kings
        black_score = feats.black_men + 3 * feats.black_kings

        if board.turn == Color.WHITE:
            return white_score - black_score
        else:
            return black_score - white_score
