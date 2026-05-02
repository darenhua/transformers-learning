import random
from draughts import BaseAgent


class RandomAgent(BaseAgent):
    """Baseline agent that picks a random legal move each turn."""

    def select_move(self, board):
        return random.choice(board.legal_moves)
