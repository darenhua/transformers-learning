"""Model registry — the dropdown's source of truth.

Each entry is a factory `() -> BaseAgent` so reactive cells can build a fresh
agent (with reset stats) on every run.

Baselines (random / greedy / minimax) are pre-populated from harness.agents.
LLM agents register themselves from model notebooks via `register(name, factory)`.
"""

from typing import Callable
from draughts import BaseAgent

from harness.agents import RandomAgent, GreedyAgent, MinimaxAgent


MODELS: dict[str, Callable[[], BaseAgent]] = {
    "random":     lambda: RandomAgent(),
    "greedy":     lambda: GreedyAgent(),
    "minimax_d3": lambda: MinimaxAgent(depth=3),
    "minimax_d8": lambda: MinimaxAgent(depth=8),
}


def register(name: str, factory: Callable[[], BaseAgent]) -> None:
    """Add or overwrite an entry. Idempotent — safe to call from a cell that
    re-runs reactively."""
    MODELS[name] = factory


def names() -> list[str]:
    """For dropdown options."""
    return list(MODELS.keys())


def build(name: str) -> BaseAgent:
    """Build a fresh agent by name."""
    return MODELS[name]()
