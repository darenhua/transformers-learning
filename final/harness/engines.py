"""py-draughts engine wrappers.

Thin pass-throughs — the work is done by py-draughts. We just suppress
Benchmark's stdout (it spams a reactive marimo cell) and unpack stats.
"""

import contextlib
import io
from dataclasses import dataclass
from typing import Optional

from draughts import AmericanBoard, Benchmark, BaseAgent


@dataclass
class BenchmarkResult:
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    elo_diff: float
    # If the agent was an LLMAgent, snapshot its stats too
    legal_move_rate: Optional[float] = None
    fallback_rate: Optional[float] = None


def to_engine(agent: BaseAgent):
    """Wrap a BaseAgent into the engine interface Benchmark consumes."""
    return agent.as_engine()


def run_benchmark(
    agent: BaseAgent,
    opponent: BaseAgent,
    games: int = 50,
    opponent_name: str = "opponent",
    board_class=AmericanBoard,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a head-to-head benchmark and return a structured result.

    Pass actual BaseAgent instances (not engines) so we can introspect LLMAgent
    stats after the run.
    """
    agent_eng = agent.as_engine()
    opp_eng = opponent.as_engine()

    if verbose:
        stats = Benchmark(agent_eng, opp_eng, board_class=board_class, games=games).run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = Benchmark(agent_eng, opp_eng, board_class=board_class, games=games).run()

    result = BenchmarkResult(
        opponent=opponent_name,
        games=games,
        wins=stats.e1_wins,
        losses=stats.e2_wins,
        draws=stats.draws,
        win_rate=stats.e1_win_rate,
        elo_diff=stats.elo_diff,
    )

    # Lazy import to avoid circular dep
    from harness.llm_agent import LLMAgent
    if isinstance(agent, LLMAgent):
        result.legal_move_rate = agent.legal_move_rate()
        result.fallback_rate = agent.fallback_rate()

    return result
