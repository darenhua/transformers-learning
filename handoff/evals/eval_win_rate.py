"""Primary Metric: Win rate + legal move rate.

Plays an LLM agent against baseline opponents using the py-draughts Benchmark
class (8x8 American checkers) and reports win rate, Elo difference, and the
LLM's legal-move rate.
"""

import io
import contextlib

from draughts import AmericanBoard, Benchmark

from models.random_bot import RandomAgent
from models.minimax_bot import MinimaxAgent


BASELINES = {
    "random":     lambda: RandomAgent().as_engine(),
    "minimax_d3": lambda: MinimaxAgent(depth=3).as_engine(),
    "minimax_d8": lambda: MinimaxAgent(depth=8).as_engine(),
}


def eval_win_rate(agent_engine, opponent_name: str = "random", num_games: int = 100, verbose: bool = False) -> dict:
    """Run a win-rate evaluation and return structured results."""
    opponent_engine = BASELINES[opponent_name]()

    if verbose:
        stats = Benchmark(agent_engine, opponent_engine, board_class=AmericanBoard, games=num_games).run()
        print(stats)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = Benchmark(agent_engine, opponent_engine, board_class=AmericanBoard, games=num_games).run()

    return {
        "opponent": opponent_name,
        "wins": stats.e1_wins,
        "losses": stats.e2_wins,
        "draws": stats.draws,
        "win_rate": stats.e1_win_rate,
        "elo_diff": stats.elo_diff,
        "games": num_games,
    }


def eval_all_baselines(agent_engine, num_games: int = 100, verbose: bool = False) -> list[dict]:
    """Run win-rate eval against every baseline. Returns list of result dicts."""
    results = []
    for name in BASELINES:
        result = eval_win_rate(agent_engine, opponent_name=name, num_games=num_games, verbose=verbose)
        results.append(result)
    return results


def print_summary(results: list[dict]):
    """Print a concise summary table."""
    print(f"\n{'Opponent':<14} {'W-L-D':>10} {'Win%':>7} {'Elo':>6}")
    print("-" * 40)
    for r in results:
        wld = f"{r['wins']}-{r['losses']}-{r['draws']}"
        print(f"{r['opponent']:<14} {wld:>10} {r['win_rate']:>6.1f}% {r['elo_diff']:>+6.0f}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an agent's win rate against baselines")
    parser.add_argument("-n", "--num-games", type=int, default=10, help="games per baseline (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full Benchmark output per matchup")
    args = parser.parse_args()

    agent = RandomAgent().as_engine()
    results = eval_all_baselines(agent, num_games=args.num_games, verbose=args.verbose)
    print_summary(results)
