"""Secondary Metric: Move consistency.

Given a set of fixed board positions, present each to the model N times
and measure how consistently it picks the same move. High consistency
suggests a learned strategy; low consistency suggests sampling randomness.
"""

import csv
from collections import Counter

from draughts import AmericanBoard as Board


def load_board_states(csv_path: str) -> list[dict]:
    """Load board positions from a CSV. Expected columns: fen, description (optional)."""
    states = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append({
                "fen": row["fen"],
                "description": row.get("description", ""),
            })
    return states


def eval_consistency(agent, csv_path: str, num_sessions: int = 10) -> dict:
    """Evaluate move consistency of an agent across repeated sessions.

    Returns dict with per_position details and overall_consistency score.
    """
    board_states = load_board_states(csv_path)
    per_position = []

    for state in board_states:
        fen = state["fen"]
        move_counts = Counter()

        for _ in range(num_sessions):
            board = Board.from_fen(fen)
            if not board.legal_moves:
                break
            move = agent.select_move(board)
            move_counts[str(move)] += 1

        if move_counts:
            most_common_count = move_counts.most_common(1)[0][1]
            consistency = most_common_count / num_sessions
        else:
            consistency = 0.0

        per_position.append({
            "fen": fen,
            "description": state["description"],
            "move_counts": dict(move_counts),
            "consistency_score": consistency,
        })

    overall = (
        sum(p["consistency_score"] for p in per_position) / len(per_position)
        if per_position else 0.0
    )

    return {
        "per_position": per_position,
        "overall_consistency": overall,
        "num_sessions": num_sessions,
        "num_positions": len(per_position),
    }


def print_summary(results: dict):
    """Print concise one-line-per-position summary."""
    print(f"\n{'Position':<30} {'Consistency':>12} {'Top Move':>12}")
    print("-" * 56)
    for p in results["per_position"]:
        label = p["description"][:28] if p["description"] else p["fen"][:28]
        top_move = max(p["move_counts"], key=p["move_counts"].get) if p["move_counts"] else "n/a"
        print(f"{label:<30} {p['consistency_score']:>11.0%} {top_move:>12}")

    print("-" * 56)
    print(f"{'OVERALL':.<30} {results['overall_consistency']:>11.0%}")
    print(f"  ({results['num_positions']} positions x {results['num_sessions']} sessions)\n")


def print_verbose(results: dict):
    """Print full per-position move distribution."""
    for p in results["per_position"]:
        desc = f" ({p['description']})" if p["description"] else ""
        print(f"\n  FEN: {p['fen']}{desc}")
        print(f"  Consistency: {p['consistency_score']:.0%}")
        for move, count in sorted(p["move_counts"].items(), key=lambda x: -x[1]):
            bar = "#" * count
            print(f"    {move:>10s}: {count:>3d}  {bar}")

    print(f"\n  Overall: {results['overall_consistency']:.0%}\n")


if __name__ == "__main__":
    import os
    import argparse
    from models.random_bot import RandomAgent

    parser = argparse.ArgumentParser(description="Evaluate move consistency across fixed board positions")
    parser.add_argument("-n", "--num-sessions", type=int, default=10, help="sessions per position (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full move distributions")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "board_states.csv")
    agent = RandomAgent()
    results = eval_consistency(agent, csv_path, num_sessions=args.num_sessions)

    if args.verbose:
        print_verbose(results)
    print_summary(results)
