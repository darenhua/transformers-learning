"""Headless CLI for dataset generation. Wraps `_dataset_generate` so
background_task.sh can produce datasets without driving the marimo UI.

Examples:
    python final/generate_dataset_cli.py valid \\
        --target 20000 --output-base valid_move

    python final/generate_dataset_cli.py optimal \\
        --target 5000 --output-base optimal_move \\
        --scan-path ../scan/scan_31/scan_linux
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _dataset_generate as gen


def _add_common(p):
    p.add_argument("--target", type=int, required=True, help="Total rows")
    p.add_argument("--output-base", required=True)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--randomize-start-plies", type=int, default=4)
    p.add_argument("--opening-pct", type=int, default=30)
    p.add_argument("--endgame-pct", type=int, default=30)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--games-per-worker", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    v = sub.add_parser("valid", help="SFT (valid_move) generator")
    _add_common(v)
    v.add_argument("--random-ratio", type=float, default=0.2)

    o = sub.add_parser("optimal", help="DPO (optimal_move) generator")
    _add_common(o)
    o.add_argument("--scan-path", required=True)
    o.add_argument("--strong-time", type=float, default=0.3)
    o.add_argument("--weak-time", type=float, default=0.05)

    a = ap.parse_args()

    if a.mode == "valid":
        paths, var, games = gen.generate_valid_move_dataset(
            target_total=a.target,
            output_base=a.output_base,
            test_size=a.test_size,
            random_ratio=a.random_ratio,
            seed=a.seed,
            opening_pct=a.opening_pct,
            endgame_pct=a.endgame_pct,
            randomize_start_plies=a.randomize_start_plies,
            max_workers=a.max_workers,
            games_per_worker=a.games_per_worker,
        )
    else:
        paths, var, games = gen.generate_optimal_move_dataset(
            target_total=a.target,
            output_base=a.output_base,
            test_size=a.test_size,
            scan_path=a.scan_path,
            strong_time=a.strong_time,
            weak_time=a.weak_time,
            seed=a.seed,
            opening_pct=a.opening_pct,
            endgame_pct=a.endgame_pct,
            randomize_start_plies=a.randomize_start_plies,
            max_workers=a.max_workers,
            games_per_worker=a.games_per_worker,
        )

    print("\n=== written ===")
    for path, n in paths:
        print(f"  {path}  ({n} rows)")
    print("\n=== variance ===")
    print(json.dumps(var, indent=2, default=str))


if __name__ == "__main__":
    main()
