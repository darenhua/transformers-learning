# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Generate checkers datasets

    Two output formats, both produced from minimax/Scan self-play:

    - **valid_move (SFT)** — `{fen, move, game_idx}` rows. minimax depth-2
      vs itself, with a random-move injection ratio so the model sees
      reasonable but not-always-optimal positions.
    - **optimal_move (DPO)** — `{fen, chosen, rejected, game_idx}` rows.
      Strong Scan engine produces `chosen`, weak Scan produces `rejected`;
      ties are skipped. Requires the Scan binary (path configurable).

    If `Test split fraction` > 0, each generator writes
    `<base>_train.jsonl` and `<base>_test.jsonl`. Otherwise a single
    `<base>.jsonl`. The held-out `<base>_test.jsonl` is what
    `harness_benchmark.py` consumes in **Test dataset** mode for cheap
    per-checkpoint top-1 accuracy + legal-rate scoring.
    """)
    return


@app.cell
def _():
    from draughts import Board, AlphaBetaEngine, HubEngine

    return AlphaBetaEngine, Board, HubEngine


@app.cell
def _(AlphaBetaEngine, Board, HubEngine):
    import json
    import random

    MAX_MOVES_PER_GAME = 300

    # pydraughts 1.7.1's HubEngine._read_line uses select() + readline() on
    # the text stream, which deadlocks when Scan emits several lines in one
    # OS chunk: the lines land in Python's TextIOWrapper buffer, but select
    # stops reporting the underlying fd as readable, so subsequent reads
    # time out forever. Swap in a plain blocking readline; Scan responds
    # promptly.
    def _hub_blocking_read_line(self, timeout: float = 1.0):
        if self.process is None or self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        return line.strip()

    HubEngine._read_line = _hub_blocking_read_line

    def write_records(records, output_base, test_size, seed):
        """Write records as a single jsonl, or split into <base>_train.jsonl
        + <base>_test.jsonl if test_size > 0. Returns [(path, count), ...]."""
        if test_size <= 0 or len(records) == 0:
            path = f"{output_base}.jsonl"
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            return [(path, len(records))]

        rng = random.Random(seed)
        idxs = list(range(len(records)))
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(records) * test_size)))
        test_idx = set(idxs[:n_test])
        train, test = [], []
        for i, r in enumerate(records):
            (test if i in test_idx else train).append(r)
        paths = []
        for split_name, rs in [("train", train), ("test", test)]:
            p = f"{output_base}_{split_name}.jsonl"
            with open(p, "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            paths.append((p, len(rs)))
        return paths

    def generate_valid_move_dataset(
        num_games, output_base, test_size, random_ratio, seed
    ):
        """SFT format: {fen, move, game_idx}. minimax depth-2 self-play with
        `random_ratio` chance per ply of a random legal move instead of
        the engine's choice (so the model sees suboptimal positions, not
        just on-policy minimax states)."""
        rng = random.Random(seed)
        engine = AlphaBetaEngine(depth_limit=2)
        records = []
        for i in range(num_games):
            if num_games >= 100 and i % 100 == 0:
                print(f"valid_move game {i}/{num_games}")
            board = Board()
            for _ in range(MAX_MOVES_PER_GAME):
                if not board.legal_moves:
                    break
                if rng.random() >= random_ratio:
                    move = engine.get_best_move(board)
                else:
                    move = rng.choice(list(board.legal_moves))
                if move is None:
                    break
                records.append(
                    {
                        "fen": board.fen,
                        "move": str(move),
                        "game_idx": i,
                    }
                )
                board.push(move)
        print(
            f"valid_move: collected {len(records)} records from {num_games} games"
        )
        return write_records(records, output_base, test_size, seed)

    def generate_optimal_move_dataset(
        num_games,
        output_base,
        test_size,
        scan_path,
        strong_time,
        weak_time,
        seed,
    ):
        """DPO format: {fen, chosen, rejected, game_idx}. Strong Scan picks
        `chosen` and drives the trajectory; weak Scan picks `rejected` from
        the same position. Ties are skipped (no preference signal)."""
        records = []
        skipped = 0
        with (
            HubEngine(
                scan_path, time_limit=strong_time, init_timeout=60.0
            ) as strong,
            HubEngine(
                scan_path, time_limit=weak_time, init_timeout=60.0
            ) as weak,
        ):
            for i in range(num_games):
                if num_games >= 10 and i % 10 == 0:
                    print(f"optimal_move game {i}/{num_games}")
                strong.new_game()
                weak.new_game()
                board = Board()
                for _ in range(MAX_MOVES_PER_GAME):
                    if not board.legal_moves:
                        break
                    chosen = strong.get_best_move(board)
                    if chosen is None:
                        break
                    rejected = weak.get_best_move(board)
                    chosen_str = str(chosen)
                    if rejected is None or str(rejected) == chosen_str:
                        skipped += 1
                    else:
                        records.append(
                            {
                                "fen": board.fen,
                                "chosen": chosen_str,
                                "rejected": str(rejected),
                                "game_idx": i,
                            }
                        )
                    board.push(chosen)
        print(
            f"optimal_move: collected {len(records)} records, skipped {skipped} ties"
        )
        return write_records(records, output_base, test_size, seed)

    return generate_optimal_move_dataset, generate_valid_move_dataset


@app.cell
def _(AlphaBetaEngine, Board):
    print("=== smoke test: 1 move at depth 2 ===")
    _b = Board()
    _e = AlphaBetaEngine(depth_limit=2)
    _m = _e.get_best_move(_b)
    print(f"initial fen: {_b.fen!r}")
    print(f"best move:   {_m!r}, str={str(_m)!r}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    output_base = mo.ui.text(
        value="valid_move",
        label="Output base name (no extension)",
        full_width=True,
    )
    num_games = mo.ui.slider(1, 5000, value=20, label="Num games")
    random_ratio = mo.ui.slider(
        0.0,
        1.0,
        value=0.2,
        step=0.05,
        label="Random-move injection ratio (valid_move only)",
    )
    test_size = mo.ui.slider(
        0.0,
        0.5,
        value=0.1,
        step=0.05,
        label="Test split fraction (0 = single file)",
    )
    seed = mo.ui.number(value=42, label="Seed", start=0, stop=2**31 - 1)
    scan_path = mo.ui.text(
        value="../scan/scan_31/scan_linux",
        label="Scan binary path (optimal_move only)",
        full_width=True,
    )

    mo.vstack(
        [
            output_base,
            mo.hstack([num_games, test_size, seed]),
            random_ratio,
            scan_path,
        ]
    )
    return num_games, output_base, random_ratio, scan_path, seed, test_size


@app.cell
def _(mo):
    valid_button = mo.ui.run_button(
        label="Generate valid_move (SFT)", kind="success"
    )
    optimal_button = mo.ui.run_button(
        label="Generate optimal_move (DPO)", kind="warn"
    )
    mo.hstack([valid_button, optimal_button])
    return optimal_button, valid_button


@app.cell
def _(
    generate_optimal_move_dataset,
    generate_valid_move_dataset,
    mo,
    num_games,
    optimal_button,
    output_base,
    random_ratio,
    scan_path,
    seed,
    test_size,
    valid_button,
):
    import os as _os

    last_paths = None
    error_msg = None

    if valid_button.value:
        last_paths = generate_valid_move_dataset(
            num_games=num_games.value,
            output_base=output_base.value,
            test_size=test_size.value,
            random_ratio=random_ratio.value,
            seed=int(seed.value),
        )
    elif optimal_button.value:
        # Try literal scan path then `../<path>` fallback so this notebook
        # works whether marimo was launched from project root or final/.
        _candidates = [scan_path.value, f"../{scan_path.value}"]
        _resolved = next(
            (p for p in _candidates if _os.path.exists(p)), None
        )
        if _resolved is None:
            error_msg = (
                f"Scan binary not found — tried {_candidates!r}. "
                f"Edit the path field above."
            )
        else:
            last_paths = generate_optimal_move_dataset(
                num_games=num_games.value,
                output_base=output_base.value,
                test_size=test_size.value,
                scan_path=_resolved,
                strong_time=0.3,
                weak_time=0.05,
                seed=int(seed.value),
            )

    if error_msg is not None:
        view = mo.md(f"⚠️ **{error_msg}**")
    elif last_paths is None:
        view = mo.md("_click a generate button to start_")
    else:
        _abs = [(_os.path.abspath(p), p, n) for p, n in last_paths]
        _lines = ["### wrote:"] + [
            f"- `{rel}` ({n} rows) → `{abs_}`" for abs_, rel, n in _abs
        ]
        view = mo.md("\n".join(_lines))
    view
    return


if __name__ == "__main__":
    app.run()
