# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "tqdm",
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

    return AlphaBetaEngine, Board


@app.cell
def _():
    import os
    import sys

    for _root in (".", "final"):
        if os.path.isdir(_root) and _root not in sys.path:
            sys.path.insert(0, _root)
    # All orchestration lives in _dataset_generate so the same code is
    # used by this notebook and the headless background_task.sh runner.
    import _dataset_generate as _gen

    generate_valid_move_dataset = _gen.generate_valid_move_dataset
    generate_optimal_move_dataset = _gen.generate_optimal_move_dataset
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
    import os as _os_ui

    output_base = mo.ui.text(
        value="valid_move",
        label="Output base name (no extension)",
        full_width=True,
    )
    valid_target = mo.ui.number(
        value=20000, start=100, stop=200000, step=100,
        label="Target rows (valid_move SFT)",
    )
    optimal_target = mo.ui.number(
        value=5000, start=100, stop=50000, step=100,
        label="Target rows (optimal_move DPO)",
    )
    test_size = mo.ui.slider(
        0.0, 0.5, value=0.1, step=0.05,
        label="Test split fraction (0 = single file)",
    )
    random_ratio = mo.ui.slider(
        0.0, 1.0, value=0.2, step=0.05,
        label="Random-move injection ratio (valid_move only)",
    )
    randomize_start_plies = mo.ui.slider(
        0, 12, value=4, step=1,
        label="Random opening plies (0..N random plies before recording)",
    )
    opening_pct = mo.ui.number(
        value=30, start=0, stop=100, step=5, label="Opening %",
    )
    endgame_pct = mo.ui.number(
        value=30, start=0, stop=100, step=5, label="Endgame %",
    )
    max_workers = mo.ui.number(
        value=_os_ui.cpu_count() or 4, start=1, stop=128, step=1,
        label=f"Max workers (cpu_count={_os_ui.cpu_count()})",
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
            mo.hstack([valid_target, optimal_target, test_size]),
            mo.hstack([opening_pct, endgame_pct, max_workers, seed]),
            random_ratio,
            randomize_start_plies,
            scan_path,
        ]
    )
    return (
        endgame_pct,
        max_workers,
        opening_pct,
        optimal_target,
        output_base,
        random_ratio,
        randomize_start_plies,
        scan_path,
        seed,
        test_size,
        valid_target,
    )


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
    endgame_pct,
    generate_optimal_move_dataset,
    generate_valid_move_dataset,
    max_workers,
    mo,
    opening_pct,
    optimal_button,
    optimal_target,
    output_base,
    random_ratio,
    randomize_start_plies,
    scan_path,
    seed,
    test_size,
    valid_button,
    valid_target,
):
    import os as _os

    result = None
    error_msg = None

    if valid_button.value:
        result = generate_valid_move_dataset(
            target_total=int(valid_target.value),
            output_base=output_base.value,
            test_size=test_size.value,
            random_ratio=random_ratio.value,
            seed=int(seed.value),
            opening_pct=int(opening_pct.value),
            endgame_pct=int(endgame_pct.value),
            randomize_start_plies=int(randomize_start_plies.value),
            max_workers=int(max_workers.value),
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
            result = generate_optimal_move_dataset(
                target_total=int(optimal_target.value),
                output_base=output_base.value,
                test_size=test_size.value,
                scan_path=_resolved,
                strong_time=0.3,
                weak_time=0.05,
                seed=int(seed.value),
                opening_pct=int(opening_pct.value),
                endgame_pct=int(endgame_pct.value),
                randomize_start_plies=int(randomize_start_plies.value),
                max_workers=int(max_workers.value),
            )

    if error_msg is not None:
        view = mo.md(f"⚠️ **{error_msg}**")
    elif result is None:
        view = mo.md("_click a generate button to start_")
    else:
        _paths, _var, _games = result
        _path_lines = ["### wrote:"] + [
            f"- `{rel}` ({n} rows) → `{_os.path.abspath(rel)}`"
            for rel, n in _paths
        ]
        if isinstance(_var, str):
            _var_md = _var
        else:
            _phase = _var["phase"]
            _bins = _var["piece_count_bins"]
            _var_md = (
                f"### variance ({_var['total']} kept rows, {_games} games)\n"
                f"- **phase:** opening={_phase['opening']} "
                f"midgame={_phase['midgame']} endgame={_phase['endgame']}\n"
                f"- **side W:** {_var['side_W_pct']:.1f}%\n"
                f"- **forced capture:** {_var['forced_capture_pct']:.1f}%\n"
                f"- **single legal move:** {_var['single_legal_pct']:.1f}%\n"
                f"- **kings present:** {_var['kings_present_pct']:.1f}%\n"
                f"- **piece count:** avg={_var['piece_count_avg']:.1f} "
                f"min={_var['piece_count_min']} max={_var['piece_count_max']}\n"
                f"- **piece-count bins:** "
                + ", ".join(f"{k}={v}" for k, v in _bins.items())
            )
        view = mo.md("\n\n".join(["\n".join(_path_lines), _var_md]))
    view
    return


if __name__ == "__main__":
    app.run()
