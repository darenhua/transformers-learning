# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch",
#     "transformers",
#     "peft",
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
    mo.md(r"""
    # Win-rate benchmark — Qwen SFT / DPO vs random + minimax

    Uses py-draughts' `Benchmark` directly (the same one in the README's
    engine-benchmarking example) to play N games per matchup and report
    W-L-D, win rate, and Elo difference.

    Both LLMs play **two opponents** on the standard 10x10 `Board`:

    - `RandomAgent` — easy baseline (any legal-move signal should beat this);
    - `AlphaBetaEngine(depth_limit=3)` — the real test.

    Both LLMs are wrapped in a robust engine
    (`harness.llm_engine.make_robust_llm_engine`):

    - up to 5 retries on illegal moves (sampling-on-retry so retries diversify);
    - random move from `board.legal_moves` if all retries fail;
    - **the engine never returns an illegal move.**

    Per-turn `You: <FEN>\n AI: <move>` lines stream to `logs/benchmark.log`,
    which the **Live LLM-conversation log** cell at the bottom tails.
    """)
    return


@app.cell
def _():
    import os
    import sys

    for _root in (".", "final"):
        if os.path.isdir(_root) and _root not in sys.path:
            sys.path.insert(0, _root)
    for _root in ("training-jobs", "final/training-jobs"):
        if os.path.isdir(_root) and _root not in sys.path:
            sys.path.insert(0, _root)

    from draughts import AlphaBetaEngine, Benchmark, Board

    from harness.agents import RandomAgent
    from harness.llm_engine import make_robust_llm_engine

    return Benchmark, Board, RandomAgent, make_robust_llm_engine


@app.cell
def _(mo):
    sft_path = mo.ui.text(
        value="checkpoints/qwen-sft", label="Qwen SFT checkpoint", full_width=True
    )
    dpo_path = mo.ui.text(
        value="checkpoints/qwen-dpo", label="Qwen DPO checkpoint", full_width=True
    )
    mo.vstack([sft_path, dpo_path])
    return dpo_path, sft_path


@app.cell
def _(mo):
    games = mo.ui.slider(2, 40, value=10, step=2, label="Games per matchup")
    depth = mo.ui.slider(1, 6, value=3, label="Minimax depth (opponent)")
    max_moves = mo.ui.slider(40, 400, value=200, step=20, label="Max moves per game")
    run = mo.ui.run_button(label="Run benchmark", kind="success")
    mo.vstack([mo.hstack([games, depth, max_moves]), run])
    return games, max_moves, run


@app.cell
def _(
    Benchmark,
    Board,
    RandomAgent,
    dpo_path,
    games,
    make_robust_llm_engine,
    max_moves,
    mo,
    run,
    sft_path,
):
    # Heavy: loads two checkpoints and plays 4*games games (2 models × 2
    # opponents). Gated on the Run button.
    mo.stop(not run.value, mo.md("_click **Run benchmark** to start_"))

    import pathlib as _pl
    bench_log = _pl.Path("logs/benchmark.log")
    bench_log.parent.mkdir(parents=True, exist_ok=True)
    # Truncate so the tail cell shows just this run.
    bench_log.write_text("")

    # Each LLM plays both opponents — random is the easy baseline, minimax(d)
    # is the harder one. Lambdas so we get a fresh engine per matchup
    # (AlphaBetaEngine carries internal state across games we don't want).
    _opponents = [
        ("random", lambda: RandomAgent().as_engine())
        # (
        #     f"minimax(d={depth.value})",
        #     lambda: AlphaBetaEngine(depth_limit=depth.value),
        # ),
    ]

    bench_rows = []
    for _model, _path in [
        ("Qwen SFT", sft_path.value),
        ("Qwen DPO", dpo_path.value),
    ]:
        print(f"[bench] loading {_model} from {_path!r}", flush=True)
        _agent, _engine = make_robust_llm_engine(
            _path,
            max_retries=5,
            max_new_tokens=12,
            log_path=str(bench_log),
            side_label=f"AI({_model})",
        )
        _engine.name = _model

        for _opp_name, _opp_factory in _opponents:
            # Reset per-matchup so legal_move_rate / fallback_rate reflect
            # only the games against this opponent.
            _agent.reset_stats()
            with bench_log.open("a") as _f:
                _f.write(
                    f"=== {_model} vs {_opp_name} — "
                    f"{games.value} games, max_moves={max_moves.value} ===\n\n"
                )
            print(
                f"[bench] {_model} vs {_opp_name} — "
                f"{games.value} games, max_moves={max_moves.value}",
                flush=True,
            )
            _stats = Benchmark(
                _engine,
                _opp_factory(),
                board_class=Board,
                games=games.value,
                max_moves=max_moves.value,
            ).run()
            bench_rows.append(
                {
                    "model": _model,
                    "opponent": _opp_name,
                    "games": _stats.games,
                    "W-L-D": f"{_stats.e1_wins}-{_stats.e2_wins}-{_stats.draws}",
                    "win_rate": f"{_stats.e1_win_rate:.1%}",
                    "elo_diff": f"{_stats.elo_diff:+.0f}",
                    "legal_first_try": f"{_agent.legal_move_rate():.1%}",
                    "fallback_rate": f"{_agent.fallback_rate():.1%}",
                    "avg_moves": f"{_stats.avg_moves:.1f}",
                    "_stats_str": str(_stats),
                }
            )
            print(f"[bench] {_model} vs {_opp_name} done", flush=True)

    bench_rows
    return (bench_rows,)


@app.cell
def _(bench_rows, mo):
    if not bench_rows:
        view = mo.md("_(no results yet)_")
    else:
        # Strip the raw-stats string from the table view; show it underneath.
        _table_rows = [
            {k: v for k, v in r.items() if k != "_stats_str"} for r in bench_rows
        ]
        view = mo.vstack(
            [
                mo.md("## Results"),
                mo.ui.table(_table_rows, sortable=False),
                mo.md("### Raw `BenchmarkStats` output"),
                mo.md(
                    "\n\n".join(
                        f"```\n{r['_stats_str']}\n```" for r in bench_rows
                    )
                ),
            ]
        )
    view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Live LLM-conversation log (benchmark games)

    The benchmark cell above writes every LLM turn to `logs/benchmark.log` in
    `You: <FEN>\n AI: <move>` form, with `=== Qwen ... vs ... ===` headers
    between matchups. This cell tails that file.

    (The `play_server.py` script does NOT write to a file — its conversation
    log goes straight to the server process's stdout, so watch the terminal
    where you launched it.)

    Pick an auto-refresh interval (or click **Refresh log**) to re-read.
    """)
    return


@app.cell
def _(mo):
    log_path = mo.ui.text(
        value="logs/benchmark.log", label="Log file", full_width=True
    )
    tail_n = mo.ui.slider(10, 400, value=80, step=10, label="Tail (lines)")
    # mo.ui.refresh rejects "off" — every option must be a valid timestring.
    # Pause auto-refresh by emptying the dropdown selection in the UI.
    refresh_log = mo.ui.refresh(
        options=["1s", "2s", "5s", "30s"], default_interval="2s"
    )
    refresh_btn = mo.ui.run_button(label="Refresh log")
    mo.hstack([log_path, tail_n, refresh_log, refresh_btn])
    return log_path, refresh_btn, refresh_log, tail_n


@app.cell
def _(log_path, mo, refresh_btn, refresh_log, tail_n):
    import os as _os

    # `refresh_log.value` ticks on the auto-interval; refresh_btn is manual.
    _ = (refresh_log.value, refresh_btn.value)

    _path = log_path.value
    if not _os.path.exists(_path):
        view2 = mo.md(f"_no log at `{_path}` yet — start `play_server.py` to populate_")
    else:
        with open(_path, "r") as _f:
            _lines = _f.readlines()
        _tail = "".join(_lines[-tail_n.value :])
        view2 = mo.vstack(
            [
                mo.md(f"**{_path}** — {len(_lines)} lines, showing last {tail_n.value}"),
                mo.md(f"```\n{_tail}\n```"),
            ]
        )
    view2
    return


if __name__ == "__main__":
    app.run()
