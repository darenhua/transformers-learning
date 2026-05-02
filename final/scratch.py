# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Phase 0 scratch — harness import sanity check

    Verifies that all `harness/*` modules import cleanly and the model
    registry is populated. Expected output of the next cell:

    `['random', 'greedy', 'minimax_d3', 'minimax_d8']`
    """)
    return


@app.cell
def _():
    from harness.agents import RandomAgent, GreedyAgent, MinimaxAgent

    # this parses move for valid 1 or 0 score
    from harness.scoring import parse_move

    # prompt builder
    from harness.prompts import build_minimal_prompt

    # run entire flow
    from harness.engines import run_benchmark

    # fixed models
    from harness.registry import MODELS, names, build
    from harness.llm_agent import LLMAgent

    # helper for hf gen text
    from harness.hf_generate import make_hf_generate_fn

    names()
    return build, run_benchmark


@app.cell
def _(mo):
    mo.md("""
    ## Phase 1 — one-shot `run_benchmark`

    Two games of random vs minimax_d3 — runs in a few seconds. Should return
    a `BenchmarkResult` with `games=2`, `wins+losses+draws == 2`, finite
    `elo_diff`. No UI yet; just proving the engine wrapper round-trips.
    """)
    return


@app.cell
def _(build, run_benchmark):
    result = run_benchmark(
        agent=build("random"),
        opponent=build("minimax_d3"),
        games=2,
        opponent_name="minimax_d3",
    )
    result
    return


if __name__ == "__main__":
    app.run()
