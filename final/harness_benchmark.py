# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch==2.11.0",
#     "transformers",
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
    # Harness — gameplay benchmark (winrate)

    Plays N games of `agent` vs `opponent` via py-draughts `Benchmark` and
    reports W-L-D, win rate, Elo diff. If `agent` is `llm_local`, the
    per-turn LLM log (prompt FEN, raw response, parsed move, legal-first-
    try flag, fallback flag) is also rendered as a table after the run so
    you can audit what the model did.

    For the cheaper per-checkpoint legality eval (no full games, just
    "does the model produce a legal move at this position?"), use
    `harness_valid.py`.
    """)
    return


@app.cell
def _():
    from harness import registry
    from harness.engines import run_benchmark

    return registry, run_benchmark


@app.cell
def _(mo):
    mo.md("""
    ## Register a trained checkpoint
    """)
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.run_button(label="Refresh checkpoints")
    refresh_button
    return (refresh_button,)


@app.cell
def _(mo, refresh_button):
    import os as _os

    # `refresh_button` in deps so clicking it rebuilds the dropdown options.
    _ = refresh_button.value

    options = ["gpt2"]
    if _os.path.isdir("checkpoints"):
        for _name in sorted(_os.listdir("checkpoints")):
            _p = _os.path.join("checkpoints", _name)
            if _os.path.isdir(_p) and _os.path.exists(
                _os.path.join(_p, "config.json")
            ):
                options.append(_p)

    checkpoint_choice = mo.ui.dropdown(
        options=options, value=options[-1], label="Checkpoint"
    )
    register_button = mo.ui.run_button(label="Register llm_local", kind="success")
    mo.hstack([checkpoint_choice, register_button])
    return checkpoint_choice, register_button


@app.cell
def _(checkpoint_choice, mo, register_button, registry):
    # Loads the chosen checkpoint and registers it as "llm_local". Each click
    # rebuilds the agent (fresh weights, fresh generate_fn closure) so re-
    # registering after a new training run picks up the latest weights.
    registered_label = None
    if register_button.value:
        from transformers import (
            AutoModelForCausalLM as _AMC,
            AutoTokenizer as _AT,
        )
        import torch as _torch

        from harness.llm_agent import LLMAgent
        from harness.hf_generate import make_hf_generate_fn

        _path = checkpoint_choice.value
        _tok = _AT.from_pretrained(_path)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _model = _AMC.from_pretrained(_path)
        _model.config.pad_token_id = _tok.eos_token_id
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()

        _generate = make_hf_generate_fn(
            _model, _tok, max_new_tokens=12, device=_device
        )

        def _llm_factory():
            return LLMAgent(generate_fn=_generate, record_turns=True)

        registry.register("llm_local", _llm_factory)
        registered_label = f"llm_local ← `{_path}` ({_device})"

    badge = (
        mo.md(f"**registered:** {registered_label}")
        if registered_label
        else mo.md("_no llm_local registered yet_")
    )
    badge
    return (registered_label,)


@app.cell
def _(mo):
    mo.md("""
    ## Benchmark
    """)
    return


@app.cell
def _():
    # # llm_local was trained on a 10x10 (International) dataset, so the
    # # benchmark must use Board to keep prompt FENs in the same distribution.
    # # AmericanBoard is here for baseline-only matchups once you have 8x8 data.
    # board_class_choice = mo.ui.radio(
    #     options=["Board (10x10 international)", "AmericanBoard (8x8)"],
    #     value="Board (10x10 international)",
    #     label="Board class",
    # )
    # board_class_choice

    board_class_choice = "Board (10x10 international)"
    return (board_class_choice,)


@app.cell
def _(mo, registered_label, registry):
    # `registered_label` in deps so the dropdowns rebuild after a new
    # registration adds "llm_local" to registry.names().
    _ = registered_label

    agent_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Agent"
    )
    opp_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Opponent"
    )
    games = mo.ui.slider(1, 50, value=3, label="Games")
    run = mo.ui.run_button(label="Run benchmark", kind="success")
    mo.hstack([agent_choice, opp_choice, games, run])
    return agent_choice, games, opp_choice, run


@app.cell
def _(
    agent_choice,
    board_class_choice,
    games,
    opp_choice,
    registry,
    run,
    run_benchmark,
):
    from draughts import Board as _Board, AmericanBoard as _AmericanBoard

    _board_cls = _Board if "Board" in board_class_choice else _AmericanBoard

    sim_result = None
    sim_turns = []

    if run.value:
        # Build agents in the cell (not via run_benchmark's internals) so
        # we keep a reference to the agent and can read its `turns` log
        # after the benchmark finishes.
        _agent_inst = registry.build(agent_choice.value)
        _opp_inst = registry.build(opp_choice.value)
        sim_result = run_benchmark(
            agent=_agent_inst,
            opponent=_opp_inst,
            games=games.value,
            opponent_name=opp_choice.value,
            board_class=_board_cls,
        )
        sim_turns = list(getattr(_agent_inst, "turns", []))
    return sim_result, sim_turns


@app.cell
def _(agent_choice, board_class_choice, mo, sim_result, sim_turns):
    if sim_result is None:
        view = mo.md("_click **Run benchmark** to start_")
    else:
        _lines = [
            f"### {agent_choice.value} vs {sim_result.opponent}",
            f"_{sim_result.games} games · `{board_class_choice}`_",
            "",
            f"- **W-L-D:** {sim_result.wins}-{sim_result.losses}-{sim_result.draws}",
            f"- **win rate:** {sim_result.win_rate:.1%}",
            f"- **elo diff:** {sim_result.elo_diff:+.0f}",
        ]
        if sim_result.legal_move_rate is not None:
            _lines += [
                f"- **legal first try:** {sim_result.legal_move_rate:.1%}",
                f"- **fallback rate:** {sim_result.fallback_rate:.1%}",
            ]
        _summary = mo.md("\n".join(_lines))

        if sim_turns:
            _table_data = [
                {
                    "ply": _i + 1,
                    "fen_before": _t.fen_before[:60],
                    "raw_response": (_t.responses[0] if _t.responses else "")[:60],
                    "parsed": _t.parsed or "",
                    "legal_first_try": _t.legal_first_try,
                    "fell_back_random": _t.fell_back_to_random,
                    "chosen": _t.chosen_move or "",
                    "retries": len(_t.responses),
                }
                for _i, _t in enumerate(sim_turns)
            ]
            view = mo.vstack(
                [
                    _summary,
                    mo.md(
                        f"### Per-turn LLM log "
                        f"({len(sim_turns)} turns recorded)"
                    ),
                    mo.ui.table(_table_data, sortable=False),
                ]
            )
        else:
            view = mo.vstack(
                [
                    _summary,
                    mo.md(
                        "_(no turn records — pick `llm_local` as agent "
                        "to see per-turn introspection)_"
                    ),
                ]
            )
    view
    return


if __name__ == "__main__":
    app.run()
