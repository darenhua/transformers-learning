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
    # Harness 2 — single-game introspection

    Plays one game of `agent` (white) vs `opponent` (black), then renders every
    turn the agent took: prompt FEN, raw model response, parsed move, the
    legal-first-try / random-fallback flags, and the move actually played.

    Pick `llm_local` as the agent to actually see turn records — baselines
    don't record per-turn data.
    """)
    return


@app.cell
def _():
    from harness import registry

    return (registry,)


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
    ## Single game
    """)
    return


@app.cell
def _(mo):
    board_class_choice = mo.ui.radio(
        options=["Board (10x10 international)", "AmericanBoard (8x8)"],
        value="Board (10x10 international)",
        label="Board class",
    )
    board_class_choice
    return (board_class_choice,)


@app.cell
def _(mo, registered_label, registry):
    _ = registered_label

    agent_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Agent (white)"
    )
    opp_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Opponent (black)"
    )
    max_moves = mo.ui.slider(20, 400, value=200, label="Max plies (safety cap)")
    play_button = mo.ui.run_button(label="Play one game", kind="success")
    mo.hstack([agent_choice, opp_choice, max_moves, play_button])
    return agent_choice, max_moves, opp_choice, play_button


@app.cell
def _(
    agent_choice,
    board_class_choice,
    max_moves,
    opp_choice,
    play_button,
    registry,
):
    from draughts import (
        Board as _Board,
        AmericanBoard as _AmericanBoard,
        Color as _Color,
    )

    _board_cls = _Board if "Board" in board_class_choice.value else _AmericanBoard

    game_summary = None
    turns_data = []
    if play_button.value:
        agent = registry.build(agent_choice.value)
        opp = registry.build(opp_choice.value)
        board = _board_cls()

        plies = 0
        while not board.game_over and plies < max_moves.value:
            mover = agent if board.turn == _Color.WHITE else opp
            board.push(mover.select_move(board))
            plies += 1

        for _i, _t in enumerate(getattr(agent, "turns", [])):
            turns_data.append(
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
            )

        game_summary = {
            "result": board.result,
            "plies": plies,
            "agent_turns_recorded": len(getattr(agent, "turns", [])),
            "agent_legal_first_try": getattr(agent, "first_try_legal", None),
            "agent_total_fallbacks": getattr(agent, "total_fallbacks", None),
            "board_class": _board_cls.__name__,
        }
    return game_summary, turns_data


@app.cell
def _(game_summary, mo, turns_data):
    if game_summary is None:
        view = mo.md("_click **Play one game** to start_")
    else:
        _lines = [
            f"### result `{game_summary['result']}`",
            f"_played on `{game_summary['board_class']}` for {game_summary['plies']} plies_",
            "",
            f"- **agent turns recorded:** {game_summary['agent_turns_recorded']}",
        ]
        if game_summary["agent_legal_first_try"] is not None:
            _lines += [
                f"- **agent legal-first-try count:** {game_summary['agent_legal_first_try']}",
                f"- **agent random-fallback count:** {game_summary['agent_total_fallbacks']}",
            ]
        _summary = mo.md("\n".join(_lines))
        if turns_data:
            view = mo.vstack([_summary, mo.ui.table(turns_data, sortable=False)])
        else:
            view = mo.vstack(
                [
                    _summary,
                    mo.md(
                        "_(no turn records — pick `llm_local` as the agent to "
                        "see per-turn introspection)_"
                    ),
                ]
            )
    view
    return


if __name__ == "__main__":
    app.run()
