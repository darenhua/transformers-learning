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
    # Harness 3 — play against an agent

    You play **white**, the registered agent plays **black**. Pick a move from
    the dropdown of legal moves and hit **Submit**. The board, the human-vs-
    agent move history, and game-over status are held in `mo.state` so they
    persist across reactive cell re-runs.
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
    # Live game state — held across cell re-runs so the human/agent moves
    # accumulate instead of resetting each reactive cycle.
    get_state, set_state = mo.state(
        {"board": None, "agent": None, "history": [], "agent_name": None}
    )
    return get_state, set_state


@app.cell
def _(mo):
    mo.md("""
    ## New game
    """)
    return


@app.cell
def _(mo, registered_label, registry):
    _ = registered_label

    agent_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Agent (plays black)"
    )
    board_class_choice = mo.ui.radio(
        options=["Board (10x10 international)", "AmericanBoard (8x8)"],
        value="Board (10x10 international)",
        label="Board class",
    )
    new_game_button = mo.ui.run_button(label="Start new game", kind="success")
    mo.hstack([agent_choice, board_class_choice, new_game_button])
    return agent_choice, board_class_choice, new_game_button


@app.cell
def _(agent_choice, board_class_choice, new_game_button, registry, set_state):
    from draughts import Board as _Board, AmericanBoard as _AmericanBoard

    _board_cls = _Board if "Board" in board_class_choice.value else _AmericanBoard

    if new_game_button.value:
        set_state(
            {
                "board": _board_cls(),
                "agent": registry.build(agent_choice.value),
                "agent_name": agent_choice.value,
                "history": [],
            }
        )
    return


@app.cell
def _(get_state, mo):
    _s = get_state()
    if _s["board"] is None:
        board_view = mo.md("_click **Start new game** to begin_")
    else:
        _b = _s["board"]
        _status = (
            f"**result:** `{_b.result}` (game over)"
            if _b.game_over
            else f"**turn:** `{_b.turn.name}`"
        )
        board_view = mo.md(
            f"### vs `{_s['agent_name']}`\n\n{_status}\n\n```\n{_b}\n```"
        )
    board_view
    return


@app.cell
def _(get_state, mo):
    _s = get_state()
    if _s["board"] is None or _s["board"].game_over:
        move_choice = mo.ui.dropdown(
            options=["—"], value="—", label="Your move"
        )
    else:
        _legal = list(_s["board"].legal_moves)
        if not _legal:
            move_choice = mo.ui.dropdown(
                options=["no legal moves"],
                value="no legal moves",
                label="Your move",
            )
        else:
            _opts = [str(m) for m in _legal]
            move_choice = mo.ui.dropdown(
                options=_opts, value=_opts[0], label="Your move"
            )
    submit_button = mo.ui.run_button(label="Submit move", kind="success")
    mo.hstack([move_choice, submit_button])
    return move_choice, submit_button


@app.cell
def _(get_state, move_choice, set_state, submit_button):
    if submit_button.value:
        _s = get_state()
        _b = _s["board"]
        if _b is not None and not _b.game_over:
            _next = _b.copy()
            _human = next(
                (m for m in _next.legal_moves if str(m) == move_choice.value),
                None,
            )
            if _human is not None:
                _next.push(_human)
                _s["history"].append(("you", str(_human)))

                # Agent's reply (if game still going)
                if not _next.game_over:
                    _agent_move = _s["agent"].select_move(_next)
                    _next.push(_agent_move)
                    _s["history"].append((_s["agent_name"], str(_agent_move)))

                set_state(
                    {
                        "board": _next,
                        "agent": _s["agent"],
                        "agent_name": _s["agent_name"],
                        "history": _s["history"],
                    }
                )
    return


@app.cell
def _(get_state, mo):
    _s = get_state()
    if not _s["history"]:
        history_view = mo.md("_no moves played yet_")
    else:
        _rows = "\n".join(
            f"{_i + 1}. **{_who}** → `{_mv}`"
            for _i, (_who, _mv) in enumerate(_s["history"][-30:])
        )
        history_view = mo.md(f"### Recent moves\n\n{_rows}")
    history_view
    return


if __name__ == "__main__":
    app.run()
