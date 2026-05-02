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
    # Harness 3 (server flavor) — agent behind a py-draughts Server

    Wraps the registered agent's `.as_engine()` into a `draughts.Server` and
    runs it on a daemon thread so external clients can connect and play. Use
    this when you want a real network boundary (e.g. a separate web client);
    for in-marimo play, use `harness_gameplay.py` instead.

    Re-running this notebook leaks the previous server thread — the daemon
    thread dies when marimo's kernel restarts. Don't click **Start server**
    twice in a row without restarting; the second click will fail to bind.
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
    ## Start server
    """)
    return


@app.cell
def _(mo, registered_label, registry):
    _ = registered_label

    white_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="White engine"
    )
    black_choice = mo.ui.dropdown(
        options=registry.names(), value="random", label="Black engine"
    )
    board_class_choice = mo.ui.radio(
        options=["Board (10x10 international)", "AmericanBoard (8x8)"],
        value="Board (10x10 international)",
        label="Board class",
    )
    start_button = mo.ui.run_button(label="Start server", kind="success")
    mo.vstack(
        [
            mo.hstack([white_choice, black_choice]),
            mo.hstack([board_class_choice, start_button]),
        ]
    )
    return black_choice, board_class_choice, start_button, white_choice


@app.cell
def _(black_choice, board_class_choice, registry, start_button, white_choice):
    import threading

    from draughts import (
        Server as _Server,
        Board as _Board,
        AmericanBoard as _AmericanBoard,
    )

    _board_cls = _Board if "Board" in board_class_choice.value else _AmericanBoard

    server_status = None
    if start_button.value:
        _white_eng = registry.build(white_choice.value).as_engine()
        _black_eng = registry.build(black_choice.value).as_engine()
        _server = _Server(
            board=_board_cls(),
            white_engine=_white_eng,
            black_engine=_black_eng,
        )

        def _run():
            # No running event loop in this thread, so server.run()'s internal
            # asyncio.run() is fine.
            _server.run()

        threading.Thread(target=_run, daemon=True).start()
        server_status = (
            f"started · white=`{white_choice.value}` · "
            f"black=`{black_choice.value}` · board=`{_board_cls.__name__}`"
        )
    return (server_status,)


@app.cell
def _(mo, server_status):
    if server_status is None:
        view = mo.md("_click **Start server** to launch_")
    else:
        view = mo.md(f"**server:** {server_status}")
    view
    return


if __name__ == "__main__":
    app.run()
