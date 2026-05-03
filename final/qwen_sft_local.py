# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch==2.11.0",
#     "transformers",
#     "trl",
#     "datasets",
#     "altair==6.1.0",
#     "polars==1.40.1",
#     "ipython==9.13.0",
#     "ipywidgets==8.1.8",
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
    # Qwen2.5-0.5B-Instruct full SFT — overfitting smoke test

    Imports `training-jobs/qwen_sft.py` and overfits on a small slice of
    `valid_move_smoke.jsonl`. Same prompt/completion format as the GPT-2
    SFT (`f"{fen}
    Move:"`) so the same harness/inference path works
    without code changes.

    On a T4 (compute 7.5) Qwen 0.5B trains in fp16 with grad checkpointing
    and per-device batch 2, eff batch 8 — ~10 epochs on 20 samples drops
    train loss into the 0.3–0.5 range.
    """)
    return


@app.cell
def _():
    import os
    import sys

    for _candidate in ("training-jobs", "final/training-jobs"):
        if os.path.isdir(_candidate) and _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    import inference as inference_mod
    import qwen_sft

    return inference_mod, qwen_sft


@app.cell
def _():
    from draughts import Board

    return (Board,)


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(mo):
    mo.md("""
    ## Train
    """)
    return


@app.cell
def _(mo):
    run_name = mo.ui.text(value="qwen-sft", label="Run name")
    epochs = mo.ui.slider(1, 50, value=10, label="Epochs")
    learning_rate = mo.ui.number(
        start=1e-6, stop=1e-3, value=5e-5, step=1e-5, label="Learning rate"
    )
    train_button = mo.ui.run_button(label="Train", kind="success")
    mo.vstack(
        [
            mo.hstack([run_name, learning_rate]),
            epochs,
            train_button,
        ]
    )
    return epochs, learning_rate, run_name, train_button


@app.cell
def _(epochs, learning_rate, qwen_sft, run_name, train_button):
    log_history = []
    last_saved_run = None
    if train_button.value:
        result = qwen_sft.train(
            run_name=run_name.value,
            epochs=epochs.value,
            max_samples=None,
            learning_rate=learning_rate.value,
            eval_steps=50,
        )
        log_history = result["log_history"]
        last_saved_run = result["output_dir"]
        print(f"done: {last_saved_run}")
    return last_saved_run, log_history


@app.cell
def _(log_history, mo):
    import altair as alt
    import polars as pl

    _loss_rows = []
    _metric_rows = []
    for _e in log_history:
        _step = _e.get("step")
        if _step is None:
            continue
        if "loss" in _e:
            _loss_rows.append({"step": _step, "value": _e["loss"], "split": "train"})
        if "eval_loss" in _e:
            _loss_rows.append(
                {"step": _step, "value": _e["eval_loss"], "split": "eval"}
            )
        if "eval_legal_move_rate" in _e:
            _metric_rows.append(
                {"step": _step, "value": _e["eval_legal_move_rate"]}
            )
    if not _loss_rows:
        loss_chart = mo.md("_train first to see the loss curve_")
    else:
        _loss_line = (
            alt.Chart(pl.DataFrame(_loss_rows))
            .mark_line(point=True)
            .encode(
                x=alt.X("step:Q", title="step"),
                y=alt.Y("value:Q", title="loss", scale=alt.Scale(zero=False)),
                color=alt.Color("split:N", title="split"),
                tooltip=["step", "value", "split"],
            )
            .properties(width=600, height=260, title="Qwen SFT loss")
        )
        _charts = [mo.ui.altair_chart(_loss_line)]
        if _metric_rows:
            _metric_line = (
                alt.Chart(pl.DataFrame(_metric_rows))
                .mark_line(point=True, color="#2ca02c")
                .encode(
                    x=alt.X("step:Q", title="step"),
                    y=alt.Y(
                        "value:Q",
                        title="legal-move rate",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=["step", "value"],
                )
                .properties(width=600, height=200, title="Eval legal-move rate")
            )
            _charts.append(mo.ui.altair_chart(_metric_line))
        loss_chart = mo.vstack(_charts)
    loss_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Load a checkpoint for inference
    """)
    return


@app.cell
def _(last_saved_run, mo):
    _ = last_saved_run
    refresh_button = mo.ui.run_button(label="Refresh checkpoint list")
    refresh_button
    return (refresh_button,)


@app.cell
def _(mo, refresh_button):
    import os as _os

    _ = refresh_button.value
    options = ["Qwen/Qwen2.5-0.5B-Instruct"]
    if _os.path.isdir("checkpoints"):
        for _name in sorted(_os.listdir("checkpoints")):
            _p = _os.path.join("checkpoints", _name)
            if _os.path.isdir(_p) and (
                _os.path.exists(_os.path.join(_p, "config.json"))
                or _os.path.exists(_os.path.join(_p, "adapter_config.json"))
            ):
                options.append(_p)

    checkpoint_choice = mo.ui.dropdown(
        options=options, value=options[-1], label="Checkpoint"
    )
    load_button = mo.ui.run_button(label="Load", kind="success")
    mo.hstack([checkpoint_choice, load_button])
    return checkpoint_choice, load_button


@app.cell
def _(checkpoint_choice, inference_mod, load_button):
    inference_model = None
    inference_tokenizer = None
    inference_path = None
    if load_button.value:
        inference_path = checkpoint_choice.value
        inference_model, inference_tokenizer = inference_mod.load_model_and_tokenizer(
            inference_path
        )
        print(f"loaded {inference_path}")
    return inference_model, inference_path, inference_tokenizer


@app.cell
def _(inference_model, inference_path, inference_tokenizer, torch):
    if inference_model is None or inference_tokenizer is None:
        make_agent = None
    else:
        from harness.hf_generate import make_hf_generate_fn
        from harness.llm_agent import LLMAgent

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _generate = make_hf_generate_fn(
            inference_model, inference_tokenizer, max_new_tokens=12, device=_device
        )

        def make_agent():
            return LLMAgent(generate_fn=_generate, record_turns=True)

        print(f"make_agent ready (model: {inference_path})")
    return (make_agent,)


@app.cell
def _(mo):
    mo.md("""
    ## Smoke test
    """)
    return


@app.cell
def _(mo):
    smoke_button = mo.ui.run_button(label="Run smoke test", kind="success")
    smoke_button
    return (smoke_button,)


@app.cell
def _(Board, make_agent, mo, smoke_button):
    mo.stop(
        not smoke_button.value, mo.md("_load a checkpoint, then click smoke test_")
    )
    mo.stop(make_agent is None, mo.md("_load a checkpoint first_"))

    _agent = make_agent()
    _board = Board()
    _move = _agent.select_move(_board)
    _turn = _agent.turns[-1] if _agent.turns else None

    smoke_summary = mo.md(f"""
    **chosen move:** `{_move}`

    **fen:** `{_turn.fen_before if _turn else 'n/a'}`

    **raw response:** `{_turn.responses[0] if _turn else 'n/a'!r}`

    **legal first try:** `{_turn.legal_first_try if _turn else 'n/a'}`

    **fell back to random:** `{_turn.fell_back_to_random if _turn else 'n/a'}`

    **stats:** `num_turns={_agent.num_turns}` · `first_try_legal={_agent.first_try_legal}` · `total_fallbacks={_agent.total_fallbacks}`
    """)
    smoke_summary
    return


if __name__ == "__main__":
    app.run()
