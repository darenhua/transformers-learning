# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch==2.11.0",
#     "transformers",
#     "trl",
#     "peft",
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
    # Qwen DPO — overfitting smoke test

    Imports `training-jobs/qwen_dpo.py`. Loads `optimal_move_smoke.jsonl`
    (fen / chosen / rejected) and trains DPO on top of an existing SFT
    checkpoint at `checkpoints/qwen-sft/`.

    For overfitting on tiny data the notebook defaults to `--peft` mode:
    a small LoRA adapter trains on top of a frozen-base policy, with the
    base also serving as the implicit reference model. That avoids
    holding two full copies of Qwen on a T4.

    What to look for:

    - `loss` should drop from 0.69 (`ln 2`, the zero-margin baseline)
    - `rewards/margins` should grow positive
    - `rewards/accuracies` should approach 1 (chosen scored above rejected)
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
    import qwen_dpo

    return inference_mod, qwen_dpo


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

    `Base model` defaults to `checkpoints/qwen-sft/` — train that first via
    the Qwen SFT notebook, or set base to a HF model id (e.g.
    `Qwen/Qwen2.5-0.5B-Instruct`) to skip the prerequisite.
    """)
    return


@app.cell
def _(mo):
    run_name = mo.ui.text(value="qwen-dpo", label="Run name")
    base_model = mo.ui.text(value="checkpoints/qwen-sft", label="Base model")
    epochs = mo.ui.slider(1, 50, value=10, label="Epochs")
    beta = mo.ui.number(start=0.01, stop=2.0, value=0.5, step=0.05, label="Beta")
    learning_rate = mo.ui.number(
        start=1e-7, stop=1e-3, value=1e-5, step=1e-6, label="Learning rate"
    )
    use_peft = mo.ui.checkbox(value=True, label="Use LoRA adapter (--peft)")
    train_button = mo.ui.run_button(label="Train", kind="success")
    mo.vstack(
        [
            mo.hstack([run_name, base_model]),
            mo.hstack([learning_rate, beta, use_peft]),
            epochs,
            train_button,
        ]
    )
    return (
        base_model,
        beta,
        epochs,
        learning_rate,
        run_name,
        train_button,
        use_peft,
    )


@app.cell
def _(
    base_model,
    beta,
    epochs,
    learning_rate,
    qwen_dpo,
    run_name,
    train_button,
    use_peft,
):
    log_history = []
    last_saved_run = None
    if train_button.value:
        result = qwen_dpo.train(
            run_name=run_name.value,
            epochs=epochs.value,
            max_samples=None,
            base_model=base_model.value,
            learning_rate=learning_rate.value,
            beta=beta.value,
            use_peft=use_peft.value,
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
    _reward_rows = []
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
        if "rewards/chosen" in _e:
            _reward_rows.append(
                {
                    "step": _step,
                    "value": _e["rewards/chosen"],
                    "metric": "rewards/chosen",
                }
            )
        if "rewards/rejected" in _e:
            _reward_rows.append(
                {
                    "step": _step,
                    "value": _e["rewards/rejected"],
                    "metric": "rewards/rejected",
                }
            )
        if "rewards/margins" in _e:
            _reward_rows.append(
                {
                    "step": _step,
                    "value": _e["rewards/margins"],
                    "metric": "rewards/margins",
                }
            )
        if "eval_winrate" in _e:
            _metric_rows.append(
                {"step": _step, "value": _e["eval_winrate"]}
            )

    if not _loss_rows:
        chart = mo.md("_train first to see DPO curves_")
    else:
        _loss_chart = (
            alt.Chart(pl.DataFrame(_loss_rows))
            .mark_line(point=True)
            .encode(
                x=alt.X("step:Q", title="step"),
                y=alt.Y("value:Q", title="DPO loss", scale=alt.Scale(zero=False)),
                color=alt.Color("split:N", title="split"),
                tooltip=["step", "value", "split"],
            )
            .properties(width=600, height=240, title="DPO loss")
        )
        _reward_chart = (
            alt.Chart(pl.DataFrame(_reward_rows))
            .mark_line(point=True)
            .encode(
                x=alt.X("step:Q", title="step"),
                y=alt.Y("value:Q", title="reward", scale=alt.Scale(zero=False)),
                color=alt.Color("metric:N", title="metric"),
                tooltip=["step", "value", "metric"],
            )
            .properties(width=600, height=240, title="DPO rewards")
        )
        _charts = [mo.ui.altair_chart(_loss_chart), mo.ui.altair_chart(_reward_chart)]
        if _metric_rows:
            _winrate_chart = (
                alt.Chart(pl.DataFrame(_metric_rows))
                .mark_line(point=True, color="#2ca02c")
                .encode(
                    x=alt.X("step:Q", title="step"),
                    y=alt.Y(
                        "value:Q",
                        title="winrate (greedy == chosen)",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=["step", "value"],
                )
                .properties(width=600, height=200, title="Eval offline winrate")
            )
            _charts.append(mo.ui.altair_chart(_winrate_chart))
        chart = mo.vstack(_charts)
    chart
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
