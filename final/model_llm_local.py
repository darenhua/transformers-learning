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
    # gpt-2 SFT — train + load + smoke-test

    Training runs save to `checkpoints/<run_name>/`. The inference section
    below has a dropdown that scans that folder so you can flip between any
    saved run (or the un-tuned base `gpt2`) without restarting the kernel.

    Board class is `draughts.Board` (10x10 International) to match the
    `valid_move_smoke.jsonl` dataset. If you switch to American 8x8 data,
    swap to `AmericanBoard` here AND pass `board_class=AmericanBoard` to
    `run_benchmark` later.
    """)
    return


@app.cell
def _():
    from draughts import Board

    return (Board,)


@app.cell
def _(mo):
    dataset_path = mo.ui.text(
        value="valid_move_smoke.jsonl",
        label="Dataset jsonl path",
        full_width=True,
    )
    dataset_path
    return (dataset_path,)


@app.cell
def _(dataset_path):
    import os as _os
    from datasets import load_dataset as _load_dataset

    # Try the literal path then `../<path>`, so the same notebook works
    # whether marimo was launched from the project root or from final/.
    _candidates = [dataset_path.value, f"../{dataset_path.value}"]
    _resolved = next((p for p in _candidates if _os.path.exists(p)), None)
    if _resolved is None:
        raise FileNotFoundError(f"none of {_candidates} exist")
    print(f"loading {_resolved}")
    _raw = _load_dataset("json", data_files=_resolved, split="train").select(range(10))

    def _to_sft(example):
        return {
            "prompt": f"{example['fen']}\nMove:",
            "completion": f" {example['move']}",
        }

    _formatted = _raw.map(_to_sft, remove_columns=_raw.column_names)
    _split = _formatted.train_test_split(test_size=0.2, seed=42)
    training_dataset = _split["train"]
    eval_dataset = _split["test"]
    print(training_dataset)
    print(eval_dataset)
    print("first row:", training_dataset[0])
    return eval_dataset, training_dataset


@app.cell
def _():
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Native bf16 tensor-core support starts on Ampere (compute capability 8.0).
        # On Turing (T4, 7.5) bf16 only works via emulation — slower than fp16.
        bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
        print(f"CUDA on: {gpu_name} — using {'bf16' if bf16_ok else 'fp16'}")
    else:
        bf16_ok = False
        print("CUDA not available — training will fall back to CPU")
    return bf16_ok, torch


@app.cell
def _(mo):
    mo.md("""
    ## Train
    """)
    return


@app.cell
def _(mo):
    run_name = mo.ui.text(
        value="sft-run-1",
        label="Run name",
    )
    epochs = mo.ui.slider(1, 10, value=1, label="Epochs")
    train_button = mo.ui.run_button(
        label="Train (saves to checkpoints/<run_name>/)", kind="success"
    )
    mo.hstack([run_name, epochs, train_button])
    return epochs, run_name, train_button


@app.cell
def _(
    bf16_ok,
    epochs,
    eval_dataset,
    run_name,
    torch,
    train_button,
    training_dataset,
):
    log_history = []
    last_saved_run = None
    if train_button.value:
        from transformers import AutoModelForCausalLM as _AMC, AutoTokenizer as _AT
        from trl import SFTConfig as _SFTConfig, SFTTrainer as _SFTTrainer

        # Load a fresh base model per run — training mutates weights, so
        # holding one base across the cell would silently keep training the
        # same model instead of starting from gpt-2.
        _tok = _AT.from_pretrained("gpt2")
        _tok.pad_token = _tok.eos_token
        _model = _AMC.from_pretrained("gpt2")
        _model.config.pad_token_id = _tok.eos_token_id
        if torch.cuda.is_available():
            _model.to("cuda")

        _output_dir = f"checkpoints/{run_name.value}"
        _gpu = torch.cuda.is_available()
        trainer = _SFTTrainer(
            model=_model,
            processing_class=_tok,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            args=_SFTConfig(
                output_dir=_output_dir,
                num_train_epochs=epochs.value,
                per_device_train_batch_size=16 if _gpu else 4,
                per_device_eval_batch_size=16 if _gpu else 4,
                gradient_accumulation_steps=1,
                bf16=_gpu and bf16_ok,
                fp16=_gpu and not bf16_ok,
                dataloader_pin_memory=_gpu,
                logging_steps=1,
                logging_first_step=True,
                eval_strategy="steps",
                eval_steps=20,
                save_strategy="epoch",
                report_to="none",
            ),
        )
        trainer.train()
        log_history = list(trainer.state.log_history)
        # Save the final state at output_dir root (TRL's checkpoint-N dirs are
        # mid-training snapshots; we want a clean dir to load from below).
        trainer.save_model(_output_dir)
        _tok.save_pretrained(_output_dir)
        last_saved_run = _output_dir
        print(f"saved final model + tokenizer to {_output_dir}")
    return last_saved_run, log_history


@app.cell
def _(log_history, mo):
    import altair as alt
    import polars as pl

    _rows = []
    for _e in log_history:
        if "loss" in _e:
            _rows.append({"step": _e["step"], "loss": _e["loss"], "split": "train"})
        if "eval_loss" in _e:
            _rows.append(
                {"step": _e["step"], "loss": _e["eval_loss"], "split": "eval"}
            )
    if not _rows:
        loss_chart = mo.md("_train first to see the loss curve_")
    else:
        _df = pl.DataFrame(_rows)
        _line = (
            alt.Chart(_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("step:Q", title="step"),
                y=alt.Y("loss:Q", title="loss", scale=alt.Scale(zero=False)),
                color=alt.Color("split:N", title="split"),
                tooltip=["step", "loss", "split"],
            )
            .properties(width=600, height=300, title="SFT loss (train vs. eval)")
        )
        loss_chart = mo.ui.altair_chart(_line)
    loss_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Load a checkpoint for inference

    The dropdown lists every subdir under `checkpoints/` that looks like a
    saved HF model (has `config.json`), plus the un-tuned `gpt2` baseline.
    Hit **Refresh** after a new training run completes; **Load** materializes
    the selected one as `inference_model` for the smoke test below.
    """)
    return


@app.cell
def _(last_saved_run, mo):
    # Depend on `last_saved_run` so this cell re-runs after a new training
    # finishes — but we don't actually use the value, just trigger the refresh.
    _ = last_saved_run

    refresh_button = mo.ui.run_button(label="Refresh checkpoint list")
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
    load_button = mo.ui.run_button(label="Load", kind="success")
    mo.hstack([checkpoint_choice, load_button])
    return checkpoint_choice, load_button


@app.cell
def _(checkpoint_choice, load_button, torch):
    inference_model = None
    inference_tokenizer = None
    inference_path = None
    if load_button.value:
        from transformers import (
            AutoModelForCausalLM as _AMC,
            AutoTokenizer as _AT,
        )

        inference_path = checkpoint_choice.value
        inference_tokenizer = _AT.from_pretrained(inference_path)
        if inference_tokenizer.pad_token is None:
            inference_tokenizer.pad_token = inference_tokenizer.eos_token
        inference_model = _AMC.from_pretrained(inference_path)
        inference_model.config.pad_token_id = inference_tokenizer.eos_token_id
        if torch.cuda.is_available():
            inference_model.to("cuda")
        inference_model.eval()
        print(
            f"loaded {inference_path} on {next(inference_model.parameters()).device}"
        )
    return inference_model, inference_path, inference_tokenizer


@app.cell
def _(inference_model, inference_path, inference_tokenizer, torch):
    if inference_model is None or inference_tokenizer is None:
        make_agent = None
    else:
        from harness.llm_agent import LLMAgent
        from harness.hf_generate import make_hf_generate_fn

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _generate = make_hf_generate_fn(
            inference_model,
            inference_tokenizer,
            max_new_tokens=12,
            device=_device,
        )

        def make_agent():
            """Build a fresh LLMAgent over the currently loaded inference_model."""
            return LLMAgent(generate_fn=_generate, record_turns=True)

        print(f"make_agent ready (model: {inference_path})")
    return (make_agent,)


@app.cell
def _(mo):
    mo.md("""
    ## Smoke test

    Builds a fresh `LLMAgent` and asks it for one move on the initial board.
    A trained model produces something resembling a move (`34-29`, `32-28`).
    A bare `gpt2` produces gibberish — the agent's random fallback still
    returns a legal move so the harness keeps working.
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
