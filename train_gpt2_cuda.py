import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from draughts import Board

    return (Board,)


@app.cell
def _():
    import os
    from datasets import load_dataset

    _path = (
        "valid_move_dataset.jsonl"
        if os.path.exists("valid_move_dataset.jsonl")
        else "valid_move_smoke.jsonl"
    )
    print(f"loading {_path}")
    _raw = load_dataset("json", data_files=_path, split="train")

    def _to_sft(example):
        return {
            "prompt": f"{example['fen']}\nMove:",
            "completion": f" {example['move']}",
        }

    _formatted = _raw.map(_to_sft, remove_columns=_raw.column_names)
    _split = _formatted.train_test_split(test_size=0.1, seed=42)
    training_dataset = _split["train"]
    eval_dataset = _split["test"]
    print(training_dataset)
    print(eval_dataset)
    print(training_dataset[0])
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
def _(torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2 has no pad token; reuse eos so the trainer's collator can pad batches
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id
    if torch.cuda.is_available():
        gpt2_model.to("cuda")
    return gpt2_model, gpt2_tokenizer


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train GPT-2 on checkers SFT (slow)")
    train_button
    return (train_button,)


@app.cell
def _(
    bf16_ok,
    eval_dataset,
    gpt2_model,
    gpt2_tokenizer,
    torch,
    train_button,
    training_dataset,
):
    log_history = []
    trained_model = gpt2_model
    if train_button.value:
        from trl import SFTConfig, SFTTrainer

        gpu = torch.cuda.is_available()
        trainer = SFTTrainer(
            model=gpt2_model,
            processing_class=gpt2_tokenizer,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                output_dir="gpt2-checkers-sft",
                num_train_epochs=1,
                per_device_train_batch_size=16 if gpu else 4,
                per_device_eval_batch_size=16 if gpu else 4,
                gradient_accumulation_steps=1,
                bf16=gpu and bf16_ok,
                fp16=gpu and not bf16_ok,
                dataloader_pin_memory=gpu,
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
        trained_model = trainer.model
    return log_history, trained_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Heres our loss curves
    """)
    return


@app.cell
def _(log_history, mo):
    import altair as alt
    import polars as pl

    _rows = []
    for _e in log_history:
        if "loss" in _e:
            _rows.append({"step": _e["step"], "loss": _e["loss"], "split": "train"})
        if "eval_loss" in _e:
            _rows.append({"step": _e["step"], "loss": _e["eval_loss"], "split": "eval"})
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### And now inference: we got a harness up and running
    """)
    return


@app.cell
def _(Board, mo):
    get_board, set_board = mo.state(Board())
    get_gen_move, set_gen_move = mo.state("")
    return get_board, set_board, set_gen_move


@app.cell
def _(mo):
    HARDCODED_MOVES = ["31-27", "32-28", "33-29", "34-30", "35-30"]
    move_btns = [mo.ui.run_button(label=_lbl) for _lbl in HARDCODED_MOVES]
    reset_btn = mo.ui.run_button(label="Reset", kind="warn")
    gen_btn = mo.ui.run_button(label="Generate opponent move (model)")
    validate_btn = mo.ui.run_button(label="Validate and try move", kind="success")
    mo.vstack([
        mo.md("**your move (white):**"),
        mo.hstack(move_btns + [reset_btn]),
        mo.md("**opponent (model):**"),
        mo.hstack([gen_btn, validate_btn]),
    ])
    return HARDCODED_MOVES, gen_btn, move_btns, reset_btn


@app.cell
def _(get_board, mo):
    _b = get_board()
    _turn = _b.turn.name if hasattr(_b.turn, "name") else str(_b.turn)
    _legal = ", ".join(str(m) for m in list(_b.legal_moves)[:20])
    mo.md(f"""
    **turn:** `{_turn}` &nbsp;&nbsp; **legal (first 20):** `{_legal}`

    ```
    {_b}
    ```
    """)
    return


@app.cell
def _(Board, HARDCODED_MOVES, get_board, move_btns, reset_btn, set_board):
    import copy as _copy

    if reset_btn.value:
        set_board(Board())
    else:
        for _btn, _uci in zip(move_btns, HARDCODED_MOVES):
            if _btn.value:
                _b = _copy.deepcopy(get_board())
                _b.push_uci(_uci)
                set_board(_b)
                break
    return


@app.cell
def _(
    gen_btn,
    get_board,
    gpt2_tokenizer,
    mo,
    set_gen_move,
    torch,
    trained_model,
):
    mo.stop(not gen_btn.value, mo.md("_click 'Generate opponent move' to query the model_"))

    _b = get_board()
    _prompt = f"{_b.fen}\nMove:"
    _device = next(trained_model.parameters()).device
    _inputs = gpt2_tokenizer(_prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        _out = trained_model.generate(
            **_inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    _full = gpt2_tokenizer.decode(_out[0], skip_special_tokens=True)
    _completion = _full[len(_prompt):]
    _candidate = _completion.strip().split()[0] if _completion.strip() else ""
    set_gen_move(_candidate)
    print(f"prompt fen: {_b.fen}")
    print(f"raw completion: {_completion!r}")
    print(f"parsed candidate: {_candidate!r}")
    mo.md(f"**model suggests:** `{_candidate}` &nbsp; _(click 'Validate and try move' to apply)_")
    return


if __name__ == "__main__":
    app.run()
