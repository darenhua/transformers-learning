# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch==2.11.0",
#     "transformers",
#     "datasets==4.8.5",
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
    # Harness — valid-move benchmark

    Cheap per-checkpoint metric. For every row of a `valid_move_*.jsonl`
    test set, the model is asked once for a move at that FEN. The response
    is parsed and checked **for legality** — does the parsed move appear
    in `board.legal_moves` for that FEN? — and that's the score.

    No top-1 accuracy here. The dataset's `move` field is shown for
    eyeballing context only; we don't compare against it. (Two distinct
    moves can both be legal at the same position.) For winrate-based
    evaluation against baselines, use `harness_benchmark.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Register a trained checkpoint
    """)
    return


@app.cell
def _():
    from harness import registry  # noqa: F401  (kept for parity with other harnesses)

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
def _(checkpoint_choice, mo, register_button):
    # Loads the chosen checkpoint and exposes `llm_generate_fn` directly so
    # the eval can do a single forward pass per row (no LLMAgent retry /
    # fallback path — that would mask the model's true legality rate).
    registered_label = None
    llm_generate_fn = None
    if register_button.value:
        from transformers import (
            AutoModelForCausalLM as _AMC,
            AutoTokenizer as _AT,
        )
        import torch as _torch

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

        llm_generate_fn = make_hf_generate_fn(
            _model, _tok, max_new_tokens=12, device=_device
        )
        registered_label = f"llm_local ← `{_path}` ({_device})"

    badge = (
        mo.md(f"**registered:** {registered_label}")
        if registered_label
        else mo.md("_no llm_local registered yet_")
    )
    badge
    return (llm_generate_fn,)


@app.cell
def _(mo):
    mo.md("""
    ## Eval
    """)
    return


@app.cell
def _():
    # # Match the training data's board class. The valid_move dataset is
    # # produced from `Board()` (10x10 international); switch this if you
    # # generate an 8x8 American test set.
    # board_class_choice = mo.ui.radio(...)

    board_class_choice = "Board (10x10 international)"
    return (board_class_choice,)


@app.cell
def _(mo):
    test_dataset_path = mo.ui.text(
        value="valid_move_test.jsonl",
        label="Test dataset jsonl",
        full_width=True,
    )
    max_rows = mo.ui.slider(
        1, 5000, value=100, label="Max rows to eval (cap for CPU iteration)"
    )
    run = mo.ui.run_button(label="Run legality eval", kind="success")
    mo.hstack([test_dataset_path, max_rows, run])
    return max_rows, run, test_dataset_path


@app.cell
def _(board_class_choice, llm_generate_fn, max_rows, run, test_dataset_path):
    from draughts import Board as _Board, AmericanBoard as _AmericanBoard

    _board_cls = _Board if "Board" in board_class_choice else _AmericanBoard

    test_result = None
    test_rows = []
    error_msg = None

    if run.value:
        if llm_generate_fn is None:
            error_msg = (
                "Legality eval requires a registered checkpoint. "
                "Click **Register llm_local** above first."
            )
        else:
            import os as _os
            from datasets import load_dataset as _load_dataset

            from harness.scoring import parse_move

            _candidates = [
                test_dataset_path.value,
                f"../{test_dataset_path.value}",
            ]
            _resolved = next(
                (p for p in _candidates if _os.path.exists(p)), None
            )
            if _resolved is None:
                error_msg = (
                    f"none of {_candidates} exist — point at a "
                    f"valid_move jsonl with `fen` + `move` columns"
                )
            else:
                _ds = _load_dataset(
                    "json", data_files=_resolved, split="train"
                )
                _legal = 0
                _n_capped = min(max_rows.value, len(_ds))
                for _i in range(_n_capped):
                    _row = _ds[_i]
                    _fen = _row["fen"]
                    _ref = _row.get("move", "")
                    _b = _board_cls.from_fen(_fen)
                    _legal_moves = list(_b.legal_moves)
                    _prompt = f"{_fen}\nMove:"
                    _resp = llm_generate_fn(_prompt)
                    _m = parse_move(_resp, _legal_moves)
                    _is_legal = _m is not None
                    _legal += int(_is_legal)
                    test_rows.append(
                        {
                            "row": _i + 1,
                            "fen": _fen[:60],
                            "dataset_move": _ref,
                            "raw_response": _resp[:60],
                            "parsed": str(_m) if _m else "",
                            "legal": _is_legal,
                        }
                    )
                _n = len(test_rows)
                test_result = {
                    "rows_scored": _n,
                    "rows_total": len(_ds),
                    "legal_count": _legal,
                    "legal_rate": _legal / _n if _n else 0.0,
                    "dataset": _resolved,
                }
    return error_msg, test_result, test_rows


@app.cell
def _(board_class_choice, error_msg, mo, test_result, test_rows):
    if error_msg is not None:
        view = mo.md(f"⚠️ **{error_msg}**")
    elif test_result is None:
        view = mo.md("_click **Run legality eval** to start_")
    else:
        _lines = [
            f"### legality eval",
            f"_{test_result['rows_scored']}/{test_result['rows_total']} rows · "
            f"`{test_result['dataset']}` · `{board_class_choice}`_",
            "",
            f"- **legal-move rate:** {test_result['legal_rate']:.1%} "
            f"({test_result['legal_count']}/{test_result['rows_scored']})",
        ]
        _summary = mo.md("\n".join(_lines))
        view = mo.vstack(
            [
                _summary,
                mo.md(f"### Per-row scoring"),
                mo.ui.table(test_rows),
            ]
        )
    view
    return


if __name__ == "__main__":
    app.run()
