# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
#     "torch==2.11.0",
#     "transformers",
#     "peft",
#     "datasets==4.8.5",
#     "altair==6.1.0",
#     "polars==1.40.1",
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
    mo.md(r"""
    # Checkers training results — interactive review

    Two models came out of the pipeline:

    1. **QLoRA (SFT)** — Qwen2.5-0.5B-Instruct, 4-bit base + LoRA r=256 adapter,
       trained on **20k diverse self-play positions** (30/40/30 opening / midgame /
       endgame split). Final eval-loss **0.42**, callback-reported legal-move rate
       **0.69** (16-sample noisy estimate at end of training).

    2. **DPO** — same Qwen base with the QLoRA adapter merged in, then a
       second LoRA adapter trained via DPO on **5k preference pairs** (Scan
       strong vs weak). Final DPO-loss **0.59**, reward margins ~1.5–1.9. On a
       200-sample post-hoc eval: **legal-move rate 0.83** and **chosen-match 14%**
       — DPO actually *raised* the legal-move rate over the QLoRA base.

    This notebook lets you:

    - Load any checkpoint (baseline, QLoRA, QLoRA-merged, DPO)
    - Try one-shot inference on a position you pick
    - Run **legal-move rate** eval (parse-into-board.legal_moves) — same as the
      QLoRA training callback but with N you control
    - Run **chosen-match (offline winrate)** eval (greedy output equals `chosen`)
      — same as the DPO training callback

    > Run marimo from inside `final/` so the relative paths in `adapter_config.json`
    > (`base_model_name_or_path = "checkpoints/qwen-qlora-merged"`) resolve correctly.
    """)
    return


@app.cell
def _():
    import os
    import sys

    for _root in (".", "final"):
        if os.path.isdir(_root) and _root not in sys.path:
            sys.path.insert(0, _root)
    for _root in ("training-jobs", "final/training-jobs"):
        if os.path.isdir(_root) and _root not in sys.path:
            sys.path.insert(0, _root)

    import inference as inference_mod
    from _data import sample_eval_rows
    # classify_phase / count_pieces let the eval charts split outcomes by
    # game phase using the same thresholds the dataset generator used.
    from _dataset_worker import classify_phase, count_pieces

    return inference_mod, sample_eval_rows


@app.cell
def _():
    from draughts import Board

    from harness.scoring import is_legal, parse_move

    return Board, parse_move


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pick a checkpoint

    Available checkpoints from the pipeline:

    - `Qwen/Qwen2.5-0.5B-Instruct` — bare baseline, no checkers training
    - `checkpoints/qwen-qlora` — LoRA adapter from SFT on 20k positions
    - `checkpoints/qwen-qlora-merged` — same as above with the adapter folded
      into base weights (full model, no PEFT layer)
    - `checkpoints/qwen-dpo` — DPO LoRA adapter on top of qwen-qlora-merged

    Click **Load** to materialize the model in VRAM. Loading a fresh checkpoint
    while another is loaded will replace it — `torch.cuda.empty_cache()` runs
    after the swap so VRAM doesn't accumulate.
    """)
    return


@app.cell
def _(mo):
    import os as _os

    options = ["Qwen/Qwen2.5-0.5B-Instruct"]
    if _os.path.isdir("checkpoints"):
        for _name in sorted(_os.listdir("checkpoints")):
            _p = _os.path.join("checkpoints", _name)
            if _os.path.isdir(_p) and (
                _os.path.exists(_os.path.join(_p, "config.json"))
                or _os.path.exists(_os.path.join(_p, "adapter_config.json"))
            ):
                options.append(_p)

    _default = (
        "checkpoints/qwen-dpo"
        if "checkpoints/qwen-dpo" in options
        else options[-1]
    )
    checkpoint_choice = mo.ui.dropdown(
        options=options, value=_default, label="Checkpoint",
    )
    load_button = mo.ui.run_button(label="Load", kind="success")
    mo.hstack([checkpoint_choice, load_button])
    return checkpoint_choice, load_button


@app.cell
def _(checkpoint_choice, inference_mod, load_button, mo, torch):
    mo.stop(
        not load_button.value, mo.md("_pick a checkpoint above, then click Load_")
    )

    # Free VRAM from any previously-loaded model before constructing the new
    # one so swapping checkpoints doesn't accumulate state.
    import gc as _gc

    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inference_path = checkpoint_choice.value
    inference_model, inference_tokenizer = inference_mod.load_model_and_tokenizer(
        inference_path
    )
    print(f"loaded {inference_path}")
    return inference_model, inference_path, inference_tokenizer


@app.cell
def _(inference_model, inference_path, inference_tokenizer, torch):
    from harness.hf_generate import make_hf_generate_fn

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_fn = make_hf_generate_fn(
        inference_model, inference_tokenizer, max_new_tokens=12, device=_device
    )
    print(f"generate_fn ready (model={inference_path}, device={_device})")
    return (generate_fn,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Training curves

    Both training runs wrote their full log history to
    `checkpoints/<run>/checkpoint-<step>/trainer_state.json`. The plots below
    read those files directly — no model load needed.

    For QLoRA you'll see:
    - **train loss** logged every step
    - **eval loss** logged at every `eval_steps` interval
    - **eval_legal_move_rate** from the generation callback
      (16-sample noisy estimate, but the trend is still informative)

    For DPO you'll see:
    - **DPO loss** (trains toward the chosen-vs-rejected margin) — should drop
      below `ln 2 ≈ 0.693`, the zero-margin baseline
    - **rewards/margins** — `β · (logπ(chosen) − logπ(rejected))`; should grow
      positive, meaning chosen scores higher than rejected under the policy
    - **rewards/accuracies** — fraction of pairs where chosen out-scores
      rejected; should approach 1
    - **eval_winrate** — fraction of greedy outputs that exactly match `chosen`
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    def _load_history(run_dir):
        """Return the log_history list from the latest checkpoint-N dir
        under run_dir, or None if nothing's there yet."""
        p = Path(run_dir)
        if not p.exists():
            return None
        ckpts = [
            d for d in p.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if not ckpts:
            return None
        latest = max(ckpts, key=lambda d: int(d.name.split("-")[1]))
        state = latest / "trainer_state.json"
        if not state.exists():
            return None
        return json.load(open(state))["log_history"]

    qlora_history = _load_history("checkpoints/qwen-qlora")
    dpo_history = _load_history("checkpoints/qwen-dpo")
    print(
        f"qlora: {len(qlora_history) if qlora_history else 0} log entries; "
        f"dpo: {len(dpo_history) if dpo_history else 0} log entries"
    )
    return dpo_history, qlora_history


@app.cell
def _(mo, qlora_history):
    import altair as _alt
    import polars as _pl

    if not qlora_history:
        qlora_charts = mo.md("_no qlora trainer_state.json found_")
    else:
        _loss_rows = []
        _metric_rows = []
        for _e in qlora_history:
            _step = _e.get("step")
            if _step is None:
                continue
            if "loss" in _e:
                _loss_rows.append(
                    {"step": _step, "value": _e["loss"], "split": "train"}
                )
            if "eval_loss" in _e:
                _loss_rows.append(
                    {"step": _step, "value": _e["eval_loss"], "split": "eval"}
                )
            if "eval_legal_move_rate" in _e:
                _metric_rows.append(
                    {"step": _step, "value": _e["eval_legal_move_rate"]}
                )

        _loss_chart = (
            _alt.Chart(_pl.DataFrame(_loss_rows))
            .mark_line(point=False)
            .encode(
                x=_alt.X("step:Q", title="step"),
                y=_alt.Y("value:Q", title="loss", scale=_alt.Scale(zero=False)),
                color=_alt.Color(
                    "split:N",
                    scale=_alt.Scale(
                        domain=["train", "eval"], range=["#1f77b4", "#ff7f0e"]
                    ),
                    legend=_alt.Legend(title=None),
                ),
                tooltip=["step", "value", "split"],
            )
            .properties(width=600, height=260, title="QLoRA — train vs eval loss")
        )
        _metric_chart = (
            _alt.Chart(_pl.DataFrame(_metric_rows))
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=_alt.X("step:Q", title="step"),
                y=_alt.Y(
                    "value:Q",
                    title="legal-move rate (n=16 callback)",
                    scale=_alt.Scale(domain=[0, 1]),
                ),
                tooltip=["step", "value"],
            )
            .properties(
                width=600, height=200,
                title="QLoRA — eval legal-move rate (16-sample noisy)",
            )
        )
        qlora_charts = mo.vstack(
            [
                mo.md("### QLoRA"),
                mo.ui.altair_chart(_loss_chart),
                mo.ui.altair_chart(_metric_chart),
            ]
        )
    qlora_charts
    return


@app.cell
def _(dpo_history, mo):
    import altair as _alt
    import polars as _pl

    if not dpo_history:
        dpo_charts = mo.md("_no dpo trainer_state.json found_")
    else:
        _loss_rows = []
        _reward_rows = []
        _metric_rows = []
        for _e in dpo_history:
            _step = _e.get("step")
            if _step is None:
                continue
            if "loss" in _e:
                _loss_rows.append(
                    {"step": _step, "value": _e["loss"], "split": "train"}
                )
            if "eval_loss" in _e:
                _loss_rows.append(
                    {"step": _step, "value": _e["eval_loss"], "split": "eval"}
                )
            for _k in ("rewards/chosen", "rewards/rejected", "rewards/margins"):
                if _k in _e:
                    _reward_rows.append(
                        {"step": _step, "value": _e[_k], "metric": _k}
                    )
            if "rewards/accuracies" in _e:
                _reward_rows.append(
                    {
                        "step": _step,
                        "value": _e["rewards/accuracies"],
                        "metric": "rewards/accuracies",
                    }
                )
            if "eval_winrate" in _e:
                _metric_rows.append(
                    {"step": _step, "value": _e["eval_winrate"]}
                )

        _loss_chart = (
            _alt.Chart(_pl.DataFrame(_loss_rows))
            .mark_line(point=False)
            .encode(
                x=_alt.X("step:Q", title="step"),
                y=_alt.Y(
                    "value:Q", title="DPO loss", scale=_alt.Scale(zero=False)
                ),
                color=_alt.Color(
                    "split:N",
                    scale=_alt.Scale(
                        domain=["train", "eval"], range=["#1f77b4", "#ff7f0e"]
                    ),
                    legend=_alt.Legend(title=None),
                ),
                tooltip=["step", "value", "split"],
            )
            .properties(
                width=600, height=240,
                title="DPO — loss (baseline ln 2 ≈ 0.693)",
            )
        )
        _reward_chart = (
            _alt.Chart(_pl.DataFrame(_reward_rows))
            .mark_line(point=False)
            .encode(
                x=_alt.X("step:Q", title="step"),
                y=_alt.Y(
                    "value:Q", title="reward", scale=_alt.Scale(zero=False)
                ),
                color=_alt.Color("metric:N", legend=_alt.Legend(title=None)),
                tooltip=["step", "value", "metric"],
            )
            .properties(
                width=600, height=240,
                title="DPO — chosen vs rejected reward + margins + accuracies",
            )
        )
        _metric_chart = (
            _alt.Chart(_pl.DataFrame(_metric_rows))
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=_alt.X("step:Q", title="step"),
                y=_alt.Y(
                    "value:Q",
                    title="winrate (greedy == chosen)",
                    scale=_alt.Scale(domain=[0, 1]),
                ),
                tooltip=["step", "value"],
            )
            .properties(
                width=600, height=200,
                title="DPO — eval offline winrate (16-sample noisy)",
            )
        )
        dpo_charts = mo.vstack(
            [
                mo.md("### DPO"),
                mo.ui.altair_chart(_loss_chart),
                mo.ui.altair_chart(_reward_chart),
                mo.ui.altair_chart(_metric_chart),
            ]
        )
    dpo_charts
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. One-shot inference

    Pick a position and see what the loaded model produces as its greedy
    continuation. Three sources:

    - **starting position** — `Board()`, the standard 10x10 international opening
    - **test-set sample (random)** — pulls a single FEN from
      `valid_move_train.jsonl`'s held-out 10% split (deterministic by seed),
      so you can compare the model's move to the dataset's reference move
    - **custom FEN** — paste any valid pydraughts FEN

    The prompt format matches training exactly: `f"{fen}\nMove:"`. The model's
    greedy output is parsed via the harness regex `\d+[\-x]\d+` and checked
    for membership in `board.legal_moves`.
    """)
    return


@app.cell
def _(mo):
    position_source = mo.ui.radio(
        options=["starting position", "test-set sample (random)", "custom FEN"],
        value="starting position",
        label="Position source",
    )
    custom_fen = mo.ui.text(
        value='[FEN "W:W:W31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50:B1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"]',
        label="Custom FEN (used when source = custom FEN)",
        full_width=True,
    )
    sample_seed = mo.ui.number(
        value=42, start=0, stop=2**31 - 1, label="Sample seed (test-set source)",
    )
    inference_button = mo.ui.run_button(label="Generate", kind="success")
    mo.vstack(
        [
            position_source,
            custom_fen,
            mo.hstack([sample_seed, inference_button]),
        ]
    )
    return custom_fen, inference_button, position_source, sample_seed


@app.cell
def _(
    Board,
    custom_fen,
    generate_fn,
    inference_button,
    mo,
    parse_move,
    position_source,
    sample_eval_rows,
    sample_seed,
):
    mo.stop(
        not inference_button.value,
        mo.md("_load a checkpoint, then click Generate_"),
    )

    if position_source.value == "starting position":
        _board = Board()
        _ref = None
    elif position_source.value == "custom FEN":
        _board = Board.from_fen(custom_fen.value)
        _ref = None
    else:
        # test-set sample: deterministic 1-row pull from the held-out split
        _rows = sample_eval_rows(
            "valid_move_train.jsonl",
            n_samples=1,
            test_size=0.1,
            seed=int(sample_seed.value),
        )
        _board = Board.from_fen(_rows[0]["fen"])
        _ref = _rows[0].get("move")

    _legal = list(_board.legal_moves)
    _prompt = f"{_board.fen}\nMove:"
    _response = generate_fn(_prompt)
    _parsed = parse_move(_response, _legal)
    _is_legal = _parsed is not None
    _matches_ref = _ref is not None and _parsed is not None and str(_parsed) == _ref

    inference_result = mo.md(f"""
    **fen:** `{_board.fen}`

    **prompt:** `{_prompt!r}`

    **raw response:** `{_response!r}`

    **parsed move:** `{_parsed}` &nbsp;&nbsp; **legal:** `{_is_legal}`

    **legal moves at this position ({len(_legal)}):** {", ".join(str(m) for m in _legal)}

    **dataset reference move:** `{_ref or "n/a"}` &nbsp;&nbsp; **matches:** `{_matches_ref}`
    """)
    inference_result
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Legal-move rate (SFT/QLoRA-style eval)

    For each held-out position from `valid_move_train.jsonl`, ask the model
    for a move and check whether `parse_move` finds it in `board.legal_moves`.
    This is the metric the QLoRA training callback was tracking every 100 steps.

    Higher N → tighter estimate. Std-error at p=0.7, N=16 is ±11%; at N=200
    it drops to ±3%. The training callback used N=16 to keep eval cheap during
    training; here you can crank it up.
    """)
    return


@app.cell
def _(mo):
    legal_n = mo.ui.slider(
        start=8, stop=500, value=100, step=8, label="N held-out positions",
    )
    legal_seed = mo.ui.number(
        value=42, start=0, stop=2**31 - 1, label="Sampling seed",
    )
    legal_button = mo.ui.run_button(
        label="Run legal-move-rate eval", kind="success",
    )
    mo.hstack([legal_n, legal_seed, legal_button])
    return legal_button, legal_n, legal_seed


@app.cell
def _(
    Board,
    generate_fn,
    legal_button,
    legal_n,
    legal_seed,
    mo,
    parse_move,
    sample_eval_rows,
):
    import sys as _sys

    _sys.path.insert(0, ".")
    from _dataset_worker import (
        ENDGAME_THRESHOLD as _ET,
        OPENING_THRESHOLD as _OT,
        classify_phase as _classify,
        count_pieces as _count,
    )

    mo.stop(
        not legal_button.value,
        mo.md("_load a checkpoint + set N, then click Run_"),
    )

    _rows = sample_eval_rows(
        "valid_move_train.jsonl",
        n_samples=int(legal_n.value),
        test_size=0.1,
        seed=int(legal_seed.value),
    )
    _legal_count = 0
    legal_table_rows = []
    for _i, _r in enumerate(_rows):
        _board = Board.from_fen(_r["fen"])
        _legal_moves = list(_board.legal_moves)
        _resp = generate_fn(f"{_r['fen']}\nMove:")
        _move = parse_move(_resp, _legal_moves)
        _is_legal = _move is not None
        _legal_count += int(_is_legal)
        _phase = _classify(_count(_board)[0], _OT, _ET)
        legal_table_rows.append(
            {
                "row": _i + 1,
                "phase": _phase,
                "fen": _r["fen"][:60],
                "dataset_move": _r.get("move", ""),
                "raw_response": _resp[:60],
                "parsed": str(_move) if _move else "",
                "legal": _is_legal,
            }
        )

    _n = len(_rows)
    _rate = _legal_count / _n
    _se = (_rate * (1 - _rate) / _n) ** 0.5
    legal_summary = mo.md(f"""
    **legal-move rate:** `{_legal_count}/{_n}` = **`{100 * _rate:.1f}%`**

    **standard error** at p={_rate:.2f}, N={_n}: ±`{100 * _se:.1f}%`
    """)

    import altair as _alt
    import polars as _pl

    # Donut: overall legal vs illegal
    _donut_df = _pl.DataFrame(
        {
            "outcome": ["legal", "illegal"],
            "count": [_legal_count, _n - _legal_count],
        }
    )
    _donut = (
        _alt.Chart(_donut_df)
        .mark_arc(innerRadius=70, outerRadius=130)
        .encode(
            theta=_alt.Theta("count:Q"),
            color=_alt.Color(
                "outcome:N",
                scale=_alt.Scale(
                    domain=["legal", "illegal"], range=["#2ca02c", "#d62728"]
                ),
                legend=_alt.Legend(title=None),
            ),
            tooltip=["outcome", "count"],
        )
        .properties(
            width=320,
            height=320,
            title=f"Legal-move rate: {100 * _rate:.1f}% ({_legal_count}/{_n})",
        )
    )

    # By-phase grouped bar: legal-rate per opening / midgame / endgame
    _phase_rows = []
    for _phase_name in ("opening", "midgame", "endgame"):
        _bucket = [r for r in legal_table_rows if r["phase"] == _phase_name]
        if not _bucket:
            continue
        _bn = len(_bucket)
        _bl = sum(1 for r in _bucket if r["legal"])
        _phase_rows.append(
            {"phase": _phase_name, "outcome": "legal", "count": _bl, "rate": _bl / _bn, "n": _bn}
        )
        _phase_rows.append(
            {
                "phase": _phase_name,
                "outcome": "illegal",
                "count": _bn - _bl,
                "rate": (_bn - _bl) / _bn,
                "n": _bn,
            }
        )
    _phase_df = _pl.DataFrame(_phase_rows)
    _phase_chart = (
        _alt.Chart(_phase_df)
        .mark_bar()
        .encode(
            x=_alt.X(
                "phase:N",
                sort=["opening", "midgame", "endgame"],
                title="game phase (by piece count)",
            ),
            y=_alt.Y("rate:Q", title="fraction", scale=_alt.Scale(domain=[0, 1])),
            color=_alt.Color(
                "outcome:N",
                scale=_alt.Scale(
                    domain=["legal", "illegal"], range=["#2ca02c", "#d62728"]
                ),
                legend=_alt.Legend(title=None),
            ),
            tooltip=["phase", "outcome", "count", "n", _alt.Tooltip("rate:Q", format=".1%")],
        )
        .properties(width=420, height=300, title="Legal-move rate by game phase")
    )

    legal_result = mo.vstack(
        [
            legal_summary,
            mo.hstack(
                [mo.ui.altair_chart(_donut), mo.ui.altair_chart(_phase_chart)]
            ),
            mo.md("### Per-row scoring"),
            mo.ui.table(legal_table_rows),
        ]
    )
    legal_result
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Chosen-match (DPO-style offline winrate)

    For each held-out preference triple `{fen, chosen, rejected}` from
    `optimal_move_train.jsonl`, generate the model's greedy output and check
    whether it equals the `chosen` move. This is the offline winrate the DPO
    training callback was tracking.

    Strict metric — many positions have 5–15 legal moves so even a perfect
    model only matches `chosen` exactly some fraction of the time. This view
    is most informative as a *delta* between checkpoints (e.g. DPO vs QLoRA),
    not as an absolute number.

    The cell also reports legal-move rate on the same set so you can sanity-
    check that the model still produces legal moves at all.
    """)
    return


@app.cell
def _(mo):
    winrate_n = mo.ui.slider(
        start=8, stop=500, value=100, step=8, label="N held-out triples",
    )
    winrate_seed = mo.ui.number(
        value=42, start=0, stop=2**31 - 1, label="Sampling seed",
    )
    winrate_button = mo.ui.run_button(
        label="Run chosen-match eval", kind="success",
    )
    mo.hstack([winrate_n, winrate_seed, winrate_button])
    return winrate_button, winrate_n, winrate_seed


@app.cell
def _(
    Board,
    generate_fn,
    mo,
    parse_move,
    sample_eval_rows,
    winrate_button,
    winrate_n,
    winrate_seed,
):
    import sys as _sys

    _sys.path.insert(0, ".")
    from _dataset_worker import (
        ENDGAME_THRESHOLD as _ET,
        OPENING_THRESHOLD as _OT,
        classify_phase as _classify,
        count_pieces as _count,
    )

    mo.stop(
        not winrate_button.value,
        mo.md("_load a checkpoint + set N, then click Run_"),
    )

    _rows = sample_eval_rows(
        "optimal_move_train.jsonl",
        n_samples=int(winrate_n.value),
        test_size=0.1,
        seed=int(winrate_seed.value),
    )
    # 4-category bucketing per row gives the most informative DPO view:
    # how many outputs land on chosen vs rejected vs some-other-legal vs
    # an illegal string. With low-beta DPO the policy often suppresses
    # rejected without lifting chosen, so output mass shifts to "other".
    _chosen_count = 0
    _rejected_count = 0
    _other_legal_count = 0
    _illegal_count = 0
    winrate_table_rows = []
    for _i, _r in enumerate(_rows):
        _board = Board.from_fen(_r["fen"])
        _legal_moves = list(_board.legal_moves)
        _resp = generate_fn(f"{_r['fen']}\nMove:")
        _move = parse_move(_resp, _legal_moves)
        _is_legal = _move is not None
        _ms = str(_move) if _move else ""
        if not _is_legal:
            _category = "illegal"
            _illegal_count += 1
        elif _ms == _r["chosen"]:
            _category = "matches chosen"
            _chosen_count += 1
        elif _ms == _r["rejected"]:
            _category = "matches rejected"
            _rejected_count += 1
        else:
            _category = "other legal"
            _other_legal_count += 1
        _phase = _classify(_count(_board)[0], _OT, _ET)
        winrate_table_rows.append(
            {
                "row": _i + 1,
                "phase": _phase,
                "fen": _r["fen"][:60],
                "chosen": _r.get("chosen", ""),
                "rejected": _r.get("rejected", ""),
                "raw_response": _resp[:60],
                "parsed": _ms,
                "category": _category,
                "legal": _is_legal,
                "matches_chosen": _ms == _r["chosen"],
            }
        )

    _n = len(_rows)
    _legal_count = _chosen_count + _rejected_count + _other_legal_count
    _chosen_rate = _chosen_count / _n
    _legal_rate = _legal_count / _n
    _se = (_chosen_rate * (1 - _chosen_rate) / _n) ** 0.5
    winrate_summary = mo.md(f"""
    **chosen-match (winrate):** `{_chosen_count}/{_n}` = **`{100 * _chosen_rate:.1f}%`**

    **legal-move rate on the same set:** `{_legal_count}/{_n}` = **`{100 * _legal_rate:.1f}%`**

    **standard error** on chosen-match: ±`{100 * _se:.1f}%`
    """)

    import altair as _alt
    import polars as _pl

    _CATEGORY_ORDER = [
        "matches chosen",
        "matches rejected",
        "other legal",
        "illegal",
    ]
    _CATEGORY_COLORS = ["#2ca02c", "#d62728", "#ffbf00", "#7f7f7f"]
    _color_scale = _alt.Scale(domain=_CATEGORY_ORDER, range=_CATEGORY_COLORS)

    # Single horizontal stacked bar across all rows — reads like a fuel
    # gauge for "how often did the model do the right thing".
    _bar_df = _pl.DataFrame(
        {
            "category": _CATEGORY_ORDER,
            "count": [
                _chosen_count,
                _rejected_count,
                _other_legal_count,
                _illegal_count,
            ],
        }
    ).filter(_pl.col("count") > 0)
    _bar = (
        _alt.Chart(_bar_df)
        .mark_bar()
        .encode(
            x=_alt.X("count:Q", stack="normalize", title="fraction of outputs"),
            color=_alt.Color(
                "category:N",
                scale=_color_scale,
                sort=_CATEGORY_ORDER,
                legend=_alt.Legend(title="category"),
            ),
            order=_alt.Order(
                "category:N", sort="ascending"
            ),  # keeps stack order stable
            tooltip=["category", "count"],
        )
        .properties(width=600, height=70, title=f"Output breakdown over {_n} positions")
    )

    # By-phase stacked bar: same 4 categories but split by game phase.
    _phase_rows = []
    for _phase_name in ("opening", "midgame", "endgame"):
        _bucket = [r for r in winrate_table_rows if r["phase"] == _phase_name]
        if not _bucket:
            continue
        for _cat in _CATEGORY_ORDER:
            _c = sum(1 for r in _bucket if r["category"] == _cat)
            _phase_rows.append(
                {"phase": _phase_name, "category": _cat, "count": _c, "n": len(_bucket)}
            )
    _phase_df = _pl.DataFrame(_phase_rows)
    _phase_chart = (
        _alt.Chart(_phase_df)
        .mark_bar()
        .encode(
            x=_alt.X(
                "phase:N",
                sort=["opening", "midgame", "endgame"],
                title="game phase",
            ),
            y=_alt.Y("count:Q", stack="normalize", title="fraction"),
            color=_alt.Color(
                "category:N", scale=_color_scale, sort=_CATEGORY_ORDER,
            ),
            tooltip=["phase", "category", "count", "n"],
        )
        .properties(width=420, height=320, title="Output breakdown by game phase")
    )

    winrate_result = mo.vstack(
        [
            winrate_summary,
            mo.ui.altair_chart(_bar),
            mo.ui.altair_chart(_phase_chart),
            mo.md("### Per-row scoring"),
            mo.ui.table(winrate_table_rows),
        ]
    )
    winrate_result
    return


if __name__ == "__main__":
    app.run()
