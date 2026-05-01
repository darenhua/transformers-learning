# tf-learn

A marimo notebook exploring SFT fine-tuning of GPT-2 on checkers (international draughts).

## What's in here

`transformers learn.py` — the marimo notebook. It walks through:

1. **Draughts basics** — playing with `pydraughts`: legal moves, FEN, pushing UCI moves, running an `AlphaBetaEngine` vs. itself via the built-in server, and a toy `GreedyAgent` (max captures) benchmarked against alpha-beta.
2. **Dataset generation** — two JSONL datasets built from self-play:
   - `valid_move_dataset.jsonl` — `{fen, move}` records for SFT (mostly engine moves, every 5th game uses random moves for diversity).
   - `optimal_move_dataset.jsonl` — `{fen, chosen, rejected}` records for preference-style training (depth-4 chosen vs. depth-2 rejected, skipping ties).
   - Smoke-test versions (`*_smoke.jsonl`, 2 games each) are committed; full 1000-game runs are gated behind a marimo run button.
3. **SFT training** — converts the valid-move dataset into `prompt`/`completion` pairs (`"<fen>\nMove: <move>"`) and fine-tunes `gpt2` (124M) with `trl.SFTTrainer` on CPU. Training is also gated behind a run button.

## Running

```bash
uv run marimo edit "transformers learn.py"
```

Dependencies: `marimo`, `pydraughts`, `transformers`, `trl`, `datasets`.

## Layout

- `transformers learn.py` — the notebook
- `valid_move_smoke.jsonl`, `optimal_move_smoke.jsonl` — tiny sample datasets
- `gpt2-checkers-sft/` — training output directory (gitignored checkpoints)
