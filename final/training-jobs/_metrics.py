"""TrainerCallbacks that run end-to-end greedy generation on a held-out
sample and log a domain metric to state.log_history so the notebook
plots can pick it up alongside train/eval loss.

Two callbacks:
    LegalMoveRateCallback — SFT/QLoRA. Fraction of held-out positions
        where the model's greedy continuation parses to a move that's
        in `board.legal_moves`.
    DPOWinrateCallback — DPO. Fraction of held-out preference rows
        where greedy continuation matches the `chosen` move (offline
        winrate; no on-the-fly game simulation).

Cost: n_samples * n_evals greedy decodes per training run. n_samples
defaults to 64 — bump up for tighter metric, down if eval is dominating
wall time.
"""

import sys
from pathlib import Path

from transformers import TrainerCallback

# Sibling import: _data lives in the same dir as this file.
# Parent (final/) goes on sys.path too so `from harness.scoring import ...`
# works whether the train script is launched from final/ or the repo root.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class _GenerationMetricCallback(TrainerCallback):
    """Shared on_evaluate plumbing: greedy-generate against the trainer's
    current model on each row, score, log a single scalar. Subclasses
    only have to implement `_score_row` and set `metric_key`."""

    metric_key: str = ""

    def __init__(self, eval_rows, tokenizer, max_new_tokens=12):
        self.eval_rows = eval_rows
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _resolve_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return "cpu"

    def _generate(self, model, prompt, device):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    def _score_row(self, row, response):
        raise NotImplementedError

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not self.eval_rows:
            return control
        was_training = model.training
        model.eval()
        device = self._resolve_device(model)
        try:
            hits = 0
            for row in self.eval_rows:
                resp = self._generate(model, f"{row['fen']}\nMove:", device)
                hits += int(self._score_row(row, resp))
            rate = hits / len(self.eval_rows)
        finally:
            if was_training:
                model.train()
        # Append a fresh log entry at the same step the trainer just
        # logged eval_loss at — the notebook plot keys off `step`.
        state.log_history.append(
            {"step": state.global_step, self.metric_key: rate}
        )
        print(f"[_metrics] step {state.global_step} {self.metric_key}={rate:.3f}")
        return control


class LegalMoveRateCallback(_GenerationMetricCallback):
    metric_key = "eval_legal_move_rate"

    def _score_row(self, row, response):
        from draughts import Board

        from harness.scoring import is_legal

        board = Board.from_fen(row["fen"])
        return is_legal(response, list(board.legal_moves))


class DPOWinrateCallback(_GenerationMetricCallback):
    metric_key = "eval_winrate"

    def _score_row(self, row, response):
        from draughts import Board

        from harness.scoring import parse_move

        board = Board.from_fen(row["fen"])
        move = parse_move(response, list(board.legal_moves))
        return move is not None and str(move) == row["chosen"]


def make_legal_move_callback(
    dataset_path, tokenizer, n_samples=64, test_size=0.1, seed=42
):
    from _data import sample_eval_rows

    rows = sample_eval_rows(dataset_path, n_samples, test_size, seed)
    return LegalMoveRateCallback(rows, tokenizer)


def make_winrate_callback(
    dataset_path, tokenizer, n_samples=64, test_size=0.1, seed=42
):
    from _data import sample_eval_rows

    rows = sample_eval_rows(dataset_path, n_samples, test_size, seed)
    return DPOWinrateCallback(rows, tokenizer)
