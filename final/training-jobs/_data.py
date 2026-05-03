"""Shared dataset loaders + GPU detection for training-jobs scripts.

Schemas:
    valid_move_*.jsonl     -> {fen, move}                  (SFT, QLoRA)
    optimal_move_*.jsonl   -> {fen, chosen, rejected}      (DPO)

Both formats wrap into the same prompt strings the harness builds at
inference time, so any model trained here works through harness/hf_generate.py
and harness/llm_agent.py without code changes:
    prompt:     f"{fen}\nMove:"
    completion: move                (or chosen / rejected for preferences)
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FALLBACK_ROOTS = [_HERE.parent, _HERE.parent.parent]


def resolve_path(path: str) -> str:
    """Find `path` against cwd or fallback roots (final/ and the repo root).

    Lets the same script work whether you launch from `final/`, the repo
    root, or anywhere — handy when the marimo notebook lives in `final/`
    but the jsonl datasets live one dir up.
    """
    p = Path(path)
    if p.exists():
        return str(p)
    for root in _FALLBACK_ROOTS:
        candidate = root / path
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"could not resolve {path!r} against cwd or {[str(r) for r in _FALLBACK_ROOTS]}"
    )


def _load_split(dataset_path, max_samples, test_size, seed, mapper):
    from datasets import load_dataset

    resolved = resolve_path(dataset_path)
    print(f"[_data] loading {resolved}")
    raw = load_dataset("json", data_files=resolved, split="train")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))
    formatted = raw.map(mapper, remove_columns=raw.column_names)
    if test_size and test_size > 0 and len(formatted) > 1:
        split = formatted.train_test_split(test_size=test_size, seed=seed)
        return split["train"], split["test"]
    return formatted, formatted


def load_sft_dataset(dataset_path, max_samples=None, test_size=0.1, seed=42):
    """Load a valid_move-format jsonl as TRL prompt+completion pairs."""

    def mapper(example):
        return {
            "prompt": f"{example['fen']}\nMove:",
            "completion": example["move"],
        }

    return _load_split(dataset_path, max_samples, test_size, seed, mapper)


def load_dpo_dataset(dataset_path, max_samples=None, test_size=0.1, seed=42):
    """Load an optimal_move-format jsonl as TRL preference triples."""

    def mapper(example):
        return {
            "prompt": f"{example['fen']}\nMove:",
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    return _load_split(dataset_path, max_samples, test_size, seed, mapper)


def sample_eval_rows(dataset_path, n_samples=64, test_size=0.1, seed=42):
    """Return up to n_samples raw rows (dicts) from the same eval split
    that load_*_dataset uses.

    Used by the generation-based metric callbacks (legal-move rate,
    DPO winrate). Mirrors `_load_split` so the rows here don't overlap
    the trainer's training set: same HF train_test_split + seed.
    """
    import random

    from datasets import load_dataset

    resolved = resolve_path(dataset_path)
    raw = load_dataset("json", data_files=resolved, split="train")
    if test_size and test_size > 0 and len(raw) > 1:
        eval_ds = raw.train_test_split(test_size=test_size, seed=seed)["test"]
    else:
        eval_ds = raw
    rows = [dict(r) for r in eval_ds]
    if n_samples is not None and len(rows) > n_samples:
        rows = random.Random(seed + 1).sample(rows, n_samples)
    return rows


def gpu_setup():
    """Return (cuda: bool, bf16_ok: bool); print a one-line status."""
    import torch

    cuda = torch.cuda.is_available()
    bf16_ok = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    if cuda:
        print(
            f"[_data] CUDA on: {torch.cuda.get_device_name(0)} — "
            f"using {'bf16' if bf16_ok else 'fp16'}"
        )
    else:
        print("[_data] CUDA not available — falling back to CPU")
    return cuda, bf16_ok
