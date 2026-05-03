"""GPT-2 SFT on the valid_moves dataset.

Run directly:
    .env/bin/python final/training-jobs/gpt2_sft.py --run-name gpt2-sft --epochs 1

Import from a notebook (see model_llm_local.py):
    import gpt2_sft
    result = gpt2_sft.train(run_name="smoke", epochs=10, max_samples=20)

Output format matches the inference harness exactly:
    prompt:     f"{fen}\nMove:"
    completion: move

`save_model` writes to `<output_root>/<run_name>/` and a `DONE` sentinel —
the overnight bash runner uses that to skip dependent jobs (DPO needing SFT).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import gpu_setup, load_sft_dataset


def train(
    run_name: str = "gpt2-sft",
    epochs: int = 1,
    dataset_path: str = "valid_move_train.jsonl",
    max_samples: int | None = None,
    max_steps: int | None = None,
    output_root: str = "checkpoints",
    base_model: str = "gpt2",
    learning_rate: float = 5e-5,
    eval_steps: int = 50,
    test_size: float = 0.1,
    eval_metric_samples: int = 64,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from _metrics import make_legal_move_callback

    cuda, bf16_ok = gpu_setup()
    train_ds, eval_ds = load_sft_dataset(
        dataset_path, max_samples=max_samples, test_size=test_size
    )
    print(
        f"[gpt2_sft] train={len(train_ds)} eval={len(eval_ds)} "
        f"epochs={epochs} max_steps={max_steps}"
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.config.pad_token_id = tok.eos_token_id
    if cuda:
        model.to("cuda")

    output_dir = os.path.join(output_root, run_name)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=16 if cuda else 4,
        per_device_eval_batch_size=16 if cuda else 4,
        gradient_accumulation_steps=1,
        bf16=bf16_ok,
        fp16=cuda and not bf16_ok,
        dataloader_pin_memory=cuda,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        callbacks=[
            make_legal_move_callback(
                dataset_path, tok, n_samples=eval_metric_samples,
                test_size=test_size,
            )
        ],
    )
    trainer.train()
    log_history = list(trainer.state.log_history)

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    Path(output_dir, "DONE").write_text("ok\n")
    print(f"[gpt2_sft] saved final model + tokenizer to {output_dir}")
    return {"output_dir": output_dir, "log_history": log_history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="gpt2-sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--dataset", default="valid_move_train.jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-root", default="checkpoints")
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--test-size", type=float, default=0.1)
    a = p.parse_args()
    train(
        run_name=a.run_name,
        epochs=a.epochs,
        dataset_path=a.dataset,
        max_samples=a.max_samples,
        max_steps=a.max_steps,
        output_root=a.output_root,
        base_model=a.base_model,
        learning_rate=a.learning_rate,
        test_size=a.test_size,
    )


if __name__ == "__main__":
    main()
