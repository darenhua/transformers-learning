"""Qwen DPO on the optimal_move preference dataset.

Loads from an SFT checkpoint by default (`checkpoints/qwen-sft/`). Hard-fails
fast if the SFT checkpoint's `DONE` sentinel is missing — gives the bash
runner a clean signal to skip dependent jobs rather than burning hours and
crashing on a missing file.

Pass `--peft` to train a LoRA adapter on top of SFT instead of full DPO
(useful if VRAM is tight; uses higher lr=1e-5).

Run directly:
    .env/bin/python final/training-jobs/qwen_dpo.py --run-name qwen-dpo --epochs 1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import gpu_setup, load_dpo_dataset


def train(
    run_name: str = "qwen-dpo",
    epochs: int = 1,
    dataset_path: str = "optimal_move_train.jsonl",
    max_samples: int | None = None,
    max_steps: int | None = None,
    output_root: str = "checkpoints",
    base_model: str = "checkpoints/qwen-sft",
    learning_rate: float | None = None,
    beta: float = 0.1,
    use_peft: bool = False,
    eval_steps: int = 100,
    test_size: float = 0.1,
    eval_metric_samples: int = 16,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from _metrics import make_winrate_callback

    cuda, bf16_ok = gpu_setup()

    # Bash-guard pattern: if base_model points at a sibling checkpoint dir,
    # require the SFT job to have written its DONE sentinel. Skip the check
    # for HF Hub names like "Qwen/...".
    looks_local = base_model.startswith(("checkpoints/", "./checkpoints/", "/"))
    if looks_local:
        done = Path(base_model) / "DONE"
        if not done.exists():
            raise FileNotFoundError(
                f"SFT checkpoint not ready: {done} missing. "
                f"Run qwen_sft (or qwen_qlora and merge) first, "
                f"or pass --base-model <hf-id> to start from a fresh model."
            )

    train_ds, eval_ds = load_dpo_dataset(
        dataset_path, max_samples=max_samples, test_size=test_size
    )
    print(
        f"[qwen_dpo] train={len(train_ds)} eval={len(eval_ds)} "
        f"epochs={epochs} max_steps={max_steps} peft={use_peft}"
    )

    # Two precision regimes:
    # - PEFT path: load base in low precision (frozen base used as both
    #   policy backbone and reference; only fp32 LoRA params train, so AMP
    #   has no fp16 master-copy issue).
    # - Full DPO: load fp32 weights so AMP scaler can unscale; smaller
    #   batch + precompute_ref_log_probs is required to fit on T4-class
    #   GPUs.
    if use_peft:
        load_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        load_dtype = torch.bfloat16 if bf16_ok else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=load_dtype)
    model.config.pad_token_id = tok.pad_token_id
    if cuda:
        model.to("cuda")

    peft_config = None
    if use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    if learning_rate is None:
        learning_rate = 1e-5 if use_peft else 1e-6

    output_dir = os.path.join(output_root, run_name)
    # PEFT skips the AMP scaler (model already in low precision); full DPO
    # uses AMP (model in fp32, scaler unscales fp16 grads safely). Full DPO
    # also frees the ref model after a one-time logprob precompute, so a
    # T4 can handle 0.5B-param models without OOM.
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=2 if cuda else 1,
        per_device_eval_batch_size=2 if cuda else 1,
        gradient_accumulation_steps=4,
        bf16=bf16_ok and not use_peft,
        fp16=cuda and not bf16_ok and not use_peft,
        dataloader_pin_memory=cuda,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_steps=max_steps if max_steps is not None else -1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=not use_peft,
    )
    trainer = DPOTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        peft_config=peft_config,
        callbacks=[
            make_winrate_callback(
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
    print(f"[qwen_dpo] saved DPO model + tokenizer to {output_dir}")
    return {"output_dir": output_dir, "log_history": log_history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="qwen-dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--dataset", default="optimal_move_train.jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-root", default="checkpoints")
    p.add_argument("--base-model", default="checkpoints/qwen-sft")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--peft", action="store_true", help="train a LoRA adapter on top")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-metric-samples", type=int, default=16)
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
        beta=a.beta,
        use_peft=a.peft,
        test_size=a.test_size,
        eval_steps=a.eval_steps,
        eval_metric_samples=a.eval_metric_samples,
    )


if __name__ == "__main__":
    main()
