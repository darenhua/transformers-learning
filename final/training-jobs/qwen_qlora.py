"""Qwen2.5-0.5B-Instruct QLoRA: 4-bit NF4 base + LoRA adapter.

LoRA hyperparameters follow "LoRA Without Regret" (Schulman et al., 2025):
    r=256, lora_alpha=16, target_modules="all-linear",
    lr=2e-4, effective batch ≤ 32.

Saves the adapter only (~MB, not GB) — base weights stay on HF Hub.
At inference, training-jobs/inference.py reattaches the adapter.

Run directly:
    .env/bin/python final/training-jobs/qwen_qlora.py --run-name qwen-qlora --epochs 1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import gpu_setup, load_sft_dataset


def train(
    run_name: str = "qwen-qlora",
    epochs: int = 1,
    dataset_path: str = "valid_move_train.jsonl",
    max_samples: int | None = None,
    max_steps: int | None = None,
    output_root: str = "checkpoints",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_r: int = 256,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    eval_steps: int = 100,
    test_size: float = 0.1,
    eval_metric_samples: int = 16,
):
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from _metrics import make_legal_move_callback

    cuda, bf16_ok = gpu_setup()
    if not cuda:
        raise RuntimeError("QLoRA requires CUDA + bitsandbytes; no GPU available")

    train_ds, eval_ds = load_sft_dataset(
        dataset_path, max_samples=max_samples, test_size=test_size
    )
    print(
        f"[qwen_qlora] train={len(train_ds)} eval={len(eval_ds)} "
        f"epochs={epochs} max_steps={max_steps}"
    )

    # On T4 (compute 7.5) bf16 isn't natively supported, so the
    # non-quantized layers must be in fp16 to match bnb's compute dtype.
    # Mixing bf16 anywhere on T4 causes AMP "unscale not implemented
    # for BFloat16" or silent NaN gradients.
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    output_dir = os.path.join(output_root, run_name)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # eff batch=16, < 32 per LoRA-Without-Regret
        # bnb's compute_dtype already handles forward-pass precision; an
        # additional AMP scaler conflicts with bf16 LoRA adapter params.
        bf16=False,
        fp16=False,
        dataloader_pin_memory=True,
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
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        peft_config=peft_config,
        callbacks=[
            make_legal_move_callback(
                dataset_path, tok, n_samples=eval_metric_samples,
                test_size=test_size,
            )
        ],
    )
    trainer.train()
    log_history = list(trainer.state.log_history)

    trainer.save_model(output_dir)  # adapter only
    tok.save_pretrained(output_dir)
    Path(output_dir, "DONE").write_text("ok\n")
    print(f"[qwen_qlora] saved adapter + tokenizer to {output_dir}")
    return {"output_dir": output_dir, "log_history": log_history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="qwen-qlora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--dataset", default="valid_move_train.jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-root", default="checkpoints")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=256)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
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
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        learning_rate=a.learning_rate,
        test_size=a.test_size,
        eval_steps=a.eval_steps,
        eval_metric_samples=a.eval_metric_samples,
    )


if __name__ == "__main__":
    main()
