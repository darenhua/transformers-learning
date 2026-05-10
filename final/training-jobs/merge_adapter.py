"""Merge a LoRA/QLoRA adapter into its base model and save as a full
self-contained checkpoint (config.json + safetensors + tokenizer).

Why: qwen_dpo.py loads its base via AutoModelForCausalLM.from_pretrained,
which won't pick up a sibling adapter_config.json. To use a QLoRA run as
the policy backbone for DPO, we first bake the adapter weights into the
base via PeftModel.merge_and_unload().

Usage:
    .env/bin/python final/training-jobs/merge_adapter.py \\
        --adapter-path checkpoints/qwen-qlora \\
        --output-path checkpoints/qwen-qlora-merged
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", required=True, help="Dir with adapter_config.json")
    p.add_argument("--output-path", required=True, help="Where to write merged model")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="auto = bf16 if compute capability >=8 else fp16 if cuda else fp32",
    )
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(a.adapter_path)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no adapter_config.json at {cfg_path}")
    with open(cfg_path) as f:
        adapter_cfg = json.load(f)
    base_name = adapter_cfg["base_model_name_or_path"]
    print(f"[merge] base={base_name} adapter={adapter_dir}")

    if a.dtype == "auto":
        cuda = torch.cuda.is_available()
        bf16_ok = cuda and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = (
            torch.bfloat16
            if bf16_ok
            else (torch.float16 if cuda else torch.float32)
        )
    else:
        dtype = getattr(torch, a.dtype)
    print(f"[merge] loading base in {dtype}")

    # Load base in real precision (NOT 4-bit) so the LoRA merge math works
    # cleanly. The merged checkpoint is then saved at this dtype, ready to
    # be re-quantized by whoever loads it next.
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] merging adapter into base...")
    merged = model.merge_and_unload()

    output_dir = Path(a.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving full model to {output_dir}")
    merged.save_pretrained(output_dir)

    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.save_pretrained(output_dir)
    (output_dir / "DONE").write_text("ok\n")
    print(f"[merge] done. {output_dir} has config.json + safetensors + tokenizer.")


if __name__ == "__main__":
    main()
