"""Load a saved checkpoint for inference.

Auto-detects whether `checkpoint_path` is a full HF model (saved by
gpt2_sft / qwen_sft / qwen_dpo) or a LoRA adapter (saved by qwen_qlora).
For adapters, loads the base model from `adapter_config.json` and attaches
the adapter via PeftModel.
"""

import json
from pathlib import Path


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """Returns (model, tokenizer) ready for `model.generate(...)`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(checkpoint_path)
    is_adapter = (path / "adapter_config.json").exists()

    tok = AutoTokenizer.from_pretrained(checkpoint_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cuda = torch.cuda.is_available() and device == "cuda"
    bf16_ok = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = (
        torch.bfloat16 if bf16_ok else (torch.float16 if cuda else torch.float32)
    )

    if is_adapter:
        from peft import PeftModel

        with open(path / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg["base_model_name_or_path"]
        print(f"[inference] adapter at {checkpoint_path}, base={base_name}")
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        print(f"[inference] full model at {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, dtype=dtype)

    model.config.pad_token_id = tok.pad_token_id
    if cuda:
        model.to("cuda")
    model.eval()
    return model, tok
