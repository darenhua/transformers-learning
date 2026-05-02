"""Bridge: HuggingFace transformers checkpoint -> generate_fn(prompt) -> str.

Use with the gpt-2 SFT checkpoint produced by transformers_learn.py.

CRITICAL: the prompt format passed to generate_fn must match the SFT training
format exactly (`f"{fen}\\nMove:"`). Pair this with `build_minimal_prompt`
from harness.prompts.

Usage:
    from harness.llm_agent import LLMAgent
    from harness.hf_generate import make_hf_generate_fn

    generate = make_hf_generate_fn(model, tokenizer, max_new_tokens=12)
    agent = LLMAgent(generate_fn=generate)  # uses minimal prompt by default
"""

from typing import Callable


def make_hf_generate_fn(
    model,
    tokenizer,
    max_new_tokens: int = 12,
    temperature: float = 0.0,
    device: str = "cpu",
) -> Callable[[str], str]:
    """Wrap a HF causal LM into a generate_fn(prompt) -> str.

    Returns ONLY the model's continuation (the prompt is stripped from the
    output) so downstream parsing sees just the move text.
    """
    model.to(device)
    model.eval()

    do_sample = temperature > 0.0

    def generate(prompt: str) -> str:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Strip the prompt tokens — return only the continuation
        prompt_len = inputs["input_ids"].shape[1]
        continuation_ids = out[0, prompt_len:]
        return tokenizer.decode(continuation_ids, skip_special_tokens=True)

    return generate
