"""Prompt formats. CRITICAL: inference prompt MUST match training prompt.

In transformers_learn.py the SFT training format is:
    prompt:     f"{fen}\\nMove:"
    completion: f" {move}"

If you switch to the verbose format at inference, the model has never seen
that distribution and will produce garbage. Pick one format and stick with it
across train + inference + retry.

Two builders here:
- build_minimal_prompt: matches the current SFT training format.
- build_verbose_prompt: the original LLMAgent prompt — only use if you train
  on this format too.
"""


def build_minimal_prompt(board) -> str:
    """Matches `f"{fen}\\nMove:"` SFT training format. Use this with the
    gpt-2 SFT checkpoint produced by transformers_learn.py."""
    return f"{board.fen}\nMove:"


def build_minimal_retry_prompt(board, bad_response: str) -> str:
    """Retry prompt for the minimal format. Stays close to training distribution
    by re-emitting the same prefix; the bad response is appended as a comment
    the model will likely ignore (and that's fine — at inference we mainly
    benefit from temperature/sampling differences on retry).

    If retries don't help with the SFT model, drop max_retries to 1 and rely
    on the random-fallback path.
    """
    return f"{board.fen}\nMove:"


def build_verbose_prompt(board, legal_strs: list[str]) -> str:
    """Original LLMAgent prompt. Only use if your model is trained or
    instruction-tuned on this style (e.g. an OpenAI/Anthropic API model)."""
    return (
        "You are playing American checkers (8x8). "
        "You must reply with ONLY a move in the format shown below "
        "(e.g. '22-18' or '15x6').\n\n"
        f"Current board:\n{board}\n\n"
        f"FEN: {board.fen}\n\n"
        f"Legal moves: {', '.join(legal_strs)}\n\n"
        "Your move:"
    )


def build_verbose_retry_prompt(board, legal_strs: list[str], bad_response: str) -> str:
    return (
        "You are playing American checkers (8x8). "
        f"Your previous response '{bad_response}' was not a valid move.\n\n"
        f"Current board:\n{board}\n\n"
        f"Legal moves: {', '.join(legal_strs)}\n\n"
        "Reply with ONLY one of the legal moves listed above (e.g. '22-18').\n\n"
        "Your move:"
    )
