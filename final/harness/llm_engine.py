"""Robust LLM-as-engine wrapper for Benchmark and Server.

What this gives you over a raw LLMAgent:

- 5 retries on illegal moves (vs the LLMAgent default of 3). The engine NEVER
  returns an illegal move — `LLMAgent.select_move` already drops anything that
  doesn't parse to a `legal_moves` element, then falls back to a random pick
  from `board.legal_moves` once retries are exhausted.
- Sampling on retry (default temperature=0.8) so the 5 retries actually
  diversify — under pure greedy decoding the SFT model returns identical text
  on retry, which makes retries no-ops. Attempt 0 stays greedy.
- Per-turn output in conversation form ("You: <FEN>\\n AI: <move>") to either
  stdout (for `play_server.py` so the server terminal shows the game), a file
  (for the benchmark notebook's tail cell), or both. Set neither to get a
  silent engine.

Use:
    from harness.llm_engine import make_robust_llm_engine
    # benchmark — write to file
    agent, engine = make_robust_llm_engine("checkpoints/qwen-dpo",
                                           log_path="logs/benchmark.log")
    # server — stream to stdout
    agent, engine = make_robust_llm_engine("checkpoints/qwen-dpo",
                                           log_to_stdout=True)
    Benchmark(engine, AlphaBetaEngine(depth_limit=3), games=20).run()
"""

import random
import sys
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "training-jobs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from draughts import Board

from harness.hf_generate import make_hf_generate_fn
from harness.llm_agent import LLMAgent, TurnRecord
from harness.scoring import parse_move


class LoggingLLMAgent(LLMAgent):
    """LLMAgent variant with greedy-then-sampled retries and per-turn logging.

    Differs from the base LLMAgent in two ways:
      1. Optional `generate_retry_fn` — used for attempts >= 1 so retries
         actually diversify. With pure greedy, retries on the same prompt
         are no-ops.
      2. Optional `log_path` and/or `log_to_stdout` — emit a
         "You: <FEN>\\nAI: <move>\\n\\n" record per turn. If the agent fell
         back to random, the AI line is annotated.
    """

    def __init__(
        self,
        *args,
        generate_retry_fn: Optional[Callable[[str], str]] = None,
        log_path: Optional[str | Path] = None,
        log_to_stdout: bool = False,
        side_label: str = "AI",
        **kwargs,
    ):
        if log_path is not None or log_to_stdout:
            kwargs.setdefault("record_turns", True)
        super().__init__(*args, **kwargs)
        self.generate_retry_fn = generate_retry_fn or self.generate_fn
        self.log_path = Path(log_path) if log_path else None
        self.log_to_stdout = log_to_stdout
        self.side_label = side_label
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def select_move(self, board: Board):
        # Reimplements the base loop so attempt 0 uses generate_fn (greedy)
        # and subsequent attempts use generate_retry_fn (sampling).
        legal_moves = list(board.legal_moves)
        legal_strs = [str(m) for m in legal_moves]

        record = TurnRecord(fen_before=board.fen) if self.record_turns else None
        self.num_turns += 1

        prompt = self._build(self.prompt_fn, board, legal_strs)
        chosen = None
        fell_back = False

        for attempt in range(self.max_retries):
            if record is not None:
                record.prompts.append(prompt)
            gen = self.generate_fn if attempt == 0 else self.generate_retry_fn
            response = gen(prompt)
            if record is not None:
                record.responses.append(response)

            move = parse_move(response, legal_moves)
            if move is not None:
                if attempt == 0:
                    self.first_try_legal += 1
                if record is not None:
                    record.parsed = str(move)
                    record.legal_first_try = (attempt == 0)
                    record.chosen_move = str(move)
                chosen = move
                break
            prompt = self._build_retry(self.retry_prompt_fn, board, legal_strs, response)

        if chosen is None:
            self.total_fallbacks += 1
            chosen = random.choice(legal_moves)
            fell_back = True
            if record is not None:
                record.fell_back_to_random = True
                record.chosen_move = str(chosen)

        if record is not None:
            self.turns.append(record)

        if self.log_path is not None or self.log_to_stdout:
            label = self.side_label + (" (random fallback)" if fell_back else "")
            line = f"You: {board.fen}\n{label}: {chosen}\n\n"
            if self.log_path is not None:
                with open(self.log_path, "a") as f:
                    f.write(line)
            if self.log_to_stdout:
                # End=‹empty› so the trailing \n\n in `line` controls spacing,
                # and `flush=True` so it streams during a server's request loop.
                print(line, end="", flush=True)

        return chosen


def make_robust_llm_engine(
    checkpoint_path: str,
    max_retries: int = 5,
    max_new_tokens: int = 12,
    retry_temperature: float = 0.8,
    log_path: Optional[str] = None,
    log_to_stdout: bool = False,
    side_label: str = "AI",
    record_turns: bool = False,
    device: Optional[str] = None,
):
    """Load a HF/PEFT checkpoint and return (agent, engine).

    `engine` plugs straight into `Benchmark(...)` and `Server(...)`.
    `agent` is the underlying `LoggingLLMAgent` — keep a handle to read
    `agent.legal_move_rate()`, `agent.fallback_rate()`, `agent.turns` after
    a run.
    """
    import torch
    import inference

    model, tok = inference.load_model_and_tokenizer(checkpoint_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_greedy = make_hf_generate_fn(
        model, tok, max_new_tokens=max_new_tokens, temperature=0.0, device=device
    )
    gen_sampled = make_hf_generate_fn(
        model, tok, max_new_tokens=max_new_tokens,
        temperature=retry_temperature, device=device,
    )

    agent = LoggingLLMAgent(
        generate_fn=gen_greedy,
        generate_retry_fn=gen_sampled,
        max_retries=max_retries,
        record_turns=record_turns,
        log_path=log_path,
        log_to_stdout=log_to_stdout,
        side_label=side_label,
    )
    return agent, agent.as_engine()
