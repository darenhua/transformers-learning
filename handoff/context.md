# Context — what's in the source repo and what's reusable

This handoff folder has two roles:

1. **`models/` and `evals/`** — a snapshot of the GitHub source repo for
   reference. The patterns there (BaseAgent + as_engine + Benchmark factory
   dict) are the model.
2. **`harness/`** — the *consolidated, refactored* primitives the marimo
   project should actually consume. This is what gets copied (or
   pip-installed) into the marimo repo.

When in doubt, use `harness/`. `models/` is there to show what the existing
abstractions look like and to flag the bugs to avoid bringing along.

## Module-first design (with marimo's autoreloader)

Marimo has a module autoreloader: edit a `.py` module, dependent cells re-run
automatically. This changes the calculus on what should be a notebook vs a
module:

- **Module** if it's stateless logic, classes, factory functions, scoring,
  prompts. (Less `@app.cell` boilerplate, easier to import from multiple
  places.) → `harness/*.py`
- **Notebook** if it has UI cells you want to embed via `app.embed()`, **or**
  expensive state you want to hold across reactivity (e.g. a loaded gpt-2
  checkpoint). → model-loader notebooks, previewer, and the three harnesses.

## Layout

```
handoff/
  harness/                # CONSUME THIS in the marimo project
    agents.py             # RandomAgent, GreedyAgent, MinimaxAgent (consolidated)
    llm_agent.py          # LLMAgent — model-swap point
    scoring.py            # parse_move, is_legal — pass/fail validity scorer
    prompts.py            # build_minimal_prompt + build_verbose_prompt
    engines.py            # run_benchmark — Benchmark wrapper
    registry.py           # MODELS dict (random/greedy/minimax pre-registered)
    hf_generate.py        # HF transformers -> generate_fn bridge

  models/                 # REFERENCE — original source repo snapshot
    llm_agent.py          # original LLMAgent (has the legal_move_rate bug)
    random_bot.py
    minimax_bot.py
  evals/                  # REFERENCE — eval scripts from source repo
    eval_win_rate.py      # the Benchmark usage pattern is good — copy the idea
    eval_consistency.py   # secondary, do not port day 1
  datasets/
    board_states.csv      # 13 FENs for consistency eval
```

The custom `CheckersEnv.py` and `main.py` in the source repo are **not** copied —
the working pipeline uses `py-draughts` (`AmericanBoard`, `BaseAgent`, `Benchmark`)
end-to-end.

## Core abstractions (already correct, reuse as-is)

- **`draughts.BaseAgent`** — interface. Subclass and implement `select_move(board)`.
  All three agents in `models/` follow this.
- **`agent.as_engine()`** — built into py-draughts. Wraps a `BaseAgent` into the
  engine interface that `Benchmark` consumes. **You do not need to write this.**
- **`draughts.Benchmark`** — runs N games between two engines and returns
  `wins/losses/draws/win_rate/elo_diff`. Used at `evals/eval_win_rate.py:33`.
- **`draughts.AmericanBoard`** — 8x8 board class. `board.legal_moves`,
  `board.fen`, `board.push(move)`, `board.copy()`, `board.game_over`,
  `Board.from_fen(fen)`, `board.features()`.

## The agents (consolidated in `harness/`)

### `harness/agents.py` — baselines
- `RandomAgent` — uniform random over legal moves.
- `GreedyAgent` — picks the move with the longest capture chain. Cheap,
  occasionally clever in tactical positions.
- `MinimaxAgent(depth=N)` — negamax + alpha-beta. Material eval (men=1,
  kings=3). The built-in `AlphaBetaEngine` only supports 10x10, so this
  exists for 8x8. Standard depths: `3` (weak), `8` (strong).

All three are 3–60-line classes — they live in one file because separate
notebooks for each would be pure boilerplate, and autoreload makes editing a
module feel notebook-native.

### `harness/llm_agent.py` — `LLMAgent(generate_fn, prompt_fn=..., max_retries=3, record_turns=False)`
The model-swap point. `generate_fn(prompt: str) -> str` plugs in any LLM (HF,
OpenAI, local checkpoint). `prompt_fn` is **injectable** — defaults to
`build_minimal_prompt` (matches the SFT training format) but can be swapped
to `build_verbose_prompt` for instruction-tuned API models.

Per-turn flow in `select_move`:
1. Build prompt via `prompt_fn(board)` (or `prompt_fn(board, legal_strs)` for verbose).
2. Call `generate_fn(prompt)`.
3. `harness.scoring.parse_move(response, legal_moves)` checks validity.
4. If parsed and legal: return it.
5. Otherwise: rebuild retry prompt and try again, up to `max_retries`.
6. After all retries fail: random legal-move fallback.

Stats tracked: `num_turns`, `first_try_legal`, `total_fallbacks`. With
`record_turns=True`, each turn produces a `TurnRecord` with prompts,
responses, parsed move, legal/fallback flags — used by the test harness.

### Bugs from `models/llm_agent.py` that are **fixed** in `harness/llm_agent.py`
1. `legal_move_rate()` formula — now correctly `first_try_legal / num_turns`.
2. Per-turn vs per-attempt counting — `num_turns` increments once per
   `select_move`, separate from per-attempt response counting.

## The two eval scripts (reference, do not copy logic verbatim)

### `evals/eval_win_rate.py` — primary metric, **good reference**
- `BASELINES` dict (line 17) — clean factory pattern. **Copy this idea** into the
  marimo `MODELS` registry.
- `eval_win_rate()` (line 24) — wraps `Benchmark(...).run()` and unpacks stats.
- `eval_all_baselines()` sweeps every baseline.

### `evals/eval_consistency.py` — secondary metric
- Loads `datasets/board_states.csv`, queries the agent N times per FEN, counts
  most-common move via `Counter`. Useful but not on the critical path.

## Things still to watch for

1. **`Benchmark` prints to stdout.** Already handled inside
   `harness.engines.run_benchmark` via `contextlib.redirect_stdout`. If you
   call `Benchmark(...).run()` directly anywhere, wrap it the same way.

2. **Prompt format must match training.** The SFT training in
   `transformers_learn.py` uses `f"{fen}\nMove:"`. `LLMAgent` defaults to
   `build_minimal_prompt` for this reason — if you pass `prompt_fn=build_verbose_prompt`
   to a model trained on the minimal format, you'll get garbage.

3. **Reactive cells + stateful agents.** `LLMAgent` accumulates `num_turns`,
   `first_try_legal`, etc. across `select_move` calls. If you hold an agent in
   `mo.state`, call `reset_stats()` between runs. The default factory pattern
   (`registry.build("name")`) builds a fresh agent every time — recommended.

## Dependencies

`py-draughts >= 1.6.4` is the only hard dep. See `pyproject.toml` in this folder.
The marimo project will additionally need `marimo` itself and whatever the
local-model pipeline needs (transformers, torch, etc.).

## How to consume this folder from the marimo project

Either:

**Option A — `pip install -e` this folder.** Marimo cells do
`from harness.agents import RandomAgent` etc.

**Option B — Copy `harness/` into the marimo repo.** Simpler, but the two
copies will drift. Recommended only if the marimo project diverges from this
one.

Recommend Option A.

The marimo project should *not* import from `models/` — that's a snapshot of
the source repo for reference only. All real consumption goes through
`harness/`.
