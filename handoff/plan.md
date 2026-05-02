# Plan — building the marimo harness on top of these primitives

Read `context.md` first. This file is the build plan.

## Review of `transformers_learn.py` (current marimo prototype)

**Aligns:**
- `mo.ui.run_button` gating expensive ops (server, benchmark, dataset gen, training).
- `BaseAgent` → `as_engine()` → `Benchmark(...)` pattern at lines 100–112.
- `Server(board, white_engine, black_engine)` at line 76 is the right scaffold for Harness 3.
- Dataset gen produces SFT + DPO formats from real game trajectories.
- The `HubEngine._read_line` monkey-patch (lines 139–146) is a real bug fix; keep it.

**Critical gap — silent killer:**

Training prompt at line 290 is `f"{example['fen']}\nMove:"` with completion `f" {example['move']}"`.
The original `LLMAgent._build_prompt` is the wordy *"You are playing American checkers (8x8)…"* with the full legal-moves list. **The trained model has never seen the wordy format and will produce garbage at inference.**

Resolved here by:
- `harness/prompts.py` exposes both `build_minimal_prompt` (matches SFT) and `build_verbose_prompt` (for instruction-tuned API models).
- `harness/llm_agent.py` takes `prompt_fn` as a constructor arg — defaults to **minimal** so it works out-of-the-box with the SFT checkpoint.

**Smaller misalignments (port these as you migrate the notebook):**
- `GreedyAgent` defined inline inside the benchmark cell. Move it to `harness/agents.py` (already done in this handoff) and register in `harness/registry.py`.
- No bridge from trained checkpoint → `generate_fn` → `LLMAgent`. Use `harness/hf_generate.py:make_hf_generate_fn(model, tokenizer)`.
- DPO records have `chosen`/`rejected` but no `prompt` column — TRL `DPOTrainer` needs all three (`{"prompt": "{fen}\nMove:", "chosen": " {chosen}", "rejected": " {rejected}"}`).
- `print(stats)` instead of structured rendering. Move to `notebooks/previewer.py` exposing `app` for `await app.embed()`.
- `generate_valid_move_dataset`'s `i % 5 != 0` random-move injection is a magic ratio — pull into a parameter.
- Dataset gen doesn't verify `str(move)` round-trips via `Board.from_fen(fen).push_uci(move_str)`. Add a sanity check in the writer to catch silent format drift.

## Goals (assumed yes to all three)

1. **Reactive benchmark harness** — model dropdown + opponent dropdown + games
   slider → run `py-draughts` `Benchmark` → render results table / charts.
2. **Test harness** — single game, step-by-step, with prompt / response /
   parsed-move / legal? introspection per turn. Includes the valid-move scorer.
3. **Real gameplay harness** — agent wrapped behind a server (or hosted inside
   marimo) so you can play against it in a browser.

## Shared primitives — already built in `handoff/harness/`

All five primitives are **plain modules** because they have no UI and no
expensive state. Marimo's module autoreloader reruns dependent cells on
edit, so iterating on a module feels notebook-native.

| File | What it does |
|---|---|
| `harness/agents.py` | RandomAgent, GreedyAgent, MinimaxAgent — stateless baselines |
| `harness/llm_agent.py` | LLMAgent — model-swap point, takes injectable `prompt_fn` |
| `harness/scoring.py` | `parse_move`, `is_legal` — pass/fail validity scorer |
| `harness/prompts.py` | `build_minimal_prompt` (matches SFT) + `build_verbose_prompt` (API style) |
| `harness/engines.py` | `run_benchmark(agent, opponent, games)` — wraps `Benchmark` + suppresses stdout, returns `BenchmarkResult` |
| `harness/registry.py` | `MODELS` dict pre-registered with random/greedy/minimax_d3/minimax_d8 + `register(name, factory)` |
| `harness/hf_generate.py` | `make_hf_generate_fn(model, tokenizer)` — bridges HF transformers to LLMAgent |

`MODELS` values are **factories** (not instances) so reactive cells can build
a fresh agent and reset stats per run.

## Helper notebook composition strategy

You asked: "I have an eval previewer experience I like, model notebooks that
train/load AI, and want them all imported into one main notebook with a
dropdown — what's the right pattern?"

Two complementary mechanisms in marimo:

1. **Plain Python imports** — for shared *logic* (factories, scoring,
   prompts). Marimo notebooks are Python files: anything defined at the top
   level of `model_llm_local.py` is importable as `from notebooks.model_llm_local import make_agent`.
2. **`await app.embed()`** — for embedding the *UI cells* of one notebook
   inside another. Each model notebook and the previewer notebook export
   their `app` object; the main notebook imports them, awaits `app.embed()`,
   and renders `embed.output` (or named-cell outputs) directly in its own cells.

Use logic-level imports for the registry/factories. Use `app.embed()` when you
want to literally drop the previewer's chart-and-table UI into the main
notebook without rebuilding it.

### Recommended layout

```
harness/                 # MODULES (autoreload-friendly)
  agents.py              # RandomAgent, GreedyAgent, MinimaxAgent
  llm_agent.py           # LLMAgent
  scoring.py
  prompts.py
  engines.py
  registry.py            # baselines pre-registered
  hf_generate.py

notebooks/               # NOTEBOOKS (only when UI or expensive state demand it)
  model_llm_local.py     # loads gpt-2 checkpoint, exports make_agent()
  model_llm_openai.py    # if you want a "test one prompt" cell — otherwise a module is fine
  model_llm_<other>.py   # one notebook per trained model you want to hold in memory

  previewer.py           # UI cells; exports `app` for `await app.embed()`

  harness_benchmark.py   # MAIN composer — dropdown over MODELS, runs benchmark, embeds previewer
  harness_test.py        # single-game introspection
  harness_gameplay.py    # in-marimo play loop OR launches Server
```

**The principle:** notebook iff (UI cells you want to embed) **or**
(expensive state to hold across reactivity, e.g. a loaded checkpoint).
Everything else is a module.

### Adding a new baseline (no notebook needed)

Open `harness/agents.py`, add a class, then register a factory in
`harness/registry.py`:

```python
# harness/agents.py
class MyAgent(BaseAgent):
    def select_move(self, board): ...

# harness/registry.py
from harness.agents import MyAgent
MODELS["my_agent"] = lambda: MyAgent()
```

That's it. Edit either file → autoreloader reruns any benchmark cell using
the dropdown.

### Adding a new LLM model (notebook required)

A notebook is only needed when you have an expensive checkpoint to hold in
memory. The notebook exposes a `make_agent` factory; the main composer
imports and registers it.

```python
# notebooks/model_llm_local.py
@app.cell
def _():
    from harness.llm_agent import LLMAgent
    from harness.hf_generate import make_hf_generate_fn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer, LLMAgent, make_hf_generate_fn

@app.cell
def _(AutoModelForCausalLM, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained("gpt2-checkers-sft")
    model = AutoModelForCausalLM.from_pretrained("gpt2-checkers-sft")
    return model, tok

@app.cell
def _(LLMAgent, make_hf_generate_fn, model, tok):
    def make_agent():
        return LLMAgent(generate_fn=make_hf_generate_fn(model, tok), record_turns=True)
    return (make_agent,)
```

The main composer registers it explicitly (avoids reactive double-register
side effects):

```python
# harness_benchmark.py
@app.cell
def _():
    from harness import registry
    from notebooks.model_llm_local import make_agent as llm_local
    registry.register("llm_local", llm_local)
    return (registry,)
```

`registry.MODELS` is now `{random, greedy, minimax_d3, minimax_d8, llm_local}`
— the dropdown reads `registry.names()`.

### How the previewer composes — `await app.embed()`

`notebooks/previewer.py` is a normal marimo notebook whose cells render the
charts and tables you like. Each cell that produces UI gets a name (or a
return value) so the main notebook can address its outputs:

```python
# in notebooks/previewer.py
import marimo
app = marimo.App()

@app.cell
def stats_input():
    # placeholder; main notebook will override `stats` via embed kwargs
    stats = None
    return (stats,)

@app.cell
def summary(mo, stats):
    summary = mo.md(f"### Win rate: {stats.win_rate:.1%}") if stats else mo.md("")
    return (summary,)

@app.cell
def table(mo, stats):
    table = mo.ui.table([...]) if stats else mo.md("")
    return (table,)

@app.cell
def chart(mo, stats):
    chart = mo.ui.altair_chart(...) if stats else mo.md("")
    return (chart,)
```

In the main notebook:

```python
# in harness_benchmark.py
@app.cell
async def _(mo, run_benchmark, MODELS, agent_choice, opp_choice, games):
    from notebooks import previewer
    stats = run_benchmark(MODELS[agent_choice.value](),
                          MODELS[opp_choice.value](),
                          games=games.value)
    embed = await previewer.app.embed()
    # render the previewer's outputs inline
    return mo.vstack([embed.output])  # or address specific cells: embed.outputs["summary"]
```

That gives you the previewer's full UI dropped into the main notebook — no
duplication, no iframes, full reactivity. When you iterate on the previewer
standalone, your changes show up in every harness that embeds it.

### Embedding individual cells vs. whole app

`embed.output` is the full notebook output. If you only want one chart from
the previewer (e.g. just the win-rate bar), give the cell a name and address
it via `embed.outputs["chart"]`. This is how you build a main notebook that
mixes-and-matches pieces of previewer/model notebooks.

## Three harnesses — what each needs

### Harness 1 — reactive benchmark (`harness_benchmark.py`)

Cells:
1. Import each model notebook's `make_agent` and build a `MODELS` dict.
2. `agent = mo.ui.dropdown(options=list(MODELS), label="Agent")`
   `opponent = mo.ui.dropdown(options=list(MODELS), label="Opponent")`
   `games = mo.ui.slider(1, 200, value=50, label="Games")`
   `run = mo.ui.run_button(label="Run benchmark")`
3. Gated on `run.value`: build agents from factories, call
   `run_benchmark(MODELS[agent.value](), MODELS[opponent.value](), games=games.value)`.
4. `embed = await previewer.app.embed()` and return `embed.output` (or specific
   `embed.outputs[name]`) to render charts.

This is `eval_win_rate.eval_win_rate()` exploded across reactive cells.

### Harness 2 — test harness (`harness_test.py`)

Drive a board manually — **do not** use `Benchmark` here, you want to see
inside each turn:

```python
board = AmericanBoard()
turns = []
while not board.game_over:
    if board.turn == agent_color:
        prompt = build_prompt(board, board.legal_moves)
        response = generate_fn(prompt)
        move = parse_move(response, board.legal_moves)
        turns.append({"prompt": prompt, "response": response,
                      "parsed": move, "legal": move is not None,
                      "fen_before": board.fen})
        if move is None: move = retry_or_fallback(...)
    else:
        move = opponent.select_move(board)
    board.push(move)

previewer.render_per_turn(turns)
```

The valid-move scorer (`harness/scoring.py`) is the legal? column.

### Harness 3 — gameplay (`harness_gameplay.py`)

Two flavors:

**(a) In-marimo (recommended).** Hold board in `mo.state`. UI = dropdown of
`board.legal_moves` + a "submit" button. On click: push human move → call
`agent.select_move(board)` → push → re-render. Lightweight, no server.

**(b) External server.** Small FastAPI app holds a board and exposes `/move`.
Wrap `MODELS[name]()` at startup. Only choose this if you need a real network
boundary (e.g. a separate web client).

Either way, gameplay reuses `harness/scoring.py` to validate the agent's move.
For board display, build a tiny `board_view.py` notebook (one cell that takes
a board and returns `mo.Html`) and `await board_view.app.embed()` from the
gameplay notebook.

## Build order (high level)

See `i_plan.md` for the phase-by-phase manual-verification version.

1. **`harness/` package** — already built in this handoff. Drop into the
   marimo project (or `pip install -e`).
2. **Reactive benchmark UI with baselines only** — proves dropdown +
   `run_benchmark` + factory pattern.
3. **Bridge SFT gpt-2 to LLMAgent** in `notebooks/model_llm_local.py` and
   smoke-test on a single board.
4. **Add LLM to the dropdown** — first real benchmark.
5. **`notebooks/previewer.py` + `await app.embed()`** — replace `print(stats)`.
6. **Harness 2 (test harness)** — per-turn introspection.
7. **Harness 3 (gameplay)** — in-marimo first, server flavor optional.

## Pitfalls

- **Marimo reactivity vs stateful agents.** `LLMAgent` accumulates `num_turns`,
  `first_try_legal`, etc. across `select_move` calls. Fresh agent per run
  (the factory pattern in `registry.MODELS`) handles this. If you instead hold
  an agent in `mo.state`, call `reset_stats()` between runs.
- **Don't port `eval_consistency.py` day 1.** Secondary metric, orthogonal
  to the three harnesses.
- **Long benchmark runs block the cell.** Already gated by `mo.ui.run_button`
  in the plan — keep it that way. For >50 games on CPU LLM inference,
  consider a progress widget.
- **Don't put trivial agents in their own notebooks.** `harness/agents.py`
  is the right home — autoreload makes it as ergonomic as a notebook with
  none of the boilerplate.
