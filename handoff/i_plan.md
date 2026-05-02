# Incremental Execution Plan

Sequential phases. Each phase has: **Goal → Files → Steps → Verify**. Do
**not** start the next phase until the current Verify passes. Each phase is
small enough to verify by hand in 5–15 minutes.

If a Verify fails: don't plow forward. The whole point of this plan is that
problems get caught at the smallest possible boundary.

---

## Phase 0 — Drop `harness/` into the marimo project

**Goal:** establish the shared library and confirm Python imports resolve.
Also: enable marimo's module autoreloader (Settings → Runtime →
"autorun"), so editing `harness/*.py` reruns dependent cells live.

**Files:**
- Copy `handoff/harness/` → marimo project root.

**Steps:**
1. Copy the folder.
2. Enable module autoreload in marimo notebook settings.
3. Open a one-cell scratch marimo notebook with:
   ```python
   from harness.agents import RandomAgent, GreedyAgent, MinimaxAgent
   from harness.scoring import parse_move
   from harness.prompts import build_minimal_prompt
   from harness.engines import run_benchmark
   from harness.registry import MODELS, names
   from harness.llm_agent import LLMAgent
   from harness.hf_generate import make_hf_generate_fn
   names()
   ```

**Verify:** cell runs without errors and prints
`["random", "greedy", "minimax_d3", "minimax_d8"]`.

**If it fails:** check `pyproject.toml` for `py-draughts>=1.6.4`. Check that
`harness/` is at the project root (sibling to your notebooks dir).

---

## Phase 1 — One-shot benchmark in a single cell

**Goal:** prove `run_benchmark` end-to-end with two baselines, no UI yet.

**Files:** scratch notebook only.

**Steps:**
1. Add a cell:
   ```python
   from harness.engines import run_benchmark
   from harness.registry import build
   result = run_benchmark(
       agent=build("random"),
       opponent=build("minimax_d3"),
       games=2,
       opponent_name="minimax_d3",
   )
   result
   ```

**Verify:** prints a `BenchmarkResult` with non-zero `games`, plausible
`wins/losses/draws` summing to 2, finite `elo_diff`. Two games will run in
seconds.

**If it fails:** likely a py-draughts version mismatch or `MinimaxAgent`
import path. Don't continue.

---

## Phase 2 — First reactive harness notebook (Harness 1 skeleton)

**Goal:** dropdown + run-button + inline result. **No previewer yet.**

**Files:**
- New notebook `notebooks/harness_benchmark.py`.

**Steps:**
1. Cells:
   - Imports: `mo`, `harness.registry`, `harness.engines`.
   - Dropdowns + slider + run button:
     ```python
     agent_choice = mo.ui.dropdown(options=registry.names(), value="random", label="Agent")
     opp_choice   = mo.ui.dropdown(options=registry.names(), value="minimax_d3", label="Opponent")
     games        = mo.ui.slider(1, 50, value=5, label="Games")
     run          = mo.ui.run_button(label="Run benchmark")
     mo.hstack([agent_choice, opp_choice, games, run])
     ```
   - Gated cell:
     ```python
     result = None
     if run.value:
         result = run_benchmark(
             agent=registry.build(agent_choice.value),
             opponent=registry.build(opp_choice.value),
             games=games.value,
             opponent_name=opp_choice.value,
         )
     result
     ```

**Verify:** click run with random vs minimax_d3 / 5 games. See a
`BenchmarkResult`. Change the dropdown — *result does not auto-rerun* (the
button gates it). That's correct.

**If it fails:** check that `run` is in the gated cell's parameter list (so
marimo's reactive graph triggers re-evaluation when it's clicked).

---

## Phase 3 — Sanity-check `greedy` (already in `harness/agents.py`)

**Goal:** verify autoreload + the consolidated agents module work end-to-end
*without* writing a model notebook.

`GreedyAgent` is already in `harness/agents.py` and pre-registered as
`"greedy"` in `harness/registry.py`. No new file needed — this phase is just
a reality check that the module-first design is wired correctly.

**Files:** no new files.

**Steps:**
1. In `harness_benchmark.py`, confirm the dropdowns read from
   `registry.names()` so they pick up `"greedy"` automatically. (If they
   were hard-coded to `["random", "minimax_d3", "minimax_d8"]` from earlier,
   change to `registry.names()`.)
2. Run greedy vs random / 5 games.

**Verify:** greedy wins most games against random. Now edit
`harness/agents.py` — change `GreedyAgent.select_move` to be deliberately
bad (e.g. `min` instead of `max`). The benchmark cell should mark stale or
auto-rerun (depending on autoreload setting).

**If autoreload doesn't trigger:** check Settings → Runtime → autoreload is
on. The benchmark cell must transitively import `harness.agents` (it does,
via `registry`).

**Why this matters:** confirms the "edit module → cell reruns" loop. Every
later change to scoring/prompts/agents will rely on this.

---

## Phase 4 — Bridge the SFT checkpoint into `LLMAgent`

**Goal:** prove the trained gpt-2 produces *a legal move* on a real board,
*before* you put it anywhere near the benchmark.

**Files:**
- New notebook `notebooks/model_llm_local.py`.

**Steps:**
1. Move the gpt-2 load cells from `transformers_learn.py` into this notebook.
2. Add:
   ```python
   @app.cell
   def _(gpt2_model, gpt2_tokenizer):
       from harness.hf_generate import make_hf_generate_fn
       from harness.llm_agent import LLMAgent
       generate_fn = make_hf_generate_fn(gpt2_model, gpt2_tokenizer)
       def make_agent():
           return LLMAgent(generate_fn=generate_fn, record_turns=True)
       return (make_agent,)
   ```
3. **Smoke cell** in the same notebook:
   ```python
   @app.cell
   def _(make_agent):
       from draughts import AmericanBoard
       agent = make_agent()
       board = AmericanBoard()
       move = agent.select_move(board)
       turn = agent.turns[-1]
       return move, turn  # inspect both
   ```

**Verify:** `move` is a real `Move` object. Look at `turn.responses[0]` —
it should be a string containing something resembling a checkers move
(`32-28`, `21-17`, etc.). `turn.legal_first_try` may be False if the SFT
training was tiny — that's OK, `chosen_move` will show the random fallback.

**If `turn.responses[0]` is gibberish (e.g. natural-language text):** prompt
format mismatch. Confirm `make_hf_generate_fn` is being called and that
`harness.llm_agent.LLMAgent` defaults to `build_minimal_prompt`. **Stop and
fix here** — every later phase depends on this working.

**If `turn.legal_first_try` is False most of the time:** that's fine for now.
The model is undertrained. The fallback path keeps the harness functional.
Only worth fixing once you've trained on more data.

---

## Phase 5 — LLM in the benchmark dropdown

**Goal:** the moment of truth — the trained model in the actual benchmark
loop.

**Files:** edit `harness_benchmark.py`.

**Steps:**
1. Add to the registration cell:
   ```python
   from notebooks.model_llm_local import make_agent as llm_local_factory
   registry.register("llm_local", llm_local_factory)
   ```
2. Run a benchmark: `llm_local` vs `random`, **3 games** (start tiny — LLM
   inference is slow on CPU).

**Verify:** benchmark completes without crashing. `result.legal_move_rate`
and `result.fallback_rate` are populated (the engines.py wrapper detects
LLMAgent and snapshots its stats). Win rate may be terrible — that's a
training problem, not a harness problem.

**If it crashes mid-game:** likely an unparseable response on a turn the
random fallback couldn't recover from. Inspect `agent.turns` to see what
happened. The harness is doing its job.

**If inference is too slow:** drop `games=1` and `max_new_tokens=8` in
`make_hf_generate_fn`. CPU inference on gpt-2 for full games takes minutes.

---

## Phase 6 — Previewer notebook + `app.embed()`

**Goal:** stop using `print(result)`. Build the previewer once, embed
everywhere.

**Files:**
- New notebook `notebooks/previewer.py`.

**Steps:**
1. In `previewer.py`:
   ```python
   import marimo
   app = marimo.App()

   @app.cell
   def _():
       import marimo as mo
       return (mo,)

   @app.cell
   def _():
       # placeholder; main notebook overrides via embed
       result = None
       return (result,)

   @app.cell
   def summary(mo, result):
       if result is None:
           summary = mo.md("*no result yet*")
       else:
           summary = mo.md(
               f"**{result.opponent}** — "
               f"{result.wins}W-{result.losses}L-{result.draws}D · "
               f"win {result.win_rate:.1%} · elo {result.elo_diff:+.0f}"
           )
       return (summary,)

   @app.cell
   def llm_stats(mo, result):
       if result is None or result.legal_move_rate is None:
           llm_stats = mo.md("")
       else:
           llm_stats = mo.md(
               f"legal-move rate: {result.legal_move_rate:.1%} · "
               f"fallback rate: {result.fallback_rate:.1%}"
           )
       return (llm_stats,)
   ```
2. **Standalone-verify** the previewer first: open `previewer.py` directly,
   manually set `result` to a fake `BenchmarkResult(...)`, see the rendering.
3. Then in `harness_benchmark.py`:
   ```python
   @app.cell
   async def _(mo, result):
       from notebooks import previewer
       embed = await previewer.app.embed()
       return mo.vstack([embed.output])
   ```

**Verify:** previewer cells render inline in the benchmark notebook.

**If `embed.output` is empty / shows the placeholder:** the previewer is
running with its own placeholder, not your `result`. Pass it via embed
kwargs (consult marimo docs for the exact form your version supports), or
turn the previewer's `result` cell into a parameter the embedder fills.

---

## Phase 7 — Test harness (Harness 2)

**Goal:** see inside a single LLM-vs-baseline game, turn by turn.

**Files:**
- New notebook `notebooks/harness_test.py`.

**Steps:**
1. Cells:
   - Dropdowns for agent + opponent (reuse registry).
   - Run button.
   - Gated cell that drives a game manually:
     ```python
     from draughts import AmericanBoard, Color
     board = AmericanBoard()
     agent = registry.build(agent_choice.value)
     opp = registry.build(opp_choice.value)
     while not board.game_over:
         mover = agent if board.turn == Color.WHITE else opp
         board.push(mover.select_move(board))
     turns = getattr(agent, "turns", [])
     ```
   - Render `turns` as a `mo.ui.table` with columns: `fen_before`,
     `responses[0]` (raw), `parsed`, `legal_first_try`, `chosen_move`.

**Verify:** run llm_local vs random. Table shows ~20–60 rows. Eyeball:
- Are responses well-formed move strings?
- Are illegal-first-try turns falling back correctly?
- Does `chosen_move` match a legal move at each FEN?

**If the agent is `record_turns=False`:** `turns` will be empty. Make sure
your `make_agent` in `model_llm_local.py` passes `record_turns=True` (the
template above does).

---

## Phase 8 — Gameplay (Harness 3, in-marimo flavor)

**Goal:** play a game by hand against the LLM.

**Files:**
- New notebook `notebooks/harness_gameplay.py`.

**Steps:**
1. State cell (board held across reactivity):
   ```python
   get_state, set_state = mo.state({"board": AmericanBoard(), "agent": None})
   ```
2. Reset / agent-select cell: dropdown over `registry.names()` + reset
   button. On reset: `set_state({"board": AmericanBoard(), "agent": registry.build(agent_choice.value)})`.
3. Move cell:
   ```python
   s = get_state()
   board = s["board"]
   move_choice = mo.ui.dropdown(
       options={str(m): m for m in board.legal_moves},
       label="Your move",
   )
   submit = mo.ui.run_button(label="Play")
   mo.hstack([move_choice, submit])
   ```
4. Apply cell (gated on `submit.value`):
   - Push human move.
   - If not game over: push `s["agent"].select_move(board)`.
   - `set_state({"board": board, "agent": s["agent"]})`.
5. Display cell: render `s["board"]` (try `mo.md(f"```\n{board}\n```")`
   for a quick textual rendering).

**Verify:** play 3–5 moves. Board updates after each click. LLM responds.
You can win or lose (you will lose).

**If state isn't persisting across clicks:** marimo `mo.state` requires
careful dependency wiring. The state-getter and state-setter must be
returned from the same cell; the apply cell depends on `set_state` and the
display cell depends on `get_state`.

---

## Phase 9 (optional) — Server flavor of gameplay

**Goal:** the `Server`-based flavor (lets external clients play).

**Files:** edit `harness_gameplay.py` (or split to `harness_server.py`).

**Steps:**
1. Reuse the threaded `Server` cell from `transformers_learn.py:72–87`.
2. Replace `AlphaBetaEngine(depth_limit=6)` with
   `registry.build(agent_choice.value).as_engine()`.

**Verify:** server prints "started." Connect a draughts UI client (or curl
the relevant endpoint) and play a move.

**If the server thread dies silently:** `daemon=True` swallows exceptions —
temporarily set `daemon=False` and remove the threading to see the traceback.

---

## Build-order summary (one line per phase)

0. Copy `harness/`, enable autoreload, smoke-test imports.
1. `run_benchmark` works in a scratch cell.
2. Reactive benchmark UI with baselines only.
3. Sanity-check greedy + autoreload (no new files).
4. Bridge SFT gpt-2 to `LLMAgent`, smoke-test on one board.
5. LLM in the benchmark dropdown — 3-game run.
6. Previewer notebook + `await app.embed()`.
7. Test harness with per-turn introspection.
8. In-marimo gameplay.
9. (Optional) Server-based gameplay.

## Stop-and-fix discipline

- Phase 4 is the riskiest. If the model produces gibberish, **stop**.
  Either retrain on the verbose prompt or tweak `make_hf_generate_fn`'s
  decoding params, but don't keep building harnesses on a broken model.
- Phase 6 is where most marimo-specific pain shows up. If `embed.output`
  doesn't behave as you expect, fall back to importing previewer's render
  *functions* directly (declare them at module scope alongside `app =
  marimo.App()`) — that path always works.
- Don't skip the standalone verification of each notebook (`previewer.py`
  on its own, `model_llm_local.py` on its own) before composing them.
