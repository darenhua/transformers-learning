"""Shared primitives for the marimo harness.

Module-first design (with marimo's module autoreloader, edits to these files
re-run dependent cells automatically):

- agents:     RandomAgent, GreedyAgent, MinimaxAgent — stateless baselines
- llm_agent:  LLMAgent — model-swap point, takes generate_fn + prompt_fn
- scoring:    parse_move / is_legal — pass/fail validity scorer
- prompts:    build_minimal_prompt / build_verbose_prompt — must match training format
- engines:    run_benchmark — py-draughts Benchmark wrapper, stdout suppressed
- registry:   MODELS dict + register() — dropdown source of truth
- hf_generate: make_hf_generate_fn — bridges HF transformers checkpoint to LLMAgent

Things that stay in notebooks (not modules):
- model_llm_local — loads gpt-2 weights (expensive state)
- previewer — UI cells used via `await app.embed()`
- harness_benchmark / harness_test / harness_gameplay — user-facing UIs
"""
