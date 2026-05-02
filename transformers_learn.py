import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    aight so now mane,
    we gon do some bullshit pydraughts ideation bullshit
    """)
    return


@app.cell
def _():
    import draughts

    return (draughts,)


@app.cell
def _(draughts):
    board = draughts.Board()
    print([str(m) for m in board.legal_moves])
    # legal move
    board.push_uci("31-27")
    board.turn
    print(type(board.fen))
    return (board,)


@app.cell
def _(board):
    print(board.legal_moves)
    max_move = max(board.legal_moves, key=lambda m: len(m.captured_list))
    print(max_move)
    return


@app.cell
def _():
    from draughts import Board, Server, AlphaBetaEngine, HubEngine, Benchmark, BaseAgent

    return AlphaBetaEngine, BaseAgent, Benchmark, Board, HubEngine, Server


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## server shit... this lets me start up a server
    """)
    return


@app.cell
def _(mo):
    start_server_button = mo.ui.run_button(label="Start server")
    start_server_button
    return (start_server_button,)


@app.cell
def _(AlphaBetaEngine, Board, Server, start_server_button):
    import threading

    if start_server_button.value:
        server = Server(
            board=Board(),
            white_engine=AlphaBetaEngine(depth_limit=6),
            black_engine=AlphaBetaEngine(depth_limit=6),
        )

        def _run():
            server.run()  # this thread has no running loop, so asyncio.run() works fine

        threading.Thread(target=_run, daemon=True).start()
        print("server started")
    return


@app.cell
def _(mo):
    run_benchmark_button = mo.ui.run_button(label="Run benchmark")
    run_benchmark_button
    return (run_benchmark_button,)


@app.cell
def _(AlphaBetaEngine, BaseAgent, Benchmark, Board, run_benchmark_button):
    if run_benchmark_button.value:
        class GreedyAgent(BaseAgent):
            def select_move(self, board):
                chosen_move = max(board.legal_moves, key=lambda m: len(m.captured_list))
                print(chosen_move)
                return chosen_move

        board_2 = Board()
        agent = GreedyAgent()
        move = agent.select_move(board_2)
        print(move)

        stats = Benchmark(agent.as_engine(), AlphaBetaEngine(depth_limit=4), games=10).run()
        print(stats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ok so now wtf we doing...
    ## lets get into gadamn dataset generation gadamittt
    what that mean tho? that mean we gotta take the draught board input and stuff it as an input context string
    """)
    return


@app.cell
def hubblock(AlphaBetaEngine, Board, HubEngine):
    import json
    import random

    MAX_MOVES_PER_GAME = 300
    SCAN_PATH = "scan/scan_31/scan_linux"

    # pydraughts 1.7.1's HubEngine._read_line uses select() + readline() on the
    # text stream, which deadlocks when Scan emits several lines in one OS chunk:
    # the lines land in Python's TextIOWrapper buffer, but select stops reporting
    # the underlying fd as readable, so subsequent reads time out forever. Swap
    # in a plain blocking readline; Scan responds promptly.
    def _hub_blocking_read_line(self, timeout: float = 1.0):
        if self.process is None or self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        return line.strip()
    HubEngine._read_line = _hub_blocking_read_line


    def generate_valid_move_dataset(num_games=1000, output_path="valid_move_dataset.jsonl"):
        engine = AlphaBetaEngine(depth_limit=2)
        records = []
        for i in range(num_games):
            if i % 100 == 0:
                print(f"Valid move game {i}/{num_games}")
            board = Board()
            for _ in range(MAX_MOVES_PER_GAME):
                if not board.legal_moves:
                    break
                if i % 5 != 0:
                    move = engine.get_best_move(board)
                else:
                    move = random.choice(list(board.legal_moves))
                if move is None:
                    break
                records.append({
                    "fen": board.fen,
                    "move": str(move),
                    "game_idx": i,
                })
                board.push(move)

        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records)} records to {output_path}")


    def generate_optimal_move_dataset(
        num_games=1000,
        output_path="optimal_move_dataset.jsonl",
        strong_time=0.3,
        weak_time=0.05,
    ):
        records = []
        skipped = 0
        with HubEngine(SCAN_PATH, time_limit=strong_time, init_timeout=60.0) as strong_engine, \
             HubEngine(SCAN_PATH, time_limit=weak_time, init_timeout=60.0) as weak_engine:
            for i in range(num_games):
                if i % 10 == 0:
                    print(f"Optimal move game {i}/{num_games}")
                strong_engine.new_game()
                weak_engine.new_game()
                board = Board()
                for _ in range(MAX_MOVES_PER_GAME):
                    if not board.legal_moves:
                        break
                    chosen = strong_engine.get_best_move(board)
                    if chosen is None:
                        break
                    rejected = weak_engine.get_best_move(board)
                    chosen_str = str(chosen)
                    if rejected is None or str(rejected) == chosen_str:
                        skipped += 1
                    else:
                        records.append({
                            "fen": board.fen,
                            "chosen": chosen_str,
                            "rejected": str(rejected),
                            "game_idx": i,
                        })
                    board.push(chosen)

        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records)} records to {output_path} ({skipped} skipped)")

    return generate_optimal_move_dataset, generate_valid_move_dataset


@app.cell
def _(AlphaBetaEngine, Board):
    print("=== smoke test: 2 games each ===")

    _b = Board()
    _engine = AlphaBetaEngine(depth_limit=2)
    _m = _engine.get_best_move(_b)
    print(f"sanity: fen={_b.fen!r}")
    print(f"sanity: best_move={_m!r}, str={str(_m)!r}")
    _b.push(_m)
    print(f"sanity: post-push fen={_b.fen!r}")
    return


@app.cell
def _(mo):
    # title = mo.md("1000 game dataset generation")
    run_full_button = mo.ui.run_button(
        label="sft (minimax depth2 versus itself)"
    )
    run_optimal_button = mo.ui.run_button(
        label="dpo (minimax depth2 versus scan)"
    )
    mo.ui.array([
        run_full_button,
        run_optimal_button
    ])
    return run_full_button, run_optimal_button


@app.cell
def _(
    generate_optimal_move_dataset,
    generate_valid_move_dataset,
    run_full_button,
    run_optimal_button,
):
    if run_full_button.value:    
        generate_valid_move_dataset(num_games=20, output_path="valid_move_smoke.jsonl")
    if run_optimal_button.value:
        generate_optimal_move_dataset(num_games=20, output_path="optimal_move_smoke.jsonl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## convert to TRL SFT prompt/completion format and train tiny GPT-2 (124M)

    important: for cpu!
    """)
    return


@app.cell
def _():
    import os
    from datasets import load_dataset

    _path = (
        "valid_move_dataset.jsonl"
        if os.path.exists("valid_move_dataset.jsonl")
        else "valid_move_smoke.jsonl"
    )
    print(f"loading {_path}")
    _raw = load_dataset("json", data_files=_path, split="train")

    def _to_sft(example):
        return {
            "prompt": f"{example['fen']}\nMove:",
            "completion": f" {example['move']}",
        }

    training_dataset = _raw.map(_to_sft, remove_columns=_raw.column_names)
    print(training_dataset)
    print(training_dataset[0])
    return (training_dataset,)


@app.cell
def _():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2 has no pad token; reuse eos so the trainer's collator can pad batches
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id
    return gpt2_model, gpt2_tokenizer


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train GPT-2 on checkers SFT (slow)")
    train_button
    return (train_button,)


@app.cell
def _(gpt2_model, gpt2_tokenizer, train_button, training_dataset):
    if train_button.value:
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=gpt2_model,
            processing_class=gpt2_tokenizer,
            train_dataset=training_dataset,
            args=SFTConfig(
                use_cpu=True,
                output_dir="gpt2-checkers-sft",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                logging_steps=1,
                logging_first_step=True,
                save_strategy="epoch",
            ),
        )
        trainer.train()
    return


if __name__ == "__main__":
    app.run()
