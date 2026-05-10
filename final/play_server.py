"""Spin up the py-draughts FastAPI server with two configurable engines.

Default matchup: robust Qwen-DPO LLM engine (white) vs minimax depth-3
(black) on port 4321. Each LLM turn is printed straight to this process's
stdout in "You: <FEN>\\n AI: <move>\\n\\n" form so the server's terminal is
the conversation log — `tail -f` on the terminal, or just watch it scroll.

Engine specs (`--white` / `--black`):
    random              uniform random over legal moves
    minimax[:N]         alpha-beta search at depth N (default 3)
    scan[:T]            Scan hub engine with T seconds per move (default 0.3)
    llm:<path>          our trained LLM (path is a checkpoint dir or HF id),
                        wrapped in the robust engine: 5 retries on illegal
                        moves with sampling-on-retry, then a random fallback
                        from `board.legal_moves` if all retries fail. The
                        engine NEVER returns an illegal move.

Run from `final/`:
    .env/bin/python play_server.py
    .env/bin/python play_server.py --white scan:0.5 --black minimax:4
    .env/bin/python play_server.py --white llm:checkpoints/qwen-dpo --black scan
    .env/bin/python play_server.py --port 8080 --host 127.0.0.1

Open http://<host>:<port>/ in a browser for the bundled py-draughts UI, or
hit JSON routes directly (/position, /best_move, POST /move/{src}/{tgt}).

Stop with Ctrl-C — Server.run wraps uvicorn, which calls
_cleanup_engines() on exit to shut Scan down cleanly.
"""

import argparse
import sys
from pathlib import Path

# Resolve sibling imports regardless of cwd.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "training-jobs"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _patch_hub_engine_read_line():
    """pydraughts 1.7.1's HubEngine._read_line uses select() + readline() on
    the text stream, which deadlocks on Scan startup when several lines land
    in one OS chunk: lines sit in the TextIOWrapper buffer but select stops
    flagging the fd as readable, so subsequent reads time out forever (which
    is exactly what `TimeoutError: Engine initialization timed out` is). A
    plain blocking readline avoids it. Same patch as `_dataset_worker.py`.
    """
    from draughts import HubEngine

    def _blocking_read_line(self, timeout: float = 1.0):
        if self.process is None or self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        return line.strip() if line else None

    HubEngine._read_line = _blocking_read_line


def make_engine(
    spec: str,
    scan_path: str,
    llm_max_tokens: int = 12,
    llm_side_label: str = "AI",
    llm_log_to_stdout: bool = True,
):
    """Parse an engine spec string and return a pydraughts Engine.

    Accepts: 'random', 'minimax', 'minimax:N', 'scan', 'scan:T', 'llm:<path>'.
    """
    from draughts import AlphaBetaEngine, HubEngine

    if ":" in spec:
        head, tail = spec.split(":", 1)
    else:
        head, tail = spec, ""

    if head == "random":
        # pydraughts' BaseAgent has no built-in RandomAgent registered as an
        # engine; the simplest random move-picker is one we already use in
        # harness.agents — but to keep this script standalone, define inline.
        from draughts import BaseAgent

        class _RandomAgent(BaseAgent):
            def select_move(self, board):
                import random as _rand
                return _rand.choice(list(board.legal_moves))

        return _RandomAgent().as_engine()

    if head == "minimax":
        depth = int(tail) if tail else 3
        return AlphaBetaEngine(depth_limit=depth)

    if head == "scan":
        time_limit = float(tail) if tail else 0.3
        if not Path(scan_path).exists():
            raise FileNotFoundError(f"Scan binary missing at {scan_path}")
        return HubEngine(scan_path, time_limit=time_limit, init_timeout=60.0)

    if head == "llm":
        if not tail:
            raise ValueError("llm spec needs a checkpoint path: llm:<path>")
        from harness.llm_engine import make_robust_llm_engine

        # max_retries=5 + sampling-on-retry + random fallback live inside the
        # agent — engine never returns an illegal move. log_to_stdout streams
        # "You: <FEN>\nAI: <move>" lines into the server's terminal.
        _agent, eng = make_robust_llm_engine(
            tail,
            max_retries=5,
            max_new_tokens=llm_max_tokens,
            log_to_stdout=llm_log_to_stdout,
            side_label=llm_side_label,
        )
        # Override engine name so the server UI shows the checkpoint, not
        # "LoggingLLMAgent".
        eng.name = f"llm:{tail}"
        return eng

    raise ValueError(f"unknown engine spec: {spec!r}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--white", default="llm:checkpoints/qwen-dpo",
                    help="Engine spec for white. See header.")
    ap.add_argument("--black", default="minimax:3", help="Engine spec for black.")
    ap.add_argument("--scan-path", default="../scan/scan_31/scan_linux")
    ap.add_argument("--llm-max-tokens", type=int, default=12,
                    help="Generation cap for any 'llm:' engines.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=4321)
    a = ap.parse_args()

    from draughts import Board, Server

    _patch_hub_engine_read_line()

    print(f"[play_server] white={a.white!r} black={a.black!r}", flush=True)
    print(f"[play_server] building white engine ({a.white!r})…", flush=True)
    white = make_engine(
        a.white, a.scan_path, a.llm_max_tokens,
        llm_side_label="AI(white)", llm_log_to_stdout=True,
    )
    print(f"[play_server] white ready: {type(white).__name__}", flush=True)
    print(f"[play_server] building black engine ({a.black!r})…", flush=True)
    black = make_engine(
        a.black, a.scan_path, a.llm_max_tokens,
        llm_side_label="AI(black)", llm_log_to_stdout=True,
    )
    print(f"[play_server] black ready: {type(black).__name__}", flush=True)

    # Server.__init__ auto-starts any not-yet-started HubEngine; we don't
    # need to call .start() ourselves.
    server = Server(board=Board(), white_engine=white, black_engine=black)

    # FastAPI lifespan hook fires AFTER uvicorn has bound the socket and is
    # accepting connections. Without this, our pre-run print is "about to
    # bind" not "ready", and stdout has no clean "you can now connect" line.
    @server.APP.on_event("startup")
    async def _ready():
        print(
            f"[play_server] READY — accepting connections on "
            f"http://{a.host}:{a.port}/",
            flush=True,
        )

    print(
        f"[play_server] starting uvicorn on {a.host}:{a.port} "
        f"(Ctrl-C to stop)",
        flush=True,
    )
    try:
        # log_level="info" makes uvicorn print its own "Uvicorn running on
        # http://..." banner so you have two confirmations the bind worked.
        server.run(host=a.host, port=a.port, log_level="info")
    except KeyboardInterrupt:
        print("\n[play_server] shutting down")


if __name__ == "__main__":
    main()
