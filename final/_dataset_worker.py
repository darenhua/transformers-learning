"""Worker functions for parallel checkers dataset generation.

Lives at module scope (not inside a marimo cell) so ProcessPoolExecutor
can pickle it. generate_dataset.py imports this and submits batches.

Each record carries `fen` + label + a small metadata bundle (phase,
piece_count, etc.) used for per-phase quota filtering and variance
reporting. The bundle is stripped before writing the jsonl so training
sees only `{fen, move/chosen/rejected, game_idx}`.
"""

import random

from draughts import AlphaBetaEngine, Board, HubEngine

MAX_MOVES_PER_GAME = 300
OPENING_THRESHOLD = 30  # >= this many pieces => opening
ENDGAME_THRESHOLD = 12  # <= this many pieces => endgame

# pydraughts 1.7.1's HubEngine._read_line uses select() + readline() on
# the text stream, which deadlocks when Scan emits several lines in one
# OS chunk: lines land in Python's TextIOWrapper buffer but select stops
# reporting the underlying fd as readable, so subsequent reads time out.
# A plain blocking readline avoids the issue; Scan responds promptly.
def _hub_blocking_read_line(self, timeout: float = 1.0):
    if self.process is None or self.process.stdout is None:
        return None
    line = self.process.stdout.readline()
    return line.strip() if line else None


HubEngine._read_line = _hub_blocking_read_line

METADATA_KEYS = (
    "phase",
    "side",
    "n_legal",
    "forced_capture",
    "piece_count",
    "kings_present",
)


def count_pieces(board):
    bb = (
        board.white_men,
        board.white_kings,
        board.black_men,
        board.black_kings,
    )
    total = sum(bin(b).count("1") for b in bb)
    kings = bin(board.white_kings).count("1") + bin(board.black_kings).count("1")
    return total, kings


def classify_phase(piece_count, opening_threshold, endgame_threshold):
    if piece_count >= opening_threshold:
        return "opening"
    if piece_count <= endgame_threshold:
        return "endgame"
    return "midgame"


def turn_from_fen(fen):
    """'[FEN "W:B:..."]' -> 'B'. base.py builds fen as W:{turn}:..., so
    splitting by ':' twice puts the side-to-move at index 1."""
    return fen.split(":", 2)[1]


def position_metadata(board, legal_moves, opening_threshold, endgame_threshold):
    pc, kings = count_pieces(board)
    return {
        "phase": classify_phase(pc, opening_threshold, endgame_threshold),
        "side": turn_from_fen(board.fen),
        "n_legal": len(legal_moves),
        "forced_capture": any("x" in str(m) for m in legal_moves),
        "piece_count": pc,
        "kings_present": kings > 0,
    }


def _random_opening(board, rng, randomize_start_plies):
    """Play 0..randomize_start_plies random legal plies. Diversifies
    the early-game distribution beyond what self-play alone produces."""
    n = rng.randint(0, randomize_start_plies)
    for _ in range(n):
        lm = list(board.legal_moves)
        if not lm:
            return
        board.push(rng.choice(lm))


def valid_move_worker(
    seed,
    n_games,
    random_ratio,
    randomize_start_plies,
    opening_threshold,
    endgame_threshold,
):
    """Self-play minimax depth-2 with random-move injection, returns
    a flat list of record dicts."""
    rng = random.Random(seed)
    engine = AlphaBetaEngine(depth_limit=2)
    records = []
    for _ in range(n_games):
        board = Board()
        _random_opening(board, rng, randomize_start_plies)
        for _ in range(MAX_MOVES_PER_GAME):
            lm = list(board.legal_moves)
            if not lm:
                break
            meta = position_metadata(
                board, lm, opening_threshold, endgame_threshold
            )
            if rng.random() >= random_ratio:
                move = engine.get_best_move(board)
            else:
                move = rng.choice(lm)
            if move is None:
                break
            records.append({"fen": board.fen, "move": str(move), **meta})
            board.push(move)
    return records


def optimal_move_worker(
    seed,
    n_games,
    scan_path,
    strong_time,
    weak_time,
    randomize_start_plies,
    opening_threshold,
    endgame_threshold,
):
    """Strong vs weak Scan; strong drives the trajectory. Returns
    (records, skipped_ties)."""
    rng = random.Random(seed)
    records = []
    skipped = 0
    with (
        HubEngine(scan_path, time_limit=strong_time, init_timeout=60.0) as strong,
        HubEngine(scan_path, time_limit=weak_time, init_timeout=60.0) as weak,
    ):
        for _ in range(n_games):
            strong.new_game()
            weak.new_game()
            board = Board()
            _random_opening(board, rng, randomize_start_plies)
            for _ in range(MAX_MOVES_PER_GAME):
                lm = list(board.legal_moves)
                if not lm:
                    break
                meta = position_metadata(
                    board, lm, opening_threshold, endgame_threshold
                )
                chosen = strong.get_best_move(board)
                if chosen is None:
                    break
                rejected = weak.get_best_move(board)
                chosen_str = str(chosen)
                if rejected is None or str(rejected) == chosen_str:
                    skipped += 1
                else:
                    records.append(
                        {
                            "fen": board.fen,
                            "chosen": chosen_str,
                            "rejected": str(rejected),
                            **meta,
                        }
                    )
                board.push(chosen)
    return records, skipped
