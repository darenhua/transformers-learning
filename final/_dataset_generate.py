"""High-level dataset orchestration: parallel generation + per-phase
quota fill + variance reporting. Used by both generate_dataset.py (the
marimo notebook) and generate_dataset_cli.py (headless CLI for the
background runner). Per-process workers live in _dataset_worker."""

import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# Sibling import: ensure final/ is on sys.path whether the caller cwd is
# final/ or the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _dataset_worker as _w  # noqa: E402

OUTPUT_FIELDS_SFT = ("fen", "move", "game_idx")
OUTPUT_FIELDS_DPO = ("fen", "chosen", "rejected", "game_idx")
PHASES = ("opening", "midgame", "endgame")


def _strip(record, output_fields):
    return {k: record[k] for k in output_fields if k in record}


def write_records(records, output_base, test_size, seed, output_fields):
    """Write records as a single jsonl, or split into <base>_train.jsonl
    + <base>_test.jsonl if test_size > 0. Strips per-record metadata
    (phase, side, n_legal, ...) so downstream training only sees the
    training-relevant columns."""
    slim = [_strip(r, output_fields) for r in records]
    if test_size <= 0 or not slim:
        path = f"{output_base}.jsonl"
        with open(path, "w") as f:
            for r in slim:
                f.write(json.dumps(r) + "\n")
        return [(path, len(slim))]

    rng = random.Random(seed)
    idxs = list(range(len(slim)))
    rng.shuffle(idxs)
    n_test = max(1, int(round(len(slim) * test_size)))
    test_idx = set(idxs[:n_test])
    train, test = [], []
    for i, r in enumerate(slim):
        (test if i in test_idx else train).append(r)
    paths = []
    for split_name, rs in [("train", train), ("test", test)]:
        p = f"{output_base}_{split_name}.jsonl"
        with open(p, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        paths.append((p, len(rs)))
    return paths


def compute_quotas(target_total, opening_pct, endgame_pct):
    """Returns {phase: count}. Midgame absorbs whatever's left so the sum
    equals target_total exactly."""
    opening = int(round(target_total * opening_pct / 100.0))
    endgame = int(round(target_total * endgame_pct / 100.0))
    midgame = target_total - opening - endgame
    return {"opening": opening, "midgame": midgame, "endgame": endgame}


def variance_summary(records):
    """Per-record metadata is preserved on `records` (stripped only at
    write time), so we can compute exact post-quota variance stats."""
    n = len(records)
    if n == 0:
        return "_no records_"
    side_W = sum(1 for r in records if r.get("side") == "W")
    forced = sum(1 for r in records if r.get("forced_capture"))
    single = sum(1 for r in records if r.get("n_legal") == 1)
    kings = sum(1 for r in records if r.get("kings_present"))
    pcs = [r.get("piece_count", 0) for r in records]
    avg_pc = sum(pcs) / n
    phase_counts = {p: sum(1 for r in records if r.get("phase") == p) for p in PHASES}
    bins = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40)]
    bin_counts = {
        f"{lo}-{hi}": sum(1 for pc in pcs if lo <= pc <= hi) for lo, hi in bins
    }
    return {
        "total": n,
        "phase": phase_counts,
        "side_W_pct": 100.0 * side_W / n,
        "forced_capture_pct": 100.0 * forced / n,
        "single_legal_pct": 100.0 * single / n,
        "kings_present_pct": 100.0 * kings / n,
        "piece_count_avg": avg_pc,
        "piece_count_min": min(pcs),
        "piece_count_max": max(pcs),
        "piece_count_bins": bin_counts,
    }


def _take_into_buckets(records, buckets, quotas):
    kept = 0
    for r in records:
        p = r["phase"]
        if len(buckets[p]) < quotas[p]:
            buckets[p].append(r)
            kept += 1
    return kept


def _quotas_met(buckets, quotas):
    return all(len(buckets[p]) >= quotas[p] for p in quotas)


def _finalize(buckets, quotas, seed):
    out = []
    for p in PHASES:
        out.extend(buckets[p][: quotas[p]])
    random.Random(seed + 1).shuffle(out)
    for i, r in enumerate(out):
        r["game_idx"] = i
    return out


def _resolve_workers(max_workers):
    return max_workers or (os.cpu_count() or 4)


def _streaming_parallel(
    submit_one, take_result, n_workers, target_total,
    quotas, buckets, label, max_jobs,
):
    """Keep a small queue of in-flight jobs; as each completes, drain
    records into per-phase buckets and refill until quotas are met or
    the job ceiling is hit. Renders a tqdm bar tracking records (auto-
    ETA), with phase-bucket fill levels in the postfix."""
    from tqdm.auto import tqdm

    ctx = multiprocessing.get_context("fork")
    in_flight_target = max(2, n_workers * 2)
    ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    in_flight = []
    jobs_submitted = [0]

    def _maybe_submit():
        if jobs_submitted[0] >= max_jobs:
            return None
        jobs_submitted[0] += 1
        return submit_one(ex)

    pbar = tqdm(total=target_total, desc=label, unit="rec", smoothing=0.1)
    t0 = time.monotonic()
    try:
        for _ in range(in_flight_target):
            fut = _maybe_submit()
            if fut is None:
                break
            in_flight.append(fut)
        while in_flight and not _quotas_met(buckets, quotas):
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.remove(fut)
                kept = take_result(fut.result())
                pbar.update(min(kept, max(0, target_total - pbar.n)))
                pbar.set_postfix(
                    {
                        "O/M/E": "/".join(
                            str(len(buckets[p])) for p in PHASES
                        ),
                        "jobs": jobs_submitted[0],
                    }
                )
            if _quotas_met(buckets, quotas):
                break
            while len(in_flight) < in_flight_target:
                fut = _maybe_submit()
                if fut is None:
                    break
                in_flight.append(fut)
    finally:
        pbar.close()
        ex.shutdown(wait=True, cancel_futures=True)
    return time.monotonic() - t0, jobs_submitted[0]


def generate_valid_move_dataset(
    target_total,
    output_base,
    test_size,
    random_ratio,
    seed,
    opening_pct=30,
    endgame_pct=30,
    opening_threshold=30,
    endgame_threshold=12,
    randomize_start_plies=4,
    max_workers=None,
    games_per_worker=10,
    max_jobs=2000,
):
    """SFT generator. Streams jobs across `max_workers` processes, each
    running `games_per_worker` self-play games. Stops when all per-phase
    quotas are met (or `max_jobs` jobs have been submitted as a safety
    cap). Returns (paths, variance, games_played)."""
    n_workers = _resolve_workers(max_workers)
    quotas = compute_quotas(target_total, opening_pct, endgame_pct)
    buckets = {p: [] for p in PHASES}
    rng = random.Random(seed)
    games_done = [0]

    def submit_one(ex):
        s = rng.randint(0, 2**31 - 1)
        return ex.submit(
            _w.valid_move_worker,
            s,
            games_per_worker,
            random_ratio,
            randomize_start_plies,
            opening_threshold,
            endgame_threshold,
        )

    def take_result(records):
        games_done[0] += games_per_worker
        return _take_into_buckets(records, buckets, quotas)

    elapsed, jobs = _streaming_parallel(
        submit_one, take_result, n_workers, target_total,
        quotas, buckets, "valid_move", max_jobs,
    )
    records = _finalize(buckets, quotas, seed)
    var = variance_summary(records)
    print(
        f"valid_move: {len(records)}/{target_total} records in "
        f"{elapsed:.1f}s ({games_done[0]} games, {jobs} jobs, "
        f"{n_workers} workers)"
    )
    return (
        write_records(records, output_base, test_size, seed, OUTPUT_FIELDS_SFT),
        var,
        games_done[0],
    )


def generate_optimal_move_dataset(
    target_total,
    output_base,
    test_size,
    scan_path,
    strong_time,
    weak_time,
    seed,
    opening_pct=30,
    endgame_pct=30,
    opening_threshold=30,
    endgame_threshold=12,
    randomize_start_plies=4,
    max_workers=None,
    games_per_worker=10,
    max_jobs=2000,
):
    """DPO generator. Each worker spawns its own pair of Scan
    subprocesses (strong + weak); jobs are sized large enough
    (`games_per_worker`) to amortize the ~few-second Scan boot."""
    n_workers = _resolve_workers(max_workers)
    quotas = compute_quotas(target_total, opening_pct, endgame_pct)
    buckets = {p: [] for p in PHASES}
    rng = random.Random(seed)
    games_done = [0]
    skipped_total = [0]

    def submit_one(ex):
        s = rng.randint(0, 2**31 - 1)
        return ex.submit(
            _w.optimal_move_worker,
            s,
            games_per_worker,
            scan_path,
            strong_time,
            weak_time,
            randomize_start_plies,
            opening_threshold,
            endgame_threshold,
        )

    def take_result(payload):
        recs, skipped = payload
        games_done[0] += games_per_worker
        skipped_total[0] += skipped
        return _take_into_buckets(recs, buckets, quotas)

    elapsed, jobs = _streaming_parallel(
        submit_one, take_result, n_workers, target_total,
        quotas, buckets, "optimal_move", max_jobs,
    )
    records = _finalize(buckets, quotas, seed)
    var = variance_summary(records)
    print(
        f"optimal_move: {len(records)}/{target_total} records in "
        f"{elapsed:.1f}s ({games_done[0]} games, {jobs} jobs, "
        f"{n_workers} workers, skipped {skipped_total[0]} ties)"
    )
    return (
        write_records(records, output_base, test_size, seed, OUTPUT_FIELDS_DPO),
        var,
        games_done[0],
    )
