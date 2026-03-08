import argparse
import cProfile
import os
import re
import time
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from mini_map_reduce import MapReduceEngine


DATA_DIR = Path("benchmark_data")
DATA_DIR.mkdir(exist_ok=True)

# FILE_SIZES_MB: list[int] = [50, 200, 500]
FILE_SIZES_MB: list[int] = [500]
BIG_FILE_SIZE_MB: int = 5_000
VOCAB_SIZE = 5_000
WORDS_PER_LINE = 20
WORKER_COUNTS: list[int] = [1, 2, 4, 8, 12]
#WORKER_COUNTS: list[int] = [12]

CHUNK_SIZES: list[int] = [500, 5_000, 50_000]
#CHUNK_SIZES: list[int] = [100_000]

PROFILE_MASTER: bool = False

RUN_BASELINE: bool = False


def mapper(text: str) -> Iterable[tuple[str, int]]:
    words = re.findall(r"\w+", text.lower())
    return ((word, 1) for word in words)


def reducer(word: str, counts: Iterable[int]) -> tuple[str, int]:
    return word, sum(counts)


def data_file_for_size(size_mb: int) -> Path:
    return DATA_DIR / f"big_data_input_{size_mb}mb.txt"


def generate_data(size_mb: int) -> Path:
    file_path = data_file_for_size(size_mb)
    target_bytes = size_mb * 1024 * 1024

    if file_path.exists():
        existing_size = file_path.stat().st_size
        if abs(existing_size - target_bytes) <= target_bytes * 0.01:
            print(f"[+] Reusing existing data file for {size_mb}MB: {file_path}")
            return file_path

    print(f"[*] Generating {size_mb}MB of data at {file_path} ...")
    words = [f"word{i}" for i in range(VOCAB_SIZE)]

    bytes_written = 0
    with file_path.open("w") as f:
        while bytes_written < target_bytes:
            line = " ".join(random.choices(words, k=WORDS_PER_LINE)) + "\n"
            f.write(line)
            bytes_written += len(line)

    actual_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"[+] Data generation complete: {actual_size_mb:.2f}MB written.")
    return file_path


def load_data(path: Path) -> list[str]:
    with path.open("r") as f:
        return f.readlines()


def run_sequential(data: list[str]) -> tuple[float, int]:
    start = time.perf_counter()
    counts: Counter[str] = Counter()
    for line in data:
        words = re.findall(r"\w+", line.lower())
        counts.update(words)
    end = time.perf_counter()
    return end - start, len(counts)


def run_engine(
    data: list[str],
    workers: int,
    chunk_size: int,
    *,
    use_combiner: bool = False,
) -> tuple[float, int]:
    engine = MapReduceEngine(concurrency_limit=workers, chunk_size=chunk_size)

    if PROFILE_MASTER:
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None

    start = time.perf_counter()
    result = engine.run(
        data,
        mapper=mapper,
        reducer=reducer,
        combiner=reducer if use_combiner else None,
    )
    end = time.perf_counter()

    if profiler is not None:
        profiler.disable()
        profiler.dump_stats("master_scheduler.prof")

    return end - start, len(result)


def benchmark_one_size(size_mb: int, *, use_combiner: bool) -> None:
    print("=" * 80)
    print(f"[SIZE] {size_mb}MB")
    data_path = generate_data(size_mb)

    print("[*] Loading data into memory...")
    lines = load_data(data_path)
    print(f"[+] Loaded {len(lines)} lines.")

    if RUN_BASELINE:
        print("[*] Running Sequential Baseline...")
        t_seq, n_seq = run_sequential(lines)
        print(f"[SEQ] Time: {t_seq:.4f}s | Keys: {n_seq}")
    else:
        t_seq, n_seq = None, None
        print("[*] Skipping Sequential Baseline (--baseline not set).")

    for chunk_size in CHUNK_SIZES:
        print(f"[*] Testing chunk_size={chunk_size} ...")
        for w in WORKER_COUNTS:
            label = "WITH combiner" if use_combiner else "WITHOUT combiner"
            tag = "MR+C" if use_combiner else "MR"
            print(
                f"    [*] Running MapReduceEngine {label} (workers={w}, chunk_size={chunk_size})..."
            )
            t_mr, n_mr = run_engine(lines, w, chunk_size, use_combiner=use_combiner)
            if RUN_BASELINE and t_seq is not None and n_seq is not None:
                speedup = t_seq / t_mr if t_mr > 0 else float("inf")
                status = "OK" if n_mr == n_seq else "MISMATCH"
                print(
                    f"[{tag}] workers={w} | chunk_size={chunk_size} | "
                    f"Time: {t_mr:.4f}s | Keys: {n_mr} | "
                    f"Speedup_vs_seq: {speedup:.2f}x | {status}"
                )
            else:
                print(
                    f"[{tag}] workers={w} | chunk_size={chunk_size} | "
                    f"Time: {t_mr:.4f}s | Keys: {n_mr}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mini_map_reduce engine.")
    parser.add_argument(
        "--big_file",
        action="store_true",
        help="Run only a large 5GB file benchmark instead of the standard sizes.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile for the master scheduler and worker processes.",
    )
    parser.add_argument(
        "--combiner",
        action="store_true",
        help="Also run with a combiner to compare IPC and runtime.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run the sequential baseline for comparison.",
    )
    args = parser.parse_args()

    global PROFILE_MASTER, RUN_BASELINE
    if args.profile:
        PROFILE_MASTER = True
        os.environ["MMR_PROFILE_WORKERS"] = "1"

    if args.baseline:
        RUN_BASELINE = True

    if args.big_file:
        sizes: list[int] = [BIG_FILE_SIZE_MB]
    else:
        sizes = list(FILE_SIZES_MB)

    print("[*] Starting MapReduce benchmark across sizes and worker configurations...")
    print(f"    Sizes (MB): {sizes}")
    print(f"    Worker counts: {WORKER_COUNTS}")
    print(f"    Chunk sizes: {CHUNK_SIZES}")

    for size_mb in sizes:
        benchmark_one_size(size_mb, use_combiner=args.combiner)

    print("[+] Benchmark complete.")


if __name__ == "__main__":
    main()
