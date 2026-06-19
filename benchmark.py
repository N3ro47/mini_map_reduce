import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from data_generator import TaskType, generate_data, get_file_path
from mini_map_reduce import MapReduceEngine
from tasks import get_mappers_reducers, prepare_input


def load_data(path: Path) -> list[str]:
    with path.open("r") as f:
        return f.readlines()


def run_python_engine(
    task_name: str, data: list[str], workers: int, chunk_size: int, use_combiner: bool
):
    mapper, reducer = get_mappers_reducers(task_name)
    combiner = reducer if use_combiner else None

    # Inverted index needs document ID (lines)
    prepared_inputs = prepare_input(task_name, data)

    engine = MapReduceEngine(concurrency_limit=workers, chunk_size=chunk_size)

    start = time.perf_counter()
    result = engine.run(
        prepared_inputs,
        mapper=mapper,
        reducer=reducer,
        combiner=combiner,
    )
    end = time.perf_counter()
    return end - start, len(result)


def run_spark_subprocess(task_name: str, file_path: Path) -> tuple[float, int]:
    import sys

    cmd = [sys.executable, "spark_benchmark.py", "--task", task_name, "--file", str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # The output is expected to be a single JSON line at the end
    out_lines = result.stdout.strip().split("\n")
    for line in reversed(out_lines):
        if line.startswith("{"):
            data = json.loads(line)
            return data["time_seconds"], data["keys_found"]
    raise RuntimeError(f"Could not parse Spark output: {result.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Master MapReduce Benchmark CLI")
    parser.add_argument(
        "--tests",
        type=str,
        default="word_count",
        help="Comma-separated list of tests: word_count,inverted_index,logs,all",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10], help="List of file sizes in MB"
    )
    parser.add_argument(
        "--imbalance",
        type=int,
        nargs="+",
        default=[0],
        help="Imbalance percentage (0-100), e.g. 0 50 90",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Number of repetitions for each configuration"
    )
    parser.add_argument("--run-spark", action="store_true", help="Run Spark versions as well")
    parser.add_argument("--output", type=str, default="results.csv", help="CSV output file")

    # We allow user to specify a subset of configurations to keep trials short if needed
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 2, 4, 8], help="Worker counts"
    )
    parser.add_argument("--chunks", type=int, nargs="+", default=[5000], help="Chunk sizes")

    args = parser.parse_args()

    test_choices = ["word_count", "inverted_index", "logs"]
    if args.tests == "all":
        selected_tests = test_choices
    else:
        selected_tests = [t.strip() for t in args.tests.split(",")]
        for t in selected_tests:
            if t not in test_choices:
                raise ValueError(f"Unknown test: {t}")

    task_enum_map = {
        "word_count": TaskType.WORD_COUNTING,
        "inverted_index": TaskType.INVERTED_INDEX,
        "logs": TaskType.LOG_EVENTS,
    }

    # Pre-generate data
    print("=" * 80)
    print("[*] PHASE 1: Pre-generating data...")
    data_files = {}
    seed = 42
    for t_name in selected_tests:
        t_enum = task_enum_map[t_name]
        data_files[t_name] = {}
        for size in args.sizes:
            data_files[t_name][size] = {}
            for imb in args.imbalance:
                generate_data(t_enum, size, imb, seed)
                data_files[t_name][size][imb] = get_file_path(t_enum, size, imb)
    print("[+] Data generation complete.")

    # Initialize CSV
    out_path = Path(args.output)
    write_header = not out_path.exists()

    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "Timestamp",
                    "Engine",
                    "Task",
                    "Size_MB",
                    "Imbalance",
                    "Worker_Count",
                    "Chunk_Size",
                    "Use_Combiner",
                    "Run_Index",
                    "Time_Seconds",
                    "Keys_Found",
                ]
            )
            f.flush()

        print("=" * 80)
        print("[*] PHASE 2: Running Benchmarks...")

        for t_name in selected_tests:
            for size in args.sizes:
                for imb in args.imbalance:
                    file_path = data_files[t_name][size][imb]

                    print(f"\n[*] Target: Task={t_name} | Size={size}MB | Imbalance={imb}%")
                    print("[*] Loading Python input into memory avoiding timing pollution...")
                    lines = load_data(file_path)

                    for chunk_size in args.chunks:
                        for w in args.workers:
                            for use_combiner in [False, True]:
                                for rep in range(args.repeats):
                                    print(
                                        f"    [PY] {t_name} | size={size}MB | imb={imb}% | w={w} | ch={chunk_size} | comb={use_combiner} | rep={rep + 1}/{args.repeats}"
                                    )
                                    t_mr, n_mr = run_python_engine(
                                        t_name, lines, w, chunk_size, use_combiner
                                    )
                                    print(f"        -> Time: {t_mr:.4f}s | Keys: {n_mr}")

                                    writer.writerow(
                                        [
                                            datetime.now().isoformat(),
                                            "Python",
                                            t_name,
                                            size,
                                            imb,
                                            w,
                                            chunk_size,
                                            use_combiner,
                                            rep + 1,
                                            t_mr,
                                            n_mr,
                                        ]
                                    )
                                    f.flush()

                    # allow GC to free lines if needed, since Spark runs its own file read
                    del lines

                    if args.run_spark:
                        for rep in range(args.repeats):
                            print(
                                f"    [SPARK] {t_name} | size={size}MB | imb={imb}% | rep={rep + 1}/{args.repeats}"
                            )
                            t_sp, n_sp = run_spark_subprocess(t_name, file_path)
                            print(f"        -> Time: {t_sp:.4f}s | Keys: {n_sp}")
                            writer.writerow(
                                [
                                    datetime.now().isoformat(),
                                    "Spark",
                                    t_name,
                                    size,
                                    imb,
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                    rep + 1,
                                    t_sp,
                                    n_sp,
                                ]
                            )
                            f.flush()

    print("=" * 80)
    print(f"[+] All benchmarks complete! Results saved to '{args.output}'.")


if __name__ == "__main__":
    main()
