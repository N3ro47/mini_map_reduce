import time
from pathlib import Path
from mini_map_reduce import MapReduceEngine
import tasks
from data_generator import generate_data, TaskType, get_file_path


def verify_results(iterative_result: dict, mapreduce_result: dict, task_name: str) -> bool:
    if set(iterative_result.keys()) != set(mapreduce_result.keys()):
        print(f"  [ERROR] Key mismatch in {task_name}")
        iter_keys = set(iterative_result.keys())
        mr_keys = set(mapreduce_result.keys())
        missing_in_mr = iter_keys - mr_keys
        extra_in_mr = mr_keys - iter_keys
        if missing_in_mr:
            print(f"    Missing in MapReduce: {list(missing_in_mr)[:5]}...")
        if extra_in_mr:
            print(f"    Extra in MapReduce: {list(extra_in_mr)[:5]}...")
        return False

    mismatches = []
    for key in iterative_result:
        if iterative_result[key] != mapreduce_result[key]:
            mismatches.append((key, iterative_result[key], mapreduce_result[key]))

    if mismatches:
        print(f"  [ERROR] Value mismatch in {task_name}")
        for key, iter_val, mr_val in mismatches[:3]:
            print(f"    Key '{key}': iterative={iter_val}, mapreduce={mr_val}")
        if len(mismatches) > 3:
            print(f"    ... and {len(mismatches) - 3} more mismatches")
        return False

    print(f"  [OK] Results verified for {task_name}")
    return True


def print_comparison(task_name: str, iterative_time: float, mapreduce_time: float):
    speedup = iterative_time / mapreduce_time
    improvement = ((iterative_time - mapreduce_time) / iterative_time) * 100

    print(f"\n  Performance Summary for {task_name}:")
    print(f"    Iterative:    {iterative_time:.4f}s")
    print(f"    MapReduce:    {mapreduce_time:.4f}s")
    print(f"    Speedup:      {speedup:.2f}x faster with MapReduce")
    print(f"    Improvement:  {improvement:.1f}% faster")


def run_benchmark():
    engine_8_workers = MapReduceEngine(concurrency_limit=8, chunk_size=1000)

    # --- TEST 1: EVENT AGGREGATION (Logs) ---
    print("\n" + "=" * 60)
    print("TEST 1: LOG EVENT AGGREGATION")
    print("=" * 60)
    generate_data(TaskType.LOG_EVENTS, size_mb=10, imbalance=0, seed=42)
    path = get_file_path(TaskType.LOG_EVENTS, 10, 0)

    with open(path) as f:
        log_lines = f.readlines()

    # Run iterative approach
    start = time.perf_counter()
    iterative_results = tasks.log_event_iterative(log_lines)
    iter_time = time.perf_counter() - start
    print(f"\n[Iterative] Log Aggregation took: {iter_time:.4f}s")

    # Run MapReduce approach
    start = time.perf_counter()
    mapreduce_results = engine_8_workers.run(
        log_lines,
        mapper=tasks.log_event_mapper,
        reducer=tasks.log_event_reducer,
        combiner=tasks.log_event_reducer,
    )
    mr_time = time.perf_counter() - start
    print(f"[MapReduce] Log Aggregation took: {mr_time:.4f}s")

    # Verify results
    verify_results(iterative_results, mapreduce_results, "Log Aggregation")
    print_comparison("Log Aggregation", iter_time, mr_time)

    # --- TEST 2: INVERTED INDEX ---
    print("\n" + "=" * 60)
    print("TEST 2: INVERTED INDEX")
    print("=" * 60)
    generate_data(TaskType.WORD_COUNTING, size_mb=10, imbalance=0, seed=42)
    path = get_file_path(TaskType.WORD_COUNTING, 10, 0)

    with open(path) as f:
        # Inverted index needs document ID (line number)
        indexed_lines = list(enumerate(f.readlines()))

    # Run iterative approach
    start = time.perf_counter()
    iterative_results = tasks.inverted_index_iterative(indexed_lines)
    iter_time = time.perf_counter() - start
    print(f"\n[Iterative] Inverted Index took: {iter_time:.4f}s")

    # Run MapReduce approach
    start = time.perf_counter()
    mapreduce_results = engine_8_workers.run(
        indexed_lines, mapper=tasks.inverted_index_mapper, reducer=tasks.inverted_index_reducer
    )
    mr_time = time.perf_counter() - start
    print(f"[MapReduce] Inverted Index took: {mr_time:.4f}s")

    # Verify results
    verify_results(iterative_results, mapreduce_results, "Inverted Index")
    print_comparison("Inverted Index", iter_time, mr_time)

    # --- TEST 3: WORD COUNT WITH KEY SKEW (95% imbalance) ---
    print("\n" + "=" * 60)
    print("TEST 3: WORD COUNT WITH KEY SKEW (95% imbalance)")
    print("=" * 60)
    generate_data(TaskType.WORD_COUNTING, size_mb=10, imbalance=95, seed=42)
    path = get_file_path(TaskType.WORD_COUNTING, 10, 95)

    with open(path) as f:
        skewed_lines = f.readlines()

    # Run iterative approach
    start = time.perf_counter()
    iterative_results = tasks.wordcount_iterative(skewed_lines)
    iter_time = time.perf_counter() - start
    print(f"\n[Iterative] Word Count took: {iter_time:.4f}s")

    # Run MapReduce approach
    start = time.perf_counter()
    mapreduce_results = engine_8_workers.run(
        skewed_lines,
        mapper=tasks.wordcount_mapper,
        reducer=tasks.wordcount_reducer,
        combiner=tasks.wordcount_reducer,
    )
    mr_time = time.perf_counter() - start
    print(f"[MapReduce] Word Count took: {mr_time:.4f}s")

    # Verify results
    verify_results(iterative_results, mapreduce_results, "Word Count")
    print_comparison("Word Count", iter_time, mr_time)

    # Show top word
    top_word_iter = max(iterative_results.items(), key=lambda x: x[1])
    top_word_mr = max(mapreduce_results.items(), key=lambda x: x[1])
    print(f"Top word (iterative): {top_word_iter}")
    print(f"Top word (MapReduce): {top_word_mr}")


if __name__ == "__main__":
    run_benchmark()
