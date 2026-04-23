import argparse
import time
import sys
from pathlib import Path

from mini_map_reduce import MapReduceEngine
import tasks  
from data_generator import generate_data, TaskType, get_file_path



def run_experiment(args):
    # 1. Engine initialization with user-defined parameters
    engine = MapReduceEngine(
        concurrency_limit=args.workers, 
        chunk_size=args.chunk_size
    )
    
    # 2. Task map 
    task_map = {
        "wordcount": (TaskType.WORD_COUNTING, tasks.wordcount_mapper, tasks.wordcount_reducer),
        "index": (TaskType.WORD_COUNTING, tasks.inverted_index_mapper, tasks.inverted_index_reducer),
        "logs": (TaskType.LOG_EVENTS, tasks.log_event_mapper, tasks.log_event_reducer),
    }
    
    if args.task not in task_map:
        print(f"Error: Unknown task '{args.task}'. Choose: wordcount, index or logs.")
        return

    gen_type, mapper_fn, reducer_fn = task_map[args.task]
    
    # The decision to use a combiner
    combiner_fn = reducer_fn if args.use_combiner else None

    # 3. Data preparation (generation and loading)
    print(f"[*] Preparing data: Task={args.task}, Size={args.size}MB, Imbalance={args.imb}%")
    generate_data(gen_type, size_mb=args.size, imbalance=args.imb, seed=42)
    path = get_file_path(gen_type, args.size, args.imb)
    
    with open(path, encoding="utf-8") as f:
        if args.task == "index":
            # Inverted index requires line numbering (doc_id)
            input_data = list(enumerate(f.readlines()))
        else:
            input_data = f.readlines()

    # 4. Starting the engine and measuring the time
    print(f"[*] Starting Engine: {args.workers} workers, Combiner={args.use_combiner}")
    start_time = time.perf_counter()
    
    results = engine.run(
        input_data,
        mapper=mapper_fn,
        reducer=reducer_fn,
        combiner=combiner_fn
    )
    
    duration = time.perf_counter() - start_time

    # 5. prints
    print("\n" + "="*40)
    print("         BENCHMARK SUMMARY")
    print("="*40)
    print(f" TASK:           {args.task.upper()}")
    print(f" EXECUTION TIME: {duration:.4f} s")
    print(f" UNIQUE KEYS:    {len(results)}")

    if results and args.task != "index":
            # Find the most frequent key (top result)
            top_key = max(results.items(), key=lambda x: x[1])
            print(f" TOP RESULT:     {top_key}")
            
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MapReduce")
    
    # Control parameters
    parser.add_argument("--task", choices=["wordcount", "index", "logs"], required=True, help="The type of task to perform")
    parser.add_argument("--size", type=int, default=10, help="Data size in MB")
    parser.add_argument("--imb", type=int, default=0, help="Data imbalance percentage (0-95)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of lines per data chunk")
    parser.add_argument("--use_combiner", action="store_true", help="Enable Combiner optimization")

    args = parser.parse_args()
    run_experiment(args)


    # to run the test cases in terminal, use:
    # python test_cases_benchmark.py --task index --size 10 --workers 4 
    # (in different configurations)
    # only tasks is required, the rest have defaults
    