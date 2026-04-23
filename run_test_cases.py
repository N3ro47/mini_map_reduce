import time
from pathlib import Path
from mini_map_reduce import MapReduceEngine
import tasks  
from data_generator import generate_data, TaskType, get_file_path

def run_benchmark():
    engine_8_workers = MapReduceEngine(concurrency_limit=8, chunk_size=1000)
    
    # --- TEST 1: EVENT AGGREGATION (Logs) ---
    print("\n--- Testing Log Aggregation ---")
    generate_data(TaskType.LOG_EVENTS, size_mb=10, imbalance=0, seed=42)
    path = get_file_path(TaskType.LOG_EVENTS, 10, 0)
    
    with open(path) as f:
        log_lines = f.readlines()
    
    start = time.perf_counter()
    results = engine_8_workers.run(
        log_lines,
        mapper=tasks.log_event_mapper,
        reducer=tasks.log_event_reducer,
        combiner=tasks.log_event_reducer 
    )
    print(f"Log Aggregation took: {time.perf_counter() - start:.4f}s")
    print(f"Sample results: {list(results.items())[:3]}")


    # --- TEST 2: INVERTED INDEX ---
    print("\n--- Testing Inverted Index ---")
    generate_data(TaskType.WORD_COUNTING, size_mb=10, imbalance=0, seed=42)
    path = get_file_path(TaskType.WORD_COUNTING, 10, 0)
    
    with open(path) as f:
        # Inverted index needs document ID (line number)
        indexed_lines = list(enumerate(f.readlines()))
    
    start = time.perf_counter()
    results = engine_8_workers.run(
        indexed_lines,
        mapper=tasks.inverted_index_mapper,
        reducer=tasks.inverted_index_reducer
    )
    print(f"Inverted Index took: {time.perf_counter() - start:.4f}s")
    print(f"Keys found: {len(results)}")


    # --- TEST 3: KEY SKEW (Word Count with imbalance) ---
    print("\n--- Testing Key Skew (95% imbalance) ---")
    generate_data(TaskType.WORD_COUNTING, size_mb=10, imbalance=95, seed=42)
    path = get_file_path(TaskType.WORD_COUNTING, 10, 95)
    
    with open(path) as f:
        skewed_lines = f.readlines()
        
    start = time.perf_counter()
    results = engine_8_workers.run(
        skewed_lines,
        mapper=tasks.wordcount_mapper,
        reducer=tasks.wordcount_reducer,
        combiner=tasks.wordcount_reducer
    )
    print(f"Key Skew Task took: {time.perf_counter() - start:.4f}s")
    top_word = max(results.items(), key=lambda x: x[1])
    print(f"Top word: {top_word}")

if __name__ == "__main__":
    run_benchmark()