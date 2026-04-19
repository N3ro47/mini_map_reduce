import argparse
import random
import pathlib
from enum import StrEnum
from datetime import datetime, timedelta
from typing import Any


"""
Deterministic Data Generator for MapReduce Benchmarks

This module generates large-scale text datasets specifically designed to test 
MapReduce implementations. It supports three types of tasks: Word Counting, 
Inverted Indexing, and Log Event analysis.

CORE FEATURES:
1. Determinism: Uses a random seed to ensure that the same file is generated 
   every time given the same input parameters.
2. Imbalance (Hot Keys): Allows simulating 'stragglers' in a distributed system 
   by making a single key (e.g., 'word0' or the '/home' path) appear with a 
   user-defined percentage frequency (0-100%).

USAGE EXAMPLES:
    # Generate 100MB of word data with a 90% imbalance (hot key 'word0')
    python data_generator.py word-counting --size 100 --imbalance 90

    # Generate 50MB of log data with perfectly uniform distribution
    python data_generator.py log-events --size 50

    # Generate data using a custom seed for a different random sequence
    python data_generator.py word-counting --size 10 --seed 12345
"""


# Constants
VOCAB_SIZE = 5_000
WORDS_PER_LINE = 50
USERS_COUNT = 1_000
ACTIONS = ["click", "view", "login", "logout", "post"]
STATUSES = ["200", "404", "500", "302"]
PATHS_COUNT = 100
DATA_DIR = pathlib.Path("benchmark_data")
DATA_DIR.mkdir(exist_ok=True)
SIZE_MISMATCH_TOLERANCE = 0.01


class TaskType(StrEnum):
    WORD_COUNTING = "word-counting"
    INVERTED_INDEX = "inverted-index"
    LOG_EVENTS = "log-events"


def get_file_path(task: TaskType, size_mb: int, imbalance: int) -> pathlib.Path:
    return DATA_DIR / f"{task}_{size_mb}mb_imb{imbalance}.txt"


# --- WORD TASK LOGIC ---


def prepare_word_weights(imbalance: int) -> tuple[list[str], list[float]]:
    words = [f"word{i}" for i in range(VOCAB_SIZE)]
    if imbalance == 0:
        # Uniform distribution: all words have equal weight
        weights = [1.0] * VOCAB_SIZE
    else:
        # word0 gets the imbalance percentage, others share the rest
        remaining_weight = (100 - imbalance) / (VOCAB_SIZE - 1)
        weights = [float(imbalance)] + [remaining_weight] * (VOCAB_SIZE - 1)
    return words, weights


def generate_words_file(file_path: pathlib.Path, target_bytes: int, imbalance: int) -> None:
    words, weights = prepare_word_weights(imbalance)

    bytes_written = 0
    with file_path.open("w") as f:
        while bytes_written < target_bytes:
            line = " ".join(random.choices(words, weights=weights, k=WORDS_PER_LINE)) + "\n"
            f.write(line)
            bytes_written += len(line)


# --- LOG TASK LOGIC ---


def prepare_log_params(imbalance: int) -> dict[str, Any]:
    params = {
        "users": [f"user{i}" for i in range(USERS_COUNT)],
        "actions": ACTIONS,
        "statuses": STATUSES,
        "paths": ["/home"] + [f"/path/{i}" for i in range(1, PATHS_COUNT)],
        "base_date": datetime(2026, 4, 1),
    }

    if imbalance == 0:
        params["path_weights"] = [1.0] * PATHS_COUNT
    else:
        remaining_path_weight = (100 - imbalance) / (PATHS_COUNT - 1)
        params["path_weights"] = [float(imbalance)] + [remaining_path_weight] * (PATHS_COUNT - 1)
    return params


def generate_logs_file(file_path: pathlib.Path, target_bytes: int, imbalance: int) -> None:
    p = prepare_log_params(imbalance)

    bytes_written = 0
    with file_path.open("w") as f:
        while bytes_written < target_bytes:
            date_str = (p["base_date"] + timedelta(days=random.randint(0, 29))).strftime("%Y-%m-%d")
            user = random.choice(p["users"])
            action = random.choice(p["actions"])
            path = random.choices(p["paths"], weights=p["path_weights"], k=1)[0]
            status = random.choice(p["statuses"])

            line = f"{date_str} {user} {action} {path} {status}\n"
            f.write(line)
            bytes_written += len(line)


# --- MAIN ENTRY POINT ---


def generate_data(task: TaskType, size_mb: int, imbalance: int, seed: int) -> None:
    # Set seed at the very start to ensure deterministic behavior across all function
    random.seed(seed)

    file_path = get_file_path(task, size_mb, imbalance)
    target_bytes = size_mb * 1024 * 1024

    # Reuse existing file if it matches the criteria (size within tolerance)
    if file_path.exists():
        existing_size = file_path.stat().st_size
        if abs(existing_size - target_bytes) <= target_bytes * SIZE_MISMATCH_TOLERANCE:
            print(f"[+] Reusing existing data file: {file_path}")
            return

    print(f"[*] Generating {size_mb}MB for '{task}' (Imbalance: {imbalance}%, Seed: {seed})...")

    if task in [TaskType.WORD_COUNTING, TaskType.INVERTED_INDEX]:
        generate_words_file(file_path, target_bytes, imbalance)
    elif task == TaskType.LOG_EVENTS:
        generate_logs_file(file_path, target_bytes, imbalance)

    actual_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"[+] Complete: {file_path} ({actual_size_mb:.2f}MB written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic MapReduce Data Generator")
    parser.add_argument("task", choices=[t.value for t in TaskType], help="Generation task")
    parser.add_argument("--size", type=int, default=10, help="Size in MB")
    parser.add_argument("--imbalance", type=int, default=0, help="Percentage (0-100) for hot key")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")

    args = parser.parse_args()
    generate_data(TaskType(args.task), args.size, args.imbalance, args.seed)
