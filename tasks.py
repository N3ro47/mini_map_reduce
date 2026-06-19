import re
from collections.abc import Iterable
from collections import defaultdict
from typing import Any

# --- 1. WORD COUNT ---
# Logic from Master, Naming from Current (to keep benchmark working)
def wordcount_mapper(text: str) -> Iterable[tuple[str, int]]:
    for word in re.findall(r"\w+", text.lower()):
        yield word, 1

def wordcount_reducer(word: str, counts: Iterable[int]) -> tuple[str, int]:
    return word, sum(counts)

def wordcount_iterative(lines: list[str]) -> dict[str, int]:
    result = defaultdict(int)
    for line in lines:
        for word in re.findall(r"\w+", line.lower()):
            result[word] += 1
    return dict(result)


# --- 2. INVERTED INDEX ---
# Current branch logic (yielding int doc_ids) is usually better for MapReduce engines 
# than yielding sets immediately, so we keep Current logic + Iterative.
def inverted_index_mapper(data: tuple[int, str]) -> Iterable[tuple[str, int]]:
    doc_id, text = data
    words = set(re.findall(r"\w+", text.lower()))
    for word in words:
        yield (word, doc_id)

def inverted_index_reducer(word: str, doc_ids: Iterable[int]) -> tuple[str, list[int]]:
    return (word, sorted(set(doc_ids)))

def inverted_index_iterative(indexed_lines: list[tuple[int, str]]) -> dict[str, list[int]]:
    result = defaultdict(set)
    for doc_id, text in indexed_lines:
        words = set(re.findall(r"\w+", text.lower()))
        for word in words:
            result[word].add(doc_id)
    return {word: sorted(doc_ids) for word, doc_ids in result.items()}


# --- 3. EVENT AGGREGATION (Logs) ---
# Merged: Keeps your Key Format and Iterative logic, but adds the Date Filter from Master.
def log_event_mapper(line: str) -> Iterable[tuple[str, int]]:
    parts = line.strip().split()
    if len(parts) >= 5:
        # master branch logic: date filtering
        date_str = parts[0]
        path = parts[3]
        status = parts[4]
        
        # If master had a specific date requirement, we keep it:
        # (Remove this if-check if you want to process ALL logs)
        if date_str >= "2026-04-15": 
            key = f"{path} [{status}]"
            yield (key, 1)

def log_event_reducer(key: str, counts: Iterable[int]) -> tuple[str, int]:
    return (key, sum(counts))

def log_event_iterative(lines: list[str]) -> dict[str, int]:
    result = defaultdict(int)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            date_str, _, _, path, status = parts[:5]
            if date_str >= "2026-04-15":
                key = f"{path} [{status}]"
                result[key] += 1
    return dict(result)


# --- UTILITIES FROM MASTER ---
# Updated to point to the function names used in your benchmarks
def get_mappers_reducers(task_name: str):
    if task_name == "word_count":
        return wordcount_mapper, wordcount_reducer
    elif task_name == "inverted_index":
        return inverted_index_mapper, inverted_index_reducer
    elif task_name == "logs":
        return log_event_mapper, log_event_reducer
    raise ValueError(f"Unknown task: {task_name}")

def prepare_input(task_name: str, lines: list[str]) -> list[Any]:
    if task_name == "inverted_index":
        return list(enumerate(lines))
    return lines