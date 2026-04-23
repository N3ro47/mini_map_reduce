import re
from collections.abc import Iterable
from collections import defaultdict


# --- 1. INVERTED INDEX ---
# Input: (doc_id, text_line)
def inverted_index_mapper(data: tuple[int, str]) -> Iterable[tuple[str, int]]:
    doc_id, text = data
    # We find words and make a set so as not to emit the same word 100 times from one line
    words = set(re.findall(r"\w+", text.lower()))
    for word in words:
        yield (word, doc_id)


def inverted_index_reducer(word: str, doc_ids: Iterable[int]) -> tuple[str, list[int]]:
    # Reducer collects all doc_ids for a given word and returns a sorted list of unique doc_ids
    return (word, sorted(set(doc_ids)))


def inverted_index_iterative(indexed_lines: list[tuple[int, str]]) -> dict[str, list[int]]:
    result = defaultdict(set)

    for doc_id, text in indexed_lines:
        words = set(re.findall(r"\w+", text.lower()))
        for word in words:
            result[word].add(doc_id)

    return {word: sorted(doc_ids) for word, doc_ids in result.items()}


# --- 2. WORD COUNT
# Input: text_line
def wordcount_mapper(line: str) -> Iterable[tuple[str, int]]:
    words = re.findall(r"\w+", line.lower())
    for word in words:
        yield (word, 1)


def wordcount_reducer(word: str, counts: Iterable[int]) -> tuple[str, int]:
    return (word, sum(counts))


def wordcount_iterative(lines: list[str]) -> dict[str, int]:
    result = defaultdict(int)

    for line in lines:
        words = re.findall(r"\w+", line.lower())
        for word in words:
            result[word] += 1

    return dict(result)


# --- 3. EVENT AGGREGATION (Logs) ---
def log_event_mapper(line: str) -> Iterable[tuple[str, int]]:
    parts = line.strip().split()
    if len(parts) >= 5:
        path = parts[3]
        status = parts[4]
        key = f"{path} [{status}]"
        yield (key, 1)


def log_event_reducer(key: str, counts: Iterable[int]) -> tuple[str, int]:
    return (key, sum(counts))


def log_event_iterative(lines: list[str]) -> dict[str, int]:
    result = defaultdict(int)

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            path = parts[3]
            status = parts[4]
            key = f"{path} [{status}]"
            result[key] += 1

    return dict(result)
