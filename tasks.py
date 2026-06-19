import re
from collections.abc import Iterable
from typing import Any


# --- WORD COUNT ---
def word_count_mapper(text: str) -> Iterable[tuple[str, int]]:
    for word in re.findall(r"\w+", text.lower()):
        yield word, 1


def word_count_reducer(word: str, counts: Iterable[int]) -> tuple[str, int]:
    return word, sum(counts)


# --- INVERTED INDEX ---
def inverted_index_mapper(item: tuple[int, str]) -> Iterable[tuple[str, set[int]]]:
    line_number, text = item
    for word in re.findall(r"\w+", text.lower()):
        yield word, {line_number}


def inverted_index_reducer(word: str, sets: Iterable[set[int]]) -> tuple[str, set[int]]:
    result = set()
    for s in sets:
        result.update(s)
    return word, result


# --- LOG EVENTS ---
def log_events_mapper(line: str) -> Iterable[tuple[str, int]]:
    line = line.strip()
    if not line:
        return []
    parts = line.split(" ")
    if len(parts) >= 5:
        date_str, user, action, path, status = parts[:5]
        if date_str >= "2026-04-15":
            yield f"{path}_{status}", 1


def log_events_reducer(key: str, counts: Iterable[int]) -> tuple[str, int]:
    return key, sum(counts)


def get_mappers_reducers(task_name: str):
    if task_name == "word_count":
        return word_count_mapper, word_count_reducer
    elif task_name == "inverted_index":
        return inverted_index_mapper, inverted_index_reducer
    elif task_name == "logs":
        return log_events_mapper, log_events_reducer
    raise ValueError(f"Unknown task: {task_name}")


def prepare_input(task_name: str, lines: list[str]) -> list[Any]:
    if task_name == "inverted_index":
        return list(enumerate(lines))
    return lines
