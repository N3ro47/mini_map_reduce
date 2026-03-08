from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, TypeVar

TaskKind = Literal["map", "reduce", "stop"]
ResultKind = Literal["map_ok", "reduce_ok", "err"]

T = TypeVar("T")


def iter_chunks[T](items: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
