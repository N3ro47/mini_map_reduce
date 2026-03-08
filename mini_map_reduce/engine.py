from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence

from .scheduler import Scheduler


class MapReduceEngine:
    def __init__(self, *, concurrency_limit: int | None = None, chunk_size: int = 64) -> None:
        cpu = os.cpu_count() or 2
        default_limit = max(1, cpu - 1)
        self._concurrency_limit = (
            default_limit if concurrency_limit is None else max(1, concurrency_limit)
        )
        self._chunk_size = chunk_size

    def run[T, K, V, R](
        self,
        inputs: Sequence[T],
        *,
        mapper: Callable[[T], Iterable[tuple[K, V]]],
        reducer: Callable[[K, Iterable[V]], tuple[K, R] | Iterable[tuple[K, R]]],
        combiner: Callable[
            [K, Iterable[V]],
            tuple[K, R] | Iterable[tuple[K, R]],
        ]
        | None = None,
    ) -> dict[K, R]:
        if not inputs:
            return {}

        scheduler: Scheduler[T, K, V, R] = Scheduler(
            mapper=mapper,
            reducer=reducer,
            combiner=combiner,
            concurrency_limit=self._concurrency_limit,
        )
        return scheduler.execute(inputs, chunk_size=self._chunk_size)
