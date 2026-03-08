from __future__ import annotations

import cProfile
import os
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from multiprocessing.connection import Connection
from typing import Any, cast

from .protocol import TaskKind


def worker_node[T, K, V](
    task_rx: Connection,
    result_tx: Connection,
    mapper: Callable[[T], Iterable[tuple[K, V]]],
    reducer: Callable[[K, Iterable[V]], tuple[K, Any] | Iterable[tuple[K, Any]]],
    combiner: Callable[[K, Iterable[V]], tuple[K, Any] | Iterable[tuple[K, Any]]] | None,
) -> None:
    while True:
        msg = task_rx.recv()
        kind = cast(TaskKind, msg[0])

        if kind == "stop":
            return

        task_id = cast(int, msg[1])
        try:
            if kind == "map":
                chunk = cast(Sequence[T], msg[2])

                if combiner is None:
                    out: list[tuple[K, V]] = []
                    for item in chunk:
                        out.extend(mapper(item))
                    result_tx.send(("map_ok", task_id, out))
                else:
                    buffer: defaultdict[K, list[V]] = defaultdict(list)
                    for item in chunk:
                        for k, v in mapper(item):
                            buffer[k].append(v)

                    pairs: list[tuple[Any, Any]] = []
                    for k, vs in buffer.items():
                        combined = combiner(k, vs)

                        if isinstance(combined, tuple) and len(combined) == 2:
                            pairs.append(cast(tuple[Any, Any], combined))
                        else:
                            pairs.extend(
                                cast(tuple[Any, Any], p) for p in cast(Iterable[Any], combined)
                            )

                    result_tx.send(("map_ok", task_id, pairs))
            elif kind == "reduce":
                key = cast(K, msg[2])
                values = cast(Sequence[V], msg[3])
                reduced = reducer(key, values)

                if isinstance(reduced, tuple) and len(reduced) == 2:
                    pairs: list[tuple[Any, Any]] = [cast(tuple[Any, Any], reduced)]
                else:
                    pairs = [cast(tuple[Any, Any], p) for p in cast(Iterable[Any], reduced)]

                result_tx.send(("reduce_ok", task_id, pairs))
            else:
                raise RuntimeError(f"unknown task kind: {kind!r}")
        except BaseException:
            result_tx.send(("err", task_id, kind, traceback.format_exc()))


def worker_node_profiled[T, K, V](
    task_rx: Connection,
    result_tx: Connection,
    mapper: Callable[[T], Iterable[tuple[K, V]]],
    reducer: Callable[[K, Iterable[V]], tuple[K, Any] | Iterable[tuple[K, Any]]],
    combiner: Callable[[K, Iterable[V]], tuple[K, Any] | Iterable[tuple[K, Any]]] | None,
) -> None:
    """Wraps worker_node in cProfile and writes a per-process .prof file."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        worker_node(task_rx, result_tx, mapper, reducer, combiner)
    finally:
        profiler.disable()
        profiler.dump_stats(f"worker_{os.getpid()}.prof")
