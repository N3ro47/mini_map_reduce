from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import suppress
from multiprocessing import get_context
from multiprocessing.connection import Connection, wait
from typing import Any, Literal, cast

from .protocol import TaskKind, iter_chunks
from .worker import worker_node, worker_node_profiled


class Scheduler[T, K, V, R]:
    def __init__(
        self,
        *,
        mapper: Callable[[T], Iterable[tuple[K, V]]],
        reducer: Callable[[K, Iterable[V]], tuple[K, R] | Iterable[tuple[K, R]]],
        combiner: Callable[
            [K, Iterable[V]],
            tuple[K, R] | Iterable[tuple[K, R]],
        ]
        | None,
        concurrency_limit: int,
    ) -> None:
        self._mapper = mapper
        self._reducer = reducer
        self._combiner = combiner
        self._concurrency_limit = max(1, concurrency_limit)

        self._ctx = get_context("spawn")
        self._workers: list[Any] = []
        self._task_txs: list[Connection] = []
        self._result_rxs: list[Connection] = []
        self._result_to_worker: dict[int, int] = {}

    def execute(self, inputs: Sequence[T], *, chunk_size: int) -> dict[K, R]:
        n_tasks = max(1, (len(inputs) + chunk_size - 1) // chunk_size)
        n_workers = min(self._concurrency_limit, n_tasks)
        self._start_workers(n_workers)

        try:
            grouped = self._map_then_shuffle(inputs, chunk_size=chunk_size)
            return self._reduce(grouped)
        finally:
            self._shutdown()

    def _start_workers(self, n: int) -> None:
        self._workers = []
        self._task_txs = []
        self._result_rxs = []
        self._result_to_worker = {}

        profile_workers = os.getenv("MMR_PROFILE_WORKERS") == "1"
        target = worker_node_profiled if profile_workers else worker_node

        for i in range(n):
            task_rx, task_tx = self._ctx.Pipe(duplex=False)
            result_rx, result_tx = self._ctx.Pipe(duplex=False)
            p = self._ctx.Process(
                target=target,
                args=(task_rx, result_tx, self._mapper, self._reducer, self._combiner),
            )
            p.start()

            task_rx.close()
            result_tx.close()

            self._workers.append(p)
            self._task_txs.append(task_tx)
            self._result_rxs.append(result_rx)
            self._result_to_worker[result_rx.fileno()] = i

    def _map_then_shuffle(self, inputs: Sequence[T], *, chunk_size: int) -> dict[K, list[V]]:
        tasks = iter(enumerate(iter_chunks(inputs, chunk_size)))
        grouped: dict[K, list[V]] = defaultdict(list)
        self._run_phase(
            kind="map",
            tasks=tasks,
            on_ok=lambda pairs: self._shuffle_into(grouped, pairs),
        )
        return grouped

    def _reduce(self, grouped: dict[K, list[V]]) -> dict[K, R]:
        out: dict[K, R] = {}
        tasks = ((i, (k, list(vs))) for i, (k, vs) in enumerate(grouped.items()))
        self._run_phase(
            kind="reduce",
            tasks=tasks,
            on_ok=lambda pairs: self._reduce_into(out, pairs),
        )
        return out

    def _run_phase(
        self,
        *,
        kind: Literal["map", "reduce"],
        tasks: Iterator[tuple[int, Any]],
        on_ok: Callable[[Any], None],
    ) -> None:
        idle = list(range(len(self._workers)))
        in_flight = 0

        def try_send(worker_idx: int) -> bool:
            nonlocal in_flight
            try:
                task_id, payload = next(tasks)
            except StopIteration:
                return False

            if kind == "map":
                self._task_txs[worker_idx].send(("map", task_id, payload))
            else:
                k, vs = cast(tuple[Any, Any], payload)
                self._task_txs[worker_idx].send(("reduce", task_id, k, vs))

            in_flight += 1
            return True

        newly_idle: list[int] = []
        while idle and try_send(idle[-1]):
            idle.pop()

        while in_flight:
            ready = wait(self._result_rxs)
            for conn in ready:
                rkind, task_id, *rest = conn.recv()
                in_flight -= 1
                newly_idle.append(self._result_to_worker[conn.fileno()])

                if rkind == "err":
                    phase = cast(TaskKind, rest[0])
                    tb = cast(str, rest[1])
                    raise RuntimeError(f"worker failed in {phase} task_id={task_id}\n{tb}")

                on_ok(rest[0])

            idle.extend(newly_idle)
            newly_idle.clear()
            while idle and try_send(idle[-1]):
                idle.pop()

    @staticmethod
    def _shuffle_into(grouped: dict[K, list[V]], pairs: Any) -> None:
        for k, v in cast(list[tuple[K, V]], pairs):
            grouped[k].append(v)

    @staticmethod
    def _reduce_into(out: dict[K, R], pairs: Any) -> None:
        for k, r in cast(list[tuple[K, R]], pairs):
            out[k] = r

    def _shutdown(self) -> None:
        for tx in self._task_txs:
            with suppress(OSError):
                tx.send(("stop", 0))
        for p in self._workers:
            p.join(timeout=2.0)
        for p in self._workers:
            if p.is_alive():
                p.terminate()
        for p in self._workers:
            p.join(timeout=2.0)
        for tx in self._task_txs:
            with suppress(OSError):
                tx.close()
        for rx in self._result_rxs:
            with suppress(OSError):
                rx.close()
