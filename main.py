from __future__ import annotations

from collections.abc import Iterable

from mini_map_reduce import MapReduceEngine


def wordcount_mapper(line: str) -> Iterable[tuple[str, int]]:
    for raw in line.split():
        w = raw.strip().lower()
        if w:
            yield (w, 1)


def wordcount_reducer(word: str, counts: Iterable[int]) -> tuple[str, int]:
    return (word, sum(counts))


def main() -> None:
    engine = MapReduceEngine()
    data = [
        "To be, or not to be: that is the question",
        "Whether 'tis nobler in the mind to suffer",
        "The slings and arrows of outrageous fortune",
        "Or to take arms against a sea of troubles",
        "And by opposing end them",
    ]
    counts = engine.run(
        data,
        mapper=wordcount_mapper,
        reducer=wordcount_reducer,
        combiner=wordcount_reducer,
    )

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    print("Top words:")
    for w, c in top:
        print(f"{w:>12}  {c}")


if __name__ == "__main__":
    main()
