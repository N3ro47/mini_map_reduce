import argparse
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, split

from benchmark import (
    BIG_FILE_SIZE_MB,
    DATA_DIR,
    FILE_SIZES_MB,
    VOCAB_SIZE,
    WORDS_PER_LINE,
    generate_data,
)


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("MiniMapReduceSparkBenchmark")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def spark_wordcount_file(spark: SparkSession, path: Path) -> tuple[float, int]:
    start = time.perf_counter()

    df = spark.read.text(str(path))

    # Extract tokens with approximately the same logic as the Python baseline
    # (using non-word characters as delimiters, then filtering empty tokens).
    words_df = df.select(explode(split(lower(col("value")), r"\W+")).alias("word")).where(
        col("word") != ""
    )

    counts_df = words_df.groupBy("word").count()
    result = counts_df.count()

    end = time.perf_counter()
    return end - start, result


def benchmark_one_size(spark: SparkSession, size_mb: int) -> None:
    print("=" * 80)
    print(f"[SPARK SIZE] {size_mb}MB")

    data_path = generate_data(size_mb)
    print(f"[*] Using data file: {data_path}")

    t_spark, n_keys = spark_wordcount_file(spark, data_path)
    print(f"[SPARK] Time: {t_spark:.4f}s | Keys: {n_keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark benchmark for mini_map_reduce.")
    parser.add_argument(
        "--big_file",
        action="store_true",
        help="Run only a large 5GB file benchmark instead of the standard sizes.",
    )
    args = parser.parse_args()

    if args.big_file:
        sizes: list[int] = [BIG_FILE_SIZE_MB]
    else:
        sizes = list(FILE_SIZES_MB)

    print("[*] Starting Spark benchmark across sizes...")
    print(f"    Sizes (MB): {sizes}")

    spark = build_spark_session()
    try:
        for size_mb in sizes:
            benchmark_one_size(spark, size_mb)
    finally:
        spark.stop()

    print("[+] Spark benchmark complete.")


if __name__ == "__main__":
    main()
