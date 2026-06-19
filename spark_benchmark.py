import argparse
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, explode, lower, split


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("MiniMapReduceSparkBenchmark")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def spark_word_count(spark: SparkSession, file_path: str):
    start = time.perf_counter()
    df = spark.read.text(file_path)
    words_df = df.select(explode(split(lower(col("value")), r"\W+")).alias("word")).where(
        col("word") != ""
    )
    result = words_df.groupBy("word").count().count()
    end = time.perf_counter()
    return end - start, result


def spark_inverted_index(spark: SparkSession, file_path: str):
    from pyspark.sql.functions import monotonically_increasing_id

    start = time.perf_counter()
    # Read text, attach monotonically increasing id as proxy for line number
    df = spark.read.text(file_path).withColumn("doc_id", monotonically_increasing_id())
    words_df = df.select(explode(split(lower(col("value")), r"\W+")).alias("word"), "doc_id").where(
        col("word") != ""
    )
    # grouping by word and collecting a set of doc_ids
    result = words_df.groupBy("word").agg(collect_set("doc_id")).count()
    end = time.perf_counter()
    return end - start, result


def spark_logs(spark: SparkSession, file_path: str):
    from pyspark.sql.functions import concat_ws

    # using built-in spark split instead of split_str to be safe, split is imported
    start = time.perf_counter()
    df = spark.read.text(file_path)
    # Filter and group
    # format: date_str user action path status
    parsed_df = (
        df.select(split(col("value"), " ").alias("parts"))
        .select(
            col("parts").getItem(0).alias("date"),
            col("parts").getItem(3).alias("path"),
            col("parts").getItem(4).alias("status"),
        )
        .where(col("date") >= "2026-04-15")
        .select(concat_ws("_", col("path"), col("status")).alias("key"))
    )

    result = parsed_df.groupBy("key").count().count()
    end = time.perf_counter()
    return end - start, result


def main():
    parser = argparse.ArgumentParser(description="Spark subprocess runner")
    parser.add_argument("--task", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    # Build session (not timed)
    spark = build_spark_session()
    # Suppress spark logging for cleaner subprocess output
    spark.sparkContext.setLogLevel("ERROR")

    try:
        if args.task == "word_count":
            t_diff, n_keys = spark_word_count(spark, args.file)
        elif args.task == "inverted_index":
            t_diff, n_keys = spark_inverted_index(spark, args.file)
        elif args.task == "logs":
            t_diff, n_keys = spark_logs(spark, args.file)
        else:
            raise ValueError(f"Unknown task: {args.task}")

        # Write output as JSON
        print(json.dumps({"time_seconds": t_diff, "keys_found": n_keys}))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
