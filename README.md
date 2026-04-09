# Mini MapReduce - performance analysis in Python


## 1. Project Objective & Scope
This project investigates the efficiency of the **MapReduce** paradigm within a multi-process Python environment. Inspired by the Google research paper **"MapReduce: Simplified Data Processing on Large Clusters"** (Dean & Ghemawat), we evaluate how distributing workloads across multiple CPU cores via the `multiprocessing` module bypasses Python's **Global Interpreter Lock (GIL)** to achieve parallelism and accelerate Big Data processing.
## 2. System Architecture
* **The Scheduler (Master):** Orchestrates the data flow, handles chunking, and manages the worker pool via `multiprocessing`.
* **Worker Nodes:** Independent processes that execute user-defined `mapper` and `reducer` functions.
* **Shuffle Mechanism:** A central logic that groups intermediate `<Key, Value>` pairs before the reduction phase.
* **Combiner Optimization:** A local aggregation step used to minimize Inter-Process Communication (IPC) overhead.
## 3. Research Scenarios 
To test the engine under different conditions, three different loads were defined:
### 1: Inverted Index (Data Complexity)
* **Task:** Mapping repeated terms to a list of document IDs (`word -> [doc_ids]`).
* **Focus:** Evaluating memory management and shuffle efficiency.

### 2: Key-Skew Benchmark (Load Imbalance)
* **Task:** Processing data where one key (word) accounts for 80–95% of all records.
* **Focus:** Measuring the impact of workload imbalance on total execution time.

### 3: Event Aggregation 
* **Task:** Parsing structured logs.

* **Focus:** Assessing Mapper throughput during string parsing and the effectiveness of data reduction before the Shuffle phase.

## 4. Benchmarking 
Performance is evaluated based on:

* **Scalability:** Comparison of Single-core vs. Multi-worker performance.
* **Chunk Size:** How data split sizes affect the balance between scheduling overhead and CPU usage.
* **Combiner Impact:** Performance gain from local pre-aggregation and raw data transmission (IPC).
* **Data Distribution:** Comparing system stability on uniform data and imbalanced datasets.
* **Spark Baseline:** Benchmarking our optimal setup against a Apache Spark implementation.
