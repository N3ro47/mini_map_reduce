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
### 1: Word Count (Aggregation Pattern)
Calculating unique token frequencies in datasets. This scenario evaluates the **Combiner** optimization, demonstrating how local pre-aggregation minimizes **IPC overhead** and serialization costs within `multiprocessing.Pipe`, significantly boosting throughput.
### 2: smth
### 3: smth

## 4. Benchmarking 
Performance is evaluated based on:
1. **Data Scaling:** Testing stability and processing speed on datasets of increasing size (up to ...GB).
