import sys
import time
import numpy as np
import csv


def save_results_to_csv(results, timestamp):
    file_name = f"memorybenchmark_results_{timestamp}.csv"
    file_name = file_name.replace(":", "_")  # Replace colons with underscores
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Benchmark", "Execution Time (seconds)", "Memory Usage (MB)"])
        writer.writerows(results)
    print(f"Benchmark results saved to {file_name}")

def large_array_benchmark():
    start_time = time.time()
    data = np.arange(1000000)
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = sys.getsizeof(data) / (1024 * 1024)
    return ["Large array creation", execution_time, memory_usage]


def list_comprehension_benchmark():
    start_time = time.time()
    data = [str(i) for i in range(1000000)]
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = sys.getsizeof(data) / (1024 * 1024)
    return ["List comprehension", execution_time, memory_usage]


def dictionary_creation_benchmark():
    start_time = time.time()
    data = {str(i): i for i in range(1000000)}
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = sys.getsizeof(data) / (1024 * 1024)
    return ["Dictionary creation", execution_time, memory_usage]


def string_concatenation_benchmark():
    start_time = time.time()
    data = ''.join([str(i) for i in range(1000000)])
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = sys.getsizeof(data) / (1024 * 1024)
    return ["String concatenation", execution_time, memory_usage]


def benchmark_memory():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    results = [
        large_array_benchmark(),
        list_comprehension_benchmark(),
        dictionary_creation_benchmark(),
        string_concatenation_benchmark()
    ]
    save_results_to_csv(results, timestamp)


if __name__ == "__main__":
    benchmark_memory()
