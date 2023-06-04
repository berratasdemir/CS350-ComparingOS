import time
import math
import numpy as np
import random
import csv


def save_results_to_csv(results, timestamp):
    file_name = f"cpubenchmark_results_{timestamp}.csv"
    file_name = file_name.replace(":", "_")  # Replace colons with underscores
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Benchmark", "Execution Time (seconds)"])
        writer.writerows(results)
    print(f"Benchmark results saved to {file_name}")


def is_prime(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_calculation_benchmark():
    start_time = time.time()
    primes = [x for x in range(1, 100000) if is_prime(x)]
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Prime calculation", execution_time]


def floating_point_arithmetic_benchmark():
    N = 10 ** 7
    A = np.random.rand(N)
    B = np.random.rand(N)
    start_time = time.time()
    C = A * B
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Floating point arithmetic", execution_time]


def matrix_multiplication_benchmark():
    N = 500
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    start_time = time.time()
    C = A @ B
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Matrix multiplication", execution_time]


def fibonacci_sequence_benchmark():
    def fib(n):
        if n <= 1:
            return n
        else:
            return (fib(n - 1) + fib(n - 2))

    start_time = time.time()
    fib(30)
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Fibonacci sequence (recursion)", execution_time]


def sorting_benchmark():
    N = 10 ** 6
    A = list(range(N))
    random.shuffle(A)
    start_time = time.time()
    A.sort()
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Sorting", execution_time]


def searching_benchmark():
    N = 10 ** 6
    A = list(range(N))
    random.shuffle(A)
    target = A[-1]  # worst case
    start_time = time.time()
    index = A.index(target)
    end_time = time.time()
    execution_time = end_time - start_time
    return ["Searching", execution_time]


def benchmark_cpu():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    results = [
        prime_calculation_benchmark(),
        floating_point_arithmetic_benchmark(),
        matrix_multiplication_benchmark(),
        fibonacci_sequence_benchmark(),
        sorting_benchmark(),
        searching_benchmark()
    ]
    save_results_to_csv(results, timestamp)


if __name__ == "__main__":
    benchmark_cpu()
