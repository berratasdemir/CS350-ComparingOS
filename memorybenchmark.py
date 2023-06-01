# Memory Benchmarks
import sys
import time
import numpy as np


# 1. Large array creation
def large_array_benchmark():
    start_time = time.time()
    data = np.arange(1000000)
    end_time = time.time()
    print(f"Large array creation: {end_time - start_time} seconds, {sys.getsizeof(data) / (1024 * 1024)} MB")


# 2. List comprehension
def list_comprehension_benchmark():
    start_time = time.time()
    data = [str(i) for i in range(1000000)]
    end_time = time.time()
    print(f"List comprehension: {end_time - start_time} seconds, {sys.getsizeof(data) / (1024 * 1024)} MB")


# 3. Dictionary creation
def dictionary_creation_benchmark():
    start_time = time.time()
    data = {str(i): i for i in range(1000000)}
    end_time = time.time()
    print(f"Dictionary creation: {end_time - start_time} seconds, {sys.getsizeof(data) / (1024 * 1024)} MB")


# 4. String concatenation
def string_concatenation_benchmark():
    start_time = time.time()
    data = ''.join([str(i) for i in range(1000000)])
    end_time = time.time()
    print(f"String concatenation: {end_time - start_time} seconds, {sys.getsizeof(data) / (1024 * 1024)} MB")


# Run the benchmarks
large_array_benchmark()
list_comprehension_benchmark()
dictionary_creation_benchmark()
string_concatenation_benchmark()
