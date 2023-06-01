# CPU Benchmarks
import time
import math
import numpy as np
import random


# numpy is required
# 1. Prime calculation
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


start = time.time()
primes = [x for x in range(1, 100000) if is_prime(x)]
end = time.time()
print(f"Prime calculation: {end - start} seconds")

# 2. Floating point arithmetic
N = 10 ** 7
A = np.random.rand(N)
B = np.random.rand(N)

start = time.time()
C = A * B
end = time.time()
print(f"Floating point arithmetic: {end - start} seconds")

# 3. Matrix multiplication
N = 500
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start = time.time()
C = A @ B
end = time.time()
print(f"Matrix multiplication: {end - start} seconds")


# 4. Fibonacci sequence (recursion)
def fib(n):
    if n <= 1:
        return n
    else:
        return (fib(n - 1) + fib(n - 2))


start = time.time()
fib(30)
end = time.time()
print(f"Fibonacci sequence (recursion): {end - start} seconds")

# 5. Sorting
N = 10 ** 6
A = list(range(N))
random.shuffle(A)

start = time.time()
A.sort()
end = time.time()
print(f"Sorting: {end - start} seconds")

# 6. Searching
target = A[-1]  # worst case

start = time.time()
index = A.index(target)
end = time.time()
print(f"Searching: {end - start} seconds")
