import time
import csv
import torch
import torch.nn.functional as F
import re


# torch is required to run the GPU benchmarks
def benchmark_gpu():
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        N = 5000
        results = []

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timestamp = re.sub(r'[: ]', '_', timestamp)

        # Memory Transfer
        start = time.time()
        A_cpu = torch.randn([N, N])
        B_cpu = torch.randn([N, N])
        A_gpu = A_cpu.to('cuda')
        B_gpu = B_cpu.to('cuda')
        torch.cuda.synchronize()
        end = time.time()
        results.append(["CPU to GPU memory transfer", end - start])

        start = time.time()
        A_cpu = A_gpu.to('cpu')
        B_cpu = B_gpu.to('cpu')
        end = time.time()
        results.append(["GPU to CPU memory transfer", end - start])

        # Matrix Multiplication
        start = time.time()
        C = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Matrix multiplication on GPU", end - start])

        # Element-wise operations
        start = time.time()
        C = A_gpu + B_gpu
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Element-wise addition on GPU", end - start])

        # Convolutional operation
        X = torch.randn([1, 3, 224, 224], device='cuda')
        conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()
        start = time.time()
        Y = conv(X)
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Convolutional operation on GPU", end - start])

        # Max Pooling operation
        pool = torch.nn.MaxPool2d(2, 2)
        start = time.time()
        Y = pool(Y)
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Max Pooling operation on GPU", end - start])

        # Batch Normalization
        batch_norm = torch.nn.BatchNorm2d(1).cuda()
        start = time.time()
        Y = batch_norm(A_gpu.unsqueeze(0).unsqueeze(0))
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Batch Normalization on GPU", end - start])

        # ReLU Activation
        relu = torch.nn.ReLU().cuda()
        start = time.time()
        Y = relu(A_gpu)
        torch.cuda.synchronize()
        end = time.time()
        results.append(["ReLU Activation on GPU", end - start])

        # Softmax Activation
        softmax = torch.nn.Softmax(dim=1).cuda()
        start = time.time()
        Y = softmax(A_gpu)
        torch.cuda.synchronize()
        end = time.time()
        results.append(["Softmax Activation on GPU", end - start])

        # Save results to CSV file
        save_results_to_csv(results, timestamp)

    else:
        print("No CUDA-enabled GPU is available.")


def save_results_to_csv(results, timestamp):
    file_name = f"gpubenchmark_results_{timestamp}.csv"
    transposed_results = list(map(list, zip(*results)))  # Transpose the results list
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Operation"] + transposed_results[0])  # Write test names as column headers
        writer.writerow(["Execution Time (seconds)"] + transposed_results[1])  # Write results in the corresponding column
    print(f"Benchmark results saved to {file_name}")


if __name__ == "__main__":
    benchmark_gpu()
