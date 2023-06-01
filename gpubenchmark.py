import time
import torch
import torch.nn.functional as F


# torch is required to run the gpu benchmarks
def benchmark_gpu():
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        N = 5000
        # Memory Transfer
        start = time.time()

        #using randomized data for benchmarks
        A_cpu = torch.randn([N, N])
        B_cpu = torch.randn([N, N])
        A_gpu = A_cpu.to('cuda')
        B_gpu = B_cpu.to('cuda')

        torch.cuda.synchronize()

        end = time.time()

        print(f"CPU to GPU memory transfer: {end - start} seconds")

        start = time.time()

        A_cpu = A_gpu.to('cpu')
        B_cpu = B_gpu.to('cpu')

        end = time.time()

        print(f"GPU to CPU memory transfer: {end - start} seconds")

        # Matrix Multiplication
        start = time.time()

        C = torch.matmul(A_gpu, B_gpu)

        torch.cuda.synchronize()

        end = time.time()

        print(f"Matrix multiplication on GPU: {end - start} seconds")

        # Element-wise operations
        start = time.time()

        C = A_gpu + B_gpu

        torch.cuda.synchronize()

        end = time.time()

        print(f"Element-wise addition on GPU: {end - start} seconds")

        # Convolutional operation
        X = torch.randn([1, 3, 224, 224], device='cuda')
        conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, device='cuda')
        start = time.time()

        Y = conv(X)

        torch.cuda.synchronize()

        end = time.time()

        print(f"Convolutional operation on GPU: {end - start} seconds")

        # Max Pooling operation
        pool = torch.nn.MaxPool2d(2, 2, device='cuda')
        start = time.time()

        Y = pool(Y)

        torch.cuda.synchronize()

        end = time.time()

        print(f"Max Pooling operation on GPU: {end - start} seconds")

        # Batch Normalization
        batch_norm = torch.nn.BatchNorm2d(N).cuda()
        start = time.time()

        Y = batch_norm(A_gpu.unsqueeze(0))

        torch.cuda.synchronize()

        end = time.time()

        print(f"Batch Normalization on GPU: {end - start} seconds")

        # ReLU Activation
        relu = torch.nn.ReLU().cuda()
        start = time.time()

        Y = relu(A_gpu)

        torch.cuda.synchronize()

        end = time.time()

        print(f"ReLU Activation on GPU: {end - start} seconds")

        # Softmax Activation
        softmax = torch.nn.Softmax(dim=1).cuda()
        start = time.time()

        Y = softmax(A_gpu)

        torch.cuda.synchronize()

        end = time.time()

        print(f"Softmax Activation on GPU: {end - start} seconds")

    else:
        print("No CUDA-enabled GPU is available.")


if __name__ == "__main__":
    benchmark_gpu()
