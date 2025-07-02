# import cupy as cp
# import numpy as np
# import time


# a = cp.random.rand(10000, 10000)
# b = cp.random.rand(10000, 10000)

# # GPU-accelerated matrix multiplication
# start_gpu = time.time()
# c = cp.dot(a, b)
# end_gpu = time.time()
# diff_gpu = end_gpu - start_gpu

# start_np = time.time()
# n = np.dot(a, b)
# end_np = time.time()
# diff_np = end_np - start_np

# print("GPU time :" + str(diff_gpu))
# print("Numpy time :" + str(diff_np))

import time
import numpy as np

try:
    import cupy as cp
    cp.show_config()
    cupy_available = True
except ImportError:
    cupy_available = False
    print("⚠️ CuPy is not installed. GPU test will be skipped.\nYou can install it with:")
    print("   pip install cupy-cuda12x  # Replace 12x with your CUDA version\n")

# Matrix size
N = 15000
print(f"Creating two {N}x{N} matrices...")

# CPU arrays
# A_cpu = np.random.rand(N, N)
# B_cpu = np.random.rand(N, N)

A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

# CPU benchmark
print("Running matrix multiplication on CPU (NumPy)...")
start_cpu = time.perf_counter()
C_cpu = np.dot(A_cpu, B_cpu)
end_cpu = time.perf_counter()
cpu_time = end_cpu - start_cpu
print(f"✅ CPU Time: {cpu_time:.4f} seconds")

# GPU benchmark (if CuPy is available)
if cupy_available:
    # A_gpu = cp.asarray(A_cpu)
    # B_gpu = cp.asarray(B_cpu)

    # Use separate GPU-generated matrices
    A_gpu = cp.random.rand(N, N, dtype=cp.float32)
    B_gpu = cp.random.rand(N, N, dtype=cp.float32)

    print("Running matrix multiplication on GPU (CuPy)...")
    cp.cuda.Device().synchronize()
    start_gpu = time.perf_counter()
    C_gpu = cp.dot(A_gpu, B_gpu)
    cp.cuda.Device().synchronize()
    end_gpu = time.perf_counter()

    gpu_time = end_gpu - start_gpu
    print(f"✅ GPU Time: {gpu_time:.4f} seconds")
    print(f"🚀 Speedup: {cpu_time / gpu_time:.2f}x faster on GPU")
else:
    print("❌ Skipping GPU test — CuPy not available.")
