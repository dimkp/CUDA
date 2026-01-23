# CUDA
The repository contains Kernels from my CUDA learning journey and practice code. The "CUDA by example" book is being used.
The code was created using CUDA C++ and was writen on Kaggle notebooks using the GPU T4 accelerator or on my local system, using Visual Studio 2022 and a RTX 3060 TI graphics card.
For the compilation and execution of the code was done using the following commands:
- !nvcc main.cu -o main
- !./main

## What I Have Learned So Far

### CUDA Programming Model
- Clear distinction between **Host (CPU)** and **Device (GPU)** code
- Understanding how CUDA kernels work using:
  - `__global__`
  - `blockIdx`, `threadIdx`, `blockDim`, `gridDim`
- Mapping data elements to GPU threads correctly

### Kernel Launch & Execution
- Correct usage of CUDA kernel launch syntax:
  `kernel<<<blocks, threads>>>(...);`

### Understanding and fixing common launch errors:
- invalid argument
- launching kernels with blocks = 0
- Proper synchronization with: `cudaDeviceSynchronize();`

### Memory Management (Host <-> Device)
- Allocating GPU memory using: `cudaMalloc`
- Transferring data between host and device using: `cudaMemcpy`
- Correct memory deallocation with: `cudaFree`
- Understanding the difference between:
  - host pointers
  - device pointers
- Avoiding common memory bugs:
  - `incorrect byte-size calculations`
  - `uninitialized or null pointers`

### Error Handling & Debugging
- Usage of `cudaGetLastError()` to catch kernel launch failures
- Implementing custom CUDA error-checking macros
- Debugging runtime issues such as:
  - invalid kernel launches
  - incorrect memory allocation
  - synchronization mistakes
- Using printf inside CUDA kernels for debugging

### Grid-Stride Loops
- Implementing grid-stride loops to:
  - handle arbitrary input sizes
  - decouple problem size from grid configuration
- Understanding why grid-stride loops are essential for scalable CUDA kernels

### Shared Memory
- Using shared memory with: `__shared__`
- Understanding:
  - why shared memory is faster than global memory
  - when synchronization is required
- Correct usage of: `__syncthreads();`
- Avoiding deadlocks by ensuring all threads reach finish execution correctly

### Dot Product on the GPU
- Implemented a dot product using:
  - grid-stride loops
  - shared memory reduction
- Computing partial results per block
- Accumulating final results on the host
- Identifying and fixing bugs related to:
  - incorrect memory allocation sizes
  - pointer arithmetic mistakes
  - integer overflow risks

### Multi-File CUDA Project Structure
- Proper separation of CUDA projects using:
  - `.cu` files for kernels and CUDA logic
  - `.cuh` headers for declarations
  - `.cpp` / `.cu` files for host-side code
- Avoiding bad practices such as including .cu files directly
- Understanding how NVCC and MSVC work together during compilation
