# CUDA
The repository contains Kernels from my CUDA learning journey and practice code. The "CUDA by example" book is being used.
The code was created using CUDA C++ and was writen on Kaggle notebooks using the GPU T4 accelerator.
For the compilation and execution of the code was done using the following commands:
- !nvcc main.cu -o main
- !./main

## What I learned so far:
- CUDA execution model (grid, block, thread)
- `dim3` usage for 2D indexing
- Handling non-square matrices (N×M × M×K)
- Correct memory allocation with `cudaMalloc` / `cudaMemcpy`
- Debugging memory corruption and boundary conditions
