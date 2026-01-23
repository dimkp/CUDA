/*The following file includes the kernel calls for every function of the project*/
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Error Checking
static void cudaErrorCheck(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
		printf("Error: %s -> %s \n\n", msg, cudaGetErrorString(err));
		std::exit(1);
	}
}

void CopyKernelCall(const int* x, int* y, const int size, const int byteSize);

void AddKernelCall(const int* x, const int* y, int* z, const int size, const int byteSize);

void DotProductKernelCall(const int* x, const int* y, int* result, const int size, const int byteSize);
