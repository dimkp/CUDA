#include <iostream>
#include "KernelHeader.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#define imin(a,b) (a<b?a:b)

#define THREADS_PER_BLOCK 256

// Array copy
__global__ void CopyKernel(const int* x, int* y, const int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		y[index] = x[index];
}

void CopyKernelCall(const int* x, int* y, const int size, const int byteSize)
{
	int BLOCKS_PER_GRID = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int* dx, * dy;
	cudaErrorCheck(cudaMalloc((void**)&dx, byteSize), "dx malloc");
	cudaErrorCheck(cudaMalloc((void**)&dy, byteSize), "dy malloc");
	cudaErrorCheck(cudaMemcpy(dx, x, byteSize, cudaMemcpyHostToDevice), "x -> dx copy");

	CopyKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx, dy, size);
	cudaErrorCheck(cudaGetLastError(), "Launch");
	cudaErrorCheck(cudaDeviceSynchronize(), "Sync");
	
	cudaErrorCheck(cudaMemcpy(y, dy, byteSize, cudaMemcpyDeviceToHost), "dy -> y copy");

	cudaErrorCheck(cudaFree(dx), "free dx");
	cudaErrorCheck(cudaFree(dy), "free dy");
}


// Array addition
__global__ void AddKernel(const int* x, const int* y, int* z, const int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size)
		z[index] = x[index] + y[index];
}

void AddKernelCall(const int* x, const int* y, int* z, const int size, const int byteSize)
{
	int BLOCKS_PER_GRID = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int* dx, * dy, * dz;
	
	cudaErrorCheck(cudaMalloc((void**)&dx, byteSize), "dx malloc");
	cudaErrorCheck(cudaMalloc((void**)&dy, byteSize), "dy malloc");
	cudaErrorCheck(cudaMalloc((void**)&dz, byteSize), "dz malloc");
	
	cudaErrorCheck(cudaMemcpy(dx, x, byteSize, cudaMemcpyHostToDevice), "x -> dx copy");
	cudaErrorCheck(cudaMemcpy(dy, y, byteSize, cudaMemcpyHostToDevice), "y -> dy copy");
	
	AddKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx, dy, dz, size);
	cudaErrorCheck(cudaGetLastError(), "Launch");
	cudaErrorCheck(cudaDeviceSynchronize(), "Sync");

	cudaErrorCheck(cudaMemcpy(z, dz, byteSize, cudaMemcpyDeviceToHost), "dz -> z copy");

	cudaErrorCheck(cudaFree(dx), "free dx");
	cudaErrorCheck(cudaFree(dy), "free dy");
	cudaErrorCheck(cudaFree(dz), "free dz");
}


// Dot Product of 2 arrays using shared memory
__global__ void DotProduct(const int* x, const int* y, int* partial_z, const int size)
{
	__shared__ int cache[THREADS_PER_BLOCK];
	int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	int cache_index = threadIdx.x;

	long long tmp = 0;
	while (thread_index < size)
	{
		tmp += x[thread_index] * y[thread_index];
		thread_index += blockDim.x * gridDim.x;
	}
	cache[cache_index] = tmp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cache_index < i)
			cache[cache_index] += cache[cache_index + i];
		__syncthreads();
		i /= 2;
	}

	if (cache_index == 0)
		partial_z[blockIdx.x] = cache[0];
}

void DotProductKernelCall(const int* x, const int* y, int* result, const int size, const int byteSize)
{
	int BLOCKS_PER_GRID = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	int* partial_z = (int*)malloc(BLOCKS_PER_GRID * sizeof(int));

	int* dx, * dy, * partial_dz;
	cudaErrorCheck(cudaMalloc((void**)&dx, byteSize), "dx malloc");
	cudaErrorCheck(cudaMalloc((void**)&dy, byteSize), "dy malloc");
	cudaErrorCheck(cudaMalloc((void**)&partial_dz, BLOCKS_PER_GRID * sizeof(int)), "partial_dz malloc");

	cudaErrorCheck(cudaMemcpy(dx, x, byteSize, cudaMemcpyHostToDevice), "x -> dx copy");
	cudaErrorCheck(cudaMemcpy(dy, y, byteSize, cudaMemcpyHostToDevice), "y -> dy copy");

	DotProduct<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx, dy, partial_dz, size);
	cudaErrorCheck(cudaGetLastError(), "Launch");
	cudaErrorCheck(cudaDeviceSynchronize(), "Sync");

	cudaErrorCheck(cudaMemcpy(partial_z, partial_dz, BLOCKS_PER_GRID * sizeof(int), cudaMemcpyDeviceToHost), "partial_dz -> partial_z copy");

	*result = 0;
	for (int i = 0; i < BLOCKS_PER_GRID; i++)
		*result += partial_z[i];

	cudaErrorCheck(cudaFree(dx), "free dx");
	cudaErrorCheck(cudaFree(dy), "free dy");
	cudaErrorCheck(cudaFree(partial_dz), "free partial_dz");
}
