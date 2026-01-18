%%writefile main.cu
#include <iostream>
#include <cuda_runtime.h>

#define imin(a,b) (a<b?a:b) // Limits the number of blocks per grid to a maximum of 32
#define N 1024
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID imin(32, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

// CUDA Error Checking
static void cudaErrorCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        printf("Error: %s -> %s \n\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}



__global__ void DotProduct (const int* a, const int* b, int* c, const int size)
{
    __shared__ int cache[THREADS_PER_BLOCK];
    
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int tmp = 0;
    while (threadIndex < size)
    {
        tmp += a[threadIndex] * b[threadIndex];
        threadIndex += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = tmp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}




int main ()
{
    int* ha, * hb, * partialC, c = 0;
    size_t byteSize = N * sizeof(int);
    ha = (int*)malloc(byteSize);
    hb = (int*)malloc(byteSize);
    partialC = (int*)malloc(BLOCKS_PER_GRID * sizeof(int));

    if (!ha || !hb || !partialC)
    {
        printf("Host Memory error!\n");
        free(ha);
        free(hb);
        free(partialC);
        return 1;
    }

    for (int i = 0; i < N; i++)
    {
        ha[i] = i + 1;
        hb[i] = i + 2;
    }

    int* da, * db, * dpc;
    cudaErrorCheck(cudaMalloc((void**)&da, byteSize), "da malloc");
    cudaErrorCheck(cudaMalloc((void**)&db, byteSize), "db malloc");
    cudaErrorCheck(cudaMalloc((void**)&dpc, BLOCKS_PER_GRID * sizeof(int)), "dpc malloc");

    cudaErrorCheck(cudaMemcpy(da, ha, byteSize, cudaMemcpyHostToDevice), "ha -> da copy");
    cudaErrorCheck(cudaMemcpy(db, hb, byteSize, cudaMemcpyHostToDevice), "hb -> db copy");

    DotProduct<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(da, db, dpc, N);
    cudaErrorCheck(cudaGetLastError(), "Kernel launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Synchronization");

    cudaErrorCheck(cudaMemcpy(partialC, dpc, BLOCKS_PER_GRID * sizeof(int), cudaMemcpyDeviceToHost), "dpc -> partialC copy");

    for (int i = 0; i < BLOCKS_PER_GRID; i++)
        c += partialC[i];

    printf("GPU result: %d\n", c);

    long long cpu_result = 0;
    for (int i = 0; i < N; i++) 
        cpu_result += (long long)ha[i] * hb[i];
    
    printf("CPU result  : %lld\n", cpu_result);

    if ((long long)c == cpu_result)
        printf("GPU and CPU results match\n");
    else
        printf("GPU and CPU results do not match\n");


    // Memory cleanup
    free(ha);
    free(hb);
    free(partialC);
    cudaErrorCheck(cudaFree(da), "Free da");
    cudaErrorCheck(cudaFree(db), "Free db");
    cudaErrorCheck(cudaFree(dpc), "Free dpc");
    return 0;
}
