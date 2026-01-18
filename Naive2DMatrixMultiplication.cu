%%writefile main.cu
#include <iostream>
#include <cuda_runtime.h>

#define N 5
#define M 7
#define K 3
#define BLOCK 16

// CUDA Error Checking
static void cudaErrorCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        printf("Error: %s -> %s \n\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

void HostMatrixInit (int* x, int size)
{
    for (int i = 0; i < size; i++)
        x[i] = 2;
}


__global__ void MatrixMultiplicationKernel (int* a, int* b, int* c, int n, int m, int k)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y; // Current row
    int col = threadIdx.x + blockIdx.x * blockDim.x; // Current column

    if (row < n && col < k)
    {
        int tempSum = 0;
        for (int i = 0; i < m; i++)
            tempSum += a[i + row * m] * b[col + i * k];
        c[col + row * k] = tempSum;
    }
}


int main ()
{
    size_t sizeA = N * M * sizeof(int);
    size_t sizeB = M * K * sizeof(int);
    size_t sizeC = N * K * sizeof(int);
    int* a, * b, * c;
    a = (int*)malloc(sizeA);
    b = (int*)malloc(sizeB);
    c = (int*)malloc(sizeC);
    if (!a || !b || !c)
    {
        printf("\nHost memory failure!\n\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }
    HostMatrixInit(a, N * M);
    HostMatrixInit(b, M * K);
    
    int* da, * db, * dc;
    cudaErrorCheck(cudaMalloc((void**)&da, sizeA), "da malloc");
    cudaErrorCheck(cudaMalloc((void**)&db, sizeB), "db malloc");
    cudaErrorCheck(cudaMalloc((void**)&dc, sizeC), "dc malloc");

    cudaErrorCheck(cudaMemcpy(da, a, sizeA, cudaMemcpyHostToDevice), "a -> da copy");
    cudaErrorCheck(cudaMemcpy(db, b, sizeB, cudaMemcpyHostToDevice), "b -> db copy");

    dim3 threads_per_block(BLOCK, BLOCK);
    dim3 blocks_per_grid((K + BLOCK - 1) / BLOCK, // Columns of dc
                         (N + BLOCK - 1) / BLOCK); // Rows of dc

    MatrixMultiplicationKernel<<<blocks_per_grid, threads_per_block>>>(da, db, dc, N, M, K);
    cudaErrorCheck(cudaGetLastError(), "Launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Synchornization");
    cudaErrorCheck(cudaMemcpy(c, dc, sizeC, cudaMemcpyDeviceToHost), "dc -> c copy");

    // Memory cleanup
    free(a);
    free(b);
    free(c);
    cudaErrorCheck(cudaFree(da), "da free");
    cudaErrorCheck(cudaFree(db), "db free");
    cudaErrorCheck(cudaFree(dc), "dc free");
    return 0;
}
