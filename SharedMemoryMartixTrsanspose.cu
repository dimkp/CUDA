%%writefile main.cu
#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 16

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


__global__ void transposeShared (int* in, int* out, int rows, int cols)
{
    __shared__ int tile[TILE_DIM][TILE_DIM + 1]; // +1 για αποφυγή bank conflicts

    int col = blockIdx.x * TILE_DIM + threadIdx.x; 
    int row = blockIdx.y * TILE_DIM + threadIdx.y; 

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int r = row + j;
        if (r < rows && col < cols) 
            tile[threadIdx.y + j][threadIdx.x] = in[r * cols + col];
    }
    __syncthreads();

    int out_col = threadIdx.x + blockIdx.y * TILE_DIM; 
    int out_row = threadIdx.y + blockIdx.x * TILE_DIM; 

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int rr = out_row + j;
        if (rr < cols && out_col < rows) 
            out[rr * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
    }
}


int main ()
{
    int numRows = 500, numCols = 400;

    int size = numRows * numCols;
    size_t byteSize = size * sizeof(int);
    
    int* x;
    x = (int*)malloc(byteSize);

    int* o;
    o = (int*)malloc(byteSize);

    if (!x || !o)
    {
        printf("\nHost memory failure!\n\n");
        free(x);
        free(o);
        return 1;
    }
    HostMatrixInit(x, size);
    
    int* dx;
    cudaErrorCheck(cudaMalloc((void**)&dx, byteSize), "dx malloc");
    
    int* dOut;
    cudaErrorCheck(cudaMalloc((void**)&dOut, byteSize), "dOut malloc");

    cudaErrorCheck(cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice), "x -> dx copy");

    dim3 threads_per_block(TILE_DIM, BLOCK_ROWS);
    dim3 blocks_per_grid((numCols + TILE_DIM - 1) / TILE_DIM, 
                         (numRows + TILE_DIM - 1) / TILE_DIM); 

    transposeShared<<<blocks_per_grid, threads_per_block>>>(dx, dOut, numRows, numCols);
    cudaErrorCheck(cudaGetLastError(), "Launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Synchornization");
    cudaErrorCheck(cudaMemcpy(o, dOut, size, cudaMemcpyDeviceToHost), "dOut -> o copy");

    // Memory cleanup
    free(x);
    free(o);
    cudaErrorCheck(cudaFree(dx), "dx free");
    cudaErrorCheck(cudaFree(dOut), "dOut free");
    return 0;
}
