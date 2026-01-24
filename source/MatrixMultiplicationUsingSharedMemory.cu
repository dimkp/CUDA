%%writefile main.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 10
#define M 15
#define K 13
#define BLOCKS 16

static void ck(cudaError_t e, const char* msg){
    if(e != cudaSuccess){
        printf("CUDA error: %s -> %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

void MatInit(int* x, int size){
    for(int i = 0; i < size; i++) x[i] = i + 1;
}

__global__ void MatrixMultSharedMemory(const int* a, const int* b, int* c, int n, int m, int k)
{
    __shared__ int As[BLOCKS][BLOCKS];
    __shared__ int Bs[BLOCKS][BLOCKS];

    int row = blockIdx.y * BLOCKS + threadIdx.y;
    int col = blockIdx.x * BLOCKS + threadIdx.x;

    int tmp = 0;

    int tiles = (m + BLOCKS - 1) / BLOCKS;
    for (int t = 0; t < tiles; t++)
    {
        int aRow = row;
        int aCol = t * BLOCKS + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (aRow < n && aCol < m) ? a[aRow * m + aCol] : 0;

        int bRow = t * TILE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] = (bRow < m && bCol < k) ? b[bRow * k + bCol] : 0;

        __syncthreads();

        for (int i = 0; i < BLOCKS; i++)
            tmp += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < k)
        c[row * k + col] = tmp;
}

int main()
{
    int sizeA = N * M;
    int sizeB = M * K;
    int sizeC = N * K;

    size_t bytesA = (size_t)sizeA * sizeof(int);
    size_t bytesB = (size_t)sizeB * sizeof(int);
    size_t bytesC = (size_t)sizeC * sizeof(int);

    int* a = (int*)malloc(bytesA);
    int* b = (int*)malloc(bytesB);
    int* c = (int*)malloc(bytesC);

    MatInit(a, sizeA);
    MatInit(b, sizeB);

    int *da=nullptr, *db=nullptr, *dc=nullptr;
    ck(cudaMalloc((void**)&da, bytesA), "malloc da");
    ck(cudaMalloc((void**)&db, bytesB), "malloc db");
    ck(cudaMalloc((void**)&dc, bytesC), "malloc dc");

    ck(cudaMemcpy(da, a, bytesA, cudaMemcpyHostToDevice), "copy a");
    ck(cudaMemcpy(db, b, bytesB, cudaMemcpyHostToDevice), "copy b");

    dim3 threads(BLOCKS, BLOCKS);
    dim3 blocks((K + BLOCKS - 1) / BLOCKS, (N + BLOCKS - 1) / BLOCKS);

    MatrixMultSharedMemory<<<blocks, threads>>>(da, db, dc, N, M, K);
    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "sync");

    ck(cudaMemcpy(c, dc, bytesC, cudaMemcpyDeviceToHost), "copy c back");

    // print C (host)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < K; j++){
            printf("%d ", c[i * K + j]);
        }
        printf("\n");
    }

    free(a); 
    free(b); 
    free(c);
    ck(cudaFree(da), "free da");
    ck(cudaFree(db), "free db");
    ck(cudaFree(dc), "free dc");
    return 0;
}
