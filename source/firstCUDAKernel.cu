/* 
This is my first attemt at creating a kernel that can be used to change a triangles position if the right changes are made
*/
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define pi 3.14f
#define THREADS_PER_BLOCK 32

struct Point {
    float X, Y, Z;
};

void PointInit (Point* p, float x, float y, float z)
{
    p->X = x;
    p->Y = y;
    p->Z = z;
}


__global__ void Translate (Point* arr, float tx, float ty, float tz, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) 
    {
        arr[index].X += tx;
        arr[index].Y += ty;
        arr[index].Z += tz;
    }
}


int main ()
{
    Point vertexArr[3];
    PointInit(&vertexArr[0], 1.0f, 1.0f, 0.0f);
    PointInit(&vertexArr[1], 4.0f, 4.0f, 0.0f);
    PointInit(&vertexArr[2], 7.0f, 1.0f, 0.0f);

    // Host memory
    Point* h_points = vertexArr;
    
    // Device memory
    int size = 3 * sizeof(Point);
    Point* d_points;
    cudaMalloc((void**)&d_points, size);
    cudaMemcpy(d_points, h_points, size, cudaMemcpyHostToDevice);

    float tx = 1.0f;
    float ty = 1.0f;
    float tz = 0.0f;

    Translate<<<(3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_points, tx, ty, tz, 3);
    cudaDeviceSynchronize();

    cudaMemcpy(h_points, d_points, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++)
        printf("%f, %f, %f\n", vertexArr[i].X, vertexArr[i].Y, vertexArr[i].Z);

    float theta_deg = 45.0f;
    float theta_rad = theta_deg * pi / 180.0f;
    Rotate<<<(3 + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK>>>(d_points, theta_rad);

    cudaMemcpy(h_points, d_points, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++)
        printf("%f, %f, %f\n", vertexArr[i].X, vertexArr[i].Y, vertexArr[i].Z);
    
    return 0;
}

