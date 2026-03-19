#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <chrono>

int CPUCount(const std::vector<int>& arr, int search_number)
{
    int count = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] == search_number)
            count++;
    }
    return count;
}

__global__ void GPUCount(const int* arr, int size, int* count, int search_number)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = thread_index; i < size; i += stride)
    {
        if (arr[i] == search_number)
            atomicAdd(count, 1);
    }
}

int main()
{
    std::ifstream file("/kaggle/input/datasets/dimitriospapoulias/rand-numbers/random_numbers_2M_1_to_1000.txt");
    if (!file.is_open()) {
        std::cout << "Could not open file\n";
        return 1;
    }
    std::vector<int> arr;
    int x;
    while (file >> x) {
        arr.push_back(x);
    }
    file.close();

    int search_number = 100;
    
    auto start_cpu = std::chrono::high_resolution_clock::now();

    int cpu_count = CPUCount(arr, search_number);
    printf("CPU count: %d\n", cpu_count);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU time: " << cpu_time.count() << " ms\n";

    int* device_arr;
    int* device_count;

    cudaMalloc((void**)&device_arr, arr.size() * sizeof(int));
    cudaMalloc((void**)&device_count, sizeof(int));

    cudaMemcpy(device_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(device_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int device_id;
    cudaGetDevice(&device_id);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id);

    int threads_per_block = 256;
    int blocks_per_grid = numSMs * 32;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    GPUCount<<<blocks_per_grid, threads_per_block>>>(device_arr, arr.size(), device_count, search_number);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    int gpu_count = 0;
    cudaMemcpy(&gpu_count, device_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU count: %d\n", gpu_count);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU time: " << gpu_time << " ms\n";

    cudaFree(device_arr);
    cudaFree(device_count);

    return 0;
}
