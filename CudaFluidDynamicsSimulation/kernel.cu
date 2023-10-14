#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void arrayAddition(int* c, const int* a, const int* b, const int size)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void addWithCuda(int* c, int* a, int* b, int size)
{
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;
    arrayAddition << <NUM_BLOCKS, NUM_THREADS >> > (dev_c, dev_a, dev_b, size);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}