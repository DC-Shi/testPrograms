// Test run kernel in kernel

#include <stdio.h>
#include "cudaheader.cuh"

__device__ int v = -123;

__global__ void child_k()
{
    printf("child: v = %d, (%d,%d)\n", v, blockIdx.x, threadIdx.x);
}

__global__ void parent_k()
{
    v = 1;
    child_k<<<2,2>>> ();
    //v = 2; // race condition! printf in child_k shows v is 2
    atomicAdd(&v, blockIdx.x*gridDim.x + threadIdx.x); // try atomicadd, we get 3
    //v += blockIdx.x*gridDim.x + threadIdx.x; // race condition! printf in child_k shows 0
    cudaError_t ret = cudaDeviceSynchronize();
    printf("kernel (%d,%d) returned %d\n", blockIdx.x, threadIdx.x, ret);
}

cudaError_t runTest1()
{
    parent_k<<<2,1>>>();
    cudaError_t ret = cudaDeviceSynchronize();
    gpuErrchk("host process", ret);
    
    return ret;
}

int main()
{
    printf("=====================================\n");
    
    if (runTest1() != cudaSuccess)
        printf ("xxxFAILED Test 1\n");
    else
        printf ("SUCCEEDED Test 1\n");
}
