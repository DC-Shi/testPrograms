// Test run kernel in kernel

#include <stdio.h>
#include "cudaheader.cuh"

__device__ int d_v = -123;
__device__ int* d_p;

__global__ void print_k()
{
    printf("print: d_v = %d, (%d,%d)\n", d_v, blockIdx.x, threadIdx.x);
    printf("print: d_p = %d, (%d,%d)\n", d_p, blockIdx.x, threadIdx.x);
}

__global__ void init_k()
{
    printf("[init] the pointer %p\n", d_p);
    d_p = nullptr;
    printf("[init] the pointer is now %p \n", d_p);
}

cudaError_t runTest1()
{
    cudaError_t ret;
    print_k<<<2,1>>>();
    ret = cudaDeviceSynchronize();
    gpuErrchk("print kernel", ret);
    
    init_k<<<1,1>>>();
    ret = cudaDeviceSynchronize();
    gpuErrchk("init kernel", ret);
    
    return ret;
}

int main()
{
    printf("=====================================\n");
    
    int *p_h;
    printf("%s:%s @L%d, p_h @ %p\n", __FILE__, __FUNCTION__, __LINE__, p_h);
    p_h = nullptr;
    printf("%s:%s @L%d, p_h @ %p\n", __FILE__, __FUNCTION__, __LINE__, p_h);

    if (runTest1() != cudaSuccess)
        printf ("xxxFAILED Test 1\n");
    else
        printf ("SUCCEEDED Test 1\n");
}
