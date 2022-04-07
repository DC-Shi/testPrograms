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

////////////////////// Test 2
__device__ float test2_d_f(int v)
{
    float tmp = 0;
    tmp = 1.0 /(1+exp(v*1.0));
    return tmp;
}



__global__ void test2_kernel()
{
    int idx = blockIdx.x*gridDim.x + threadIdx.x;
    float val = test2_d_f(idx);
    printf("kernel (%d,%d) calculated %f\n", blockIdx.x, threadIdx.x, val);
}

cudaError_t runTest2()
{
    test2_kernel<<<2,3>>>();
    cudaError_t ret = cudaDeviceSynchronize();
    gpuErrchk("host process", ret);
    
    return ret;
}


////////////////////// Test 3
__device__ float val3[10];
__global__ void test3_g_f(int v, float* ret, int idx)
{
    ret[idx] = 2.0 /(1.0+exp(v*1.0));
}
__global__ void test3_kernel()
{
    int idx = blockIdx.x*gridDim.x + threadIdx.x;
    test3_g_f<<<1,1>>>(idx, val3, idx);

    // k1<<<10,1>>>(val3, newidx, offset);
    // cudaDeviceSynchronize();
    // k2<<<>>>(val3);
    // k3<<<>>>(val3);
    // k4<<<>>>(val3);
    cudaDeviceSynchronize();
    printf("kernel3 (%d,%d) calculated %f\n", blockIdx.x, threadIdx.x, val3[idx]);
}

cudaError_t runTest3()
{
    test3_kernel<<<2,3>>>();


    // k1<<<>>>();
    // k2<<<>>>();
    // k3<<<>>>();
    // k4<<<>>>();
    cudaError_t ret = cudaDeviceSynchronize();
    gpuErrchk("host process", ret);
    
    return ret;
}

int main()
{
    printf("=====================================\n");
    
    if (runTest2() != cudaSuccess)
        printf ("xxxFAILED Test 2\n");
    else
        printf ("SUCCEEDED Test 2\n");
    if (runTest3() != cudaSuccess)
        printf ("xxxFAILED Test 3\n");
    else
        printf ("SUCCEEDED Test 3\n");

}
