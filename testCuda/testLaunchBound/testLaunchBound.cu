// Test run kernel in kernel

#include <stdio.h>
#include "cudaheader.cuh"
#include "kernel.cuh"

__device__ int v = -123;



cudaError_t runTest1()
{
    double * tempArray;
    size_t num_elements = 1024;
    cudaMallocManaged(&tempArray, num_elements*sizeof(double));


    kernel_64reg<<<1,512>>>(tempArray);
    cudaError_t ret = cudaGetLastError();
    gpuErrchk("test1 512 threads/block launch", ret);
    ret = cudaDeviceSynchronize();
    gpuErrchk("test1 512 threads/block sync", ret);

    cudaFree(tempArray);
    
    return ret;
}

cudaError_t runTest2()
{
    double * tempArray;
    size_t num_elements = 1024;
    cudaMallocManaged(&tempArray, num_elements*sizeof(double));


    kernel_64reg<<<1,1024>>>(tempArray);
    cudaError_t ret = cudaGetLastError();
    gpuErrchk("test2 1024 threads/block launch", ret);
    ret = cudaDeviceSynchronize();
    gpuErrchk("test2 1024 threads/block sync", ret);

    cudaFree(tempArray);
    
    return ret;
}

int main()
{
    printf("=====================================\n");
    
    if (runTest1() != cudaSuccess)
        printf ("xxxFAILED Test 1\n");
    else
        printf ("SUCCEEDED Test 1\n");

    
    if (runTest2() != cudaSuccess)
        printf ("xxxFAILED Test 2\n");
    else
        printf ("SUCCEEDED Test 2\n");

}
