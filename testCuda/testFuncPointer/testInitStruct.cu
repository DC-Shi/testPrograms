// Test init device struct by kernel function
#include <stdio.h>


#include "cudaheader.cuh"

__device__ int arr_d;
__global__ void kernInit(int *arr_in)
{
    arr_d = 3;
    printf("arr_d(%p) = %d\n", &arr_d, arr_d);
    printf("arr_in(%p)\n", arr_in);
    printf("arr_in = %d\n", *arr_in);
}


int main()
{
    printf("directly from host: arr_d address = %p\n", &arr_d);
    int* arr_host;

    gpuErrchk("load from symbol", cudaGetSymbolAddress((void**)&arr_host, arr_d));
    printf("after using cudaGetSymbolAddress: arr_d(%p), arr_host(%p)\n", &arr_d, &arr_host);



    //kernInit<<<1,1>>>(&arr_d);  // Failed, since you put into wrong pointer : illegal memory access was encountered
    kernInit<<<1,1>>>(arr_host); // Success
    gpuErrchk("after kernel", cudaDeviceSynchronize());

    // Try to access value from host
    printf("host access, x=%d\n", arr_d);
}

