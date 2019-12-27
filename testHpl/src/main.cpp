#include <stdio.h>
#include "cpu_side.h"
#include "bug.cuh"

extern "C" int main()
{
    int* b;
    int N=256;
//    cudaMallocManaged(&b, N*sizeof(int));
    //cudaCheckError();

    //for(int i=0; i<N; i++)
    //  b[i] = i*i;

    /// Some GPU code
    bugTester();


    printf("===================================\n");

    for(int i=0;i<16;i++)
    {
        f_poly8(i);
    }

    printf("===================================\n");

//    calcdouble<<<dimGrid,dimBlock1>>>();
//    cudaCheckError();
//    cudaDeviceSynchronize();
//    cudaCheckError();

//    for(int i=0;i<CpuModulus;i++)
//    {
//        f(i);
//    }

//    cudaFree(b);
    //cudaCheckError();
    
    return 0;
}
