#include<stdio.h>
#include <unistd.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

__global__ void AplusB_wait(int *ret, int a, int N, clock_t sleepInterval)
{
    clock_t start = clock64();
    while ( clock64() < start + sleepInterval ) { }
    size_t gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if ( gindex < N ) // Only change the needed.
      ret[gindex] = a + gindex;
}


int main()
{
    int *ret;
    size_t size = 1234567890012l;
    printf("Allocate %zu bytes, %zu MB\n", size, size/1048576);
    cudaMallocManaged(&ret, size );
    gpuErrchk( cudaPeekAtLastError() );

    size_t lower = 1024*1024; // at least we can run with these gpu memory.
    size_t upper = 21234567890; // running with these memory would lagging.
    size_t splitter = 21234567890; // just over 20G, make sure different GPU use differernt memory.

    int gpuCount = 2;
    int GPUs[] = {1, 3};   // Use these gpu for testing.
    int failed[2] = {0};


    while(lower < upper - 65536)
    {
       size_t testing = (lower + upper) / 2;
       size_t totalThreads = testing / 4; // 4 bytes per slot.
       size_t threadsPerBlock = 1024;
       size_t blocks = (totalThreads+threadsPerBlock-1) / threadsPerBlock;
       printf("Testing %zu B, %zu KB, %zu MB, %zu GB\n", testing, testing/1024, testing/1048576, testing / 1073741824);
       
       for ( int i = 0; i < gpuCount; i++ )
       {
          failed[i] = 0; // clear flag
          cudaSetDevice(GPUs[i]);   // run on this GPU
          int *current = &ret[i*splitter];
          AplusB_wait<<< blocks , threadsPerBlock >>>(current, 10, totalThreads, 123456);  // launch kernel
       }


       for ( int i = 0; i < gpuCount; i++ )
       {
          cudaSetDevice(GPUs[i]);   // run on this GPU
          cudaDeviceSynchronize();
          cudaError_t err = cudaGetLastError();
         if (err == cudaSuccess)
         {
            printf("OK for GPU %d\n", GPUs[i]);
            lower = testing;
         }
         else
         {
            fprintf(stderr,"Failed on GPU %d (%s), size = %zu B\n", GPUs[i], cudaGetErrorString(err), testing);
            upper = testing - 1;
         }
      }
    }
    cudaSetDevice(1);
    AplusB_wait<<< 50, 1000 >>>(ret, 10, 100000000, 1234567);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    printf("Wait 5 seconds\n");
    sleep(5);
//    for(int i = 0; i < 1000; i++)
//        printf("%d: A+B = %d\n", i, ret[i]);
    cudaFree(ret); 
    return 0;
}
