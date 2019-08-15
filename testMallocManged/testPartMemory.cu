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

__global__ void initMemory(size_t position, size_t* array)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    array[position + idx] = idx;
}


int main()
{
    size_t *ret;
    //size_t size = 12345678901234l;
    size_t size = 4294967296l * 16384l; // 2^32 * 2^14 = 2^46, 2^44 numbers
    printf("Allocate %zu bytes, %zu MB, %zu GB, %zu TB\n", size, size/1048576, size/1048576/1024, size/1048576/1048576 );
    cudaMallocManaged(&ret, size );
// Do not prefetch, it will consume too much.
//    cudaMemPrefetchAsync(ret, size, 0);
    gpuErrchk( cudaGetLastError() );

    // each kernel run would visit 1M numbers, 4MB memory.
    size_t threadsPerBlock = 1024;
    size_t blocks = 1024*1;
    
    // Visit 1M*10240, total 10G numbers, 40GB memory
    for ( size_t i = 0; i < 10240; i++ )
    {
        size_t position = 1048576l*1024l*i;  // 2^20 * 2^10 * i = 2^30 * i 
        initMemory<<< blocks , threadsPerBlock >>>(position, ret);  // launch kernel
        gpuErrchk( cudaDeviceSynchronize() );
        if ( i % 1024 == 0 )
        {
          printf("Visited 400M @ %zu GB = %zu TB, array[%zu] = %zu\n",
  position*4/1048576/1024, position*4/1048576/1048576,
  position+7, ret[position+7]);
//#          sleep(1);
        }
    }


    printf("Finished.\n");
    cudaFree(ret); 
    return 0;
}
