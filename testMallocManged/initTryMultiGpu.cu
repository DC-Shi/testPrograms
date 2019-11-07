#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

// CUDA kernel to fill elements of array
__global__
void fill(size_t n, float *arr, size_t gpuId)
{
  // size_t force casting is required! otherwise it would always handling elements idx<2^32
  size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (index < n)
    arr[index] = arr[index] + (index & 0xFFF);
}

__global__
void init(size_t n, float *arr, size_t gpuId)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    arr[index] = 123.0001 + index % 16;
}
  
int main(void)
{
  // 7-billion array, 28G mem.
  // Let's test whether it would automatically using two GPU memory.
  size_t N = 7l*(1l<<30);
  float *x;
  size_t blockSize = 1024;
  size_t numBlocks;
  size_t totalGpuInitUsed = 2;
  size_t calcUseGpus = 2;
  // 4-billion array, 16G mem(8+8) init and 1 gpu calc: compute and memory are well-ballanced.
  // But takes about 10 mins to finish. 8-billion array, 32GB, takes 12mins to finish.
  
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  
  // Init on multi GPU
  for(size_t i = 0; i < totalGpuInitUsed; i++)
  {
    cudaSetDevice(i);
    numBlocks = (N/totalGpuInitUsed + blockSize - 1) / blockSize;
    printf("Init on %lu.\n", i);
    init<<<numBlocks, blockSize>>>(N/totalGpuInitUsed, &(x[N*i/totalGpuInitUsed]), i);//
  }
  // Advice mem to device 0
  gpuErrchk( cudaMemAdvise(x, sizeof(float)*N/2, cudaMemAdviseSetPreferredLocation, 0) );
  gpuErrchk( cudaMemAdvise(&x[N/2], sizeof(float)*N/2, cudaMemAdviseSetPreferredLocation, 1) );
  // Wait for GPU to finish before accessing on host

  for(size_t i = 0; i < totalGpuInitUsed; i++)
  {
    cudaSetDevice(i);
    // Need to do device sync 1-by-1
    cudaDeviceSynchronize();
  }
  gpuErrchk( cudaGetLastError() );

  // Launch kernel on multi GPU
  for(size_t i = 0; i < calcUseGpus; i++)
  {
    cudaSetDevice(i+1);
    numBlocks = (N/calcUseGpus + blockSize - 1) / blockSize;
    fill<<<numBlocks, blockSize>>>(N/calcUseGpus, &(x[N*i/calcUseGpus]), i);//
  }

  // Launch kernel on 8-billion elements on the GPU
  //numBlocks = (N + blockSize - 1) / blockSize;
  //fill<<<numBlocks, blockSize>>>(N, x);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  gpuErrchk( cudaGetLastError() );
  
  // Question: How is memory are located and migrated??
  // Tested result: first 3 seconds no mem access.
  // 0.226s D2H only, then both D2H and H2D, takes 7.7s. Memory access: 8s.

  // Output result
  printf("x[%lu] updated to %f\n", N-1, x[N-1] );
  printf("x[%lu] updated to %f\n", 123, x[123] );
  
  // Free memory
  cudaFree(x);
  
  return 0;

  // Conclusion:
  // Memory is automatically copied back to host memory, not other GPU's memory.
  // Next: try cudaMemAdvise
  // Tried, no luck
}
