#include <stdio.h>

// CUDA kernel to fill elements of array
__global__
void fill(size_t n, float *arr)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    arr[index] = arr[index] + index;
}
  
int main(void)
{
  // 8G array, 32G mem.
  // Let's test whether it would automatically using two GPU memory.
  size_t N = 1<<33;
  float *x;
  
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  
  // initialize x and y arrays on the host side
  for (int i = 0; i < N; i++) {
    x[i] = 0.31f;
  }
  
  // Launch kernel on 1M elements on the GPU
  int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;
  fill<<<numBlocks, blockSize>>>(N, x);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // Output result
  printf("x[%d] updated to %f\n", 10, x[10]);
  
  // Free memory
  cudaFree(x);
  
  return 0;
}
