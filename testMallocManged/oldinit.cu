#include <stdio.h>

// CUDA kernel to fill elements of array
__global__
void fill(int n, float *arr)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    arr[index] = arr[index] + index;
}
  
int main(void)
{
  int N = 1<<21;
  float *x;
  float *h_arr;
  
  // Allocate GPU Memory -- accessible only from GPU
  cudaMalloc((void**) &x, N*sizeof(float));
  h_arr = (float *)malloc(N*sizeof(float));

  // initialize x and y arrays on the host side
  for (int i = 0; i < N; i++) {
    h_arr[i] = 0.31f;
  }

  // Sends data to device
  cudaMemcpy(x, h_arr, N*sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel on 1M elements on the GPU
  int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;
  fill<<<numBlocks, blockSize>>>(N, x);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Retrieves data from device 
  cudaMemcpy(h_arr, x, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Output result
  printf("h_arr[%d] updated to %f\n", 10, h_arr[10]);

  // Free memory
  cudaFree(x);
  free(h_arr);
  return 0;
}
