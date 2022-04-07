#include <stdio.h>

__global__ void secondParallel()
{
  printf("This is running in parallel.\n");
}

__global__ void firstParallel()
{

  printf("This is running in parallel.\n");
  __syncthreads();
  secondParallel<<<2, 2>>>();
  cudaDeviceSynchronize();
  __syncthreads();

}


__global__ void ChildKernel(int* data, int n, int offset)
{     //Operate on data
  int idx = threadIdx.x + blockIdx.x * gridDim.x;
  if (idx + offset < n)
  {
    for (int j = 0; j < 20; j++)
      data[idx]+=(int)tan((double)data[idx]);
  }
}


__global__ void ParentKernel(int *data, int n)
{
  if (threadIdx.x == 0)
  {
    ChildKernel<<<2, 32>>>(data, n, threadIdx.x);
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();
  }
  __syncthreads();     //Operate on data

  //Operate on data
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
  {
    data[idx] = (int)tan((double)data[idx]);
  }
}

__global__ void initData(int *data, int n)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
  {
    data[idx] = idx;
  }
}


int main()
{
  int* data;
  int N = 2<<12;


  cudaMalloc(&data, N*sizeof(int));
  cudaError_t tmp = cudaGetLastError();
  if (tmp != 0)    printf("Error on cudaMalloc %d\n", tmp);
  initData<<<N/32, 32>>>(data, N); 
  ParentKernel<<<N/32, 32>>>(data, N); 

  cudaDeviceSynchronize();

  int * data_h;
  data_h = (int*)malloc(N*sizeof(int));

  cudaMemcpy(data_h, data, N*sizeof(int), cudaMemcpyDeviceToHost);
  for ( int idx = 0; idx < N; idx++)
  {
    if (idx % (N/20) == 0) 
    {
      printf("[%d] = %d\n", idx, data_h[idx] );
    }
  }
  
  printf("Finished\n");
}
