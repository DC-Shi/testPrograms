/*
Compile this code:
    nvcc -o hello_gpu_async.exe hello.cu
Run the code:
    ./hello_gpu_async.exe
Profile the code:
    nsys profile -o profile_gpu_async --stat=true ./hello_gpu_async.exe

Profiling it, you can use:
    time (linux command)
    std::chrono (in C++), or clock() function, etc. (in C)
    (not required, profiling tools) Visual studio profiler, vtune, nsys 
*/

#include <iostream>
#include <cmath>

__global__
void saxpy_parallel(int n,
                    float a,
                    float *x,
                    float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main()
{
  int n = 1048576;  int check_idx = n - 10;
  float *x_h, *y_h; float *x_d, *y_d;
  
  // Allocate data
  //x_h = (float*)malloc(n*sizeof(float));
  //y_h = (float*)malloc(n*sizeof(float));
  cudaMallocHost(&x_h, n*sizeof(float));
  cudaMallocHost(&y_h, n*sizeof(float));
  cudaMalloc(&x_d, n*sizeof(float));
  cudaMalloc(&y_d, n*sizeof(float));

  // Initialize data
  for ( int i = 0; i < n; i++ )
  { // i^4+i+1
    x_h[i] = ( (i%251)*(i%251)*(i%251)*(i%251) ) + (i%251)+1;
    x_h[i] = fmod(x_h[i], 997.3);
    y_h[i] = -1;
  }
  printf("x[%d] = %f\n", check_idx, x_h[check_idx]);

  // Copy data from CPU to GPU
  cudaMemcpyAsync(x_d, x_h, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y_h, n*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy_parallel<<<4096,256>>>(n,2.0,x_d,y_d);
  
  cudaDeviceSynchronize();

  // Data copy back
  cudaMemcpyAsync(y_h, y_d, n*sizeof(float), cudaMemcpyDeviceToHost);

  // Check the output
  printf("Result y[%d] = %f\n", check_idx, y_h[check_idx]);
  if ( 2*x_h[check_idx] - 1 == y_h[check_idx] )
    printf("Match! y=2x-1\n");
  else
    printf("Wrong, should be %f\n", 2*x_h[check_idx] + 1);

  // Release all resource
  free(x_h);
  free(y_h);
  cudaFree(x_d);
  cudaFree(y_d);
  
  return 0;
}


