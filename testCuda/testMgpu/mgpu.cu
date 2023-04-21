#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel1(int i) {
  printf("Kernel1 with input %d\n", i);
    // kernel 1 code here
}

__global__ void kernel2(int i) {
  printf("Kernel2 with input %d\n", i);
    // kernel 2 code here
}

void printDevInfo(int i)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}


int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2) {
        printf("Error: Two or more GPUs are required for this program.\n");
        return 1;
    }
    else
    {
      printf("%d deviced detected.\n", device_count);
      for (int i = 0; i < device_count; i++)
        printDevInfo(i);
    }

    int blocks=1, threads=1;

    cudaSetDevice(0);
    kernel1<<<blocks, threads>>>(0);

    cudaSetDevice(1);
    kernel2<<<blocks, threads>>>(1);

    cudaDeviceSynchronize();

    return 0;
}
