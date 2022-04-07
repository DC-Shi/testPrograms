// Test run kernel in kernel

#include <stdio.h>
#include <stdint.h>
#include "cudaheader.cuh"

// Code is referenced from Robert Crovella on stackoverflow:
// https://stackoverflow.com/questions/30361459/what-is-the-behavior-of-thread-block-scheduling-to-specific-sms-after-cuda-kern
// Modified quite a lot to measure speed.

#define MAX_BLOCK 216
static __device__ int smidOfBlock[MAX_BLOCK];
static __device__ unsigned long long int speed[MAX_BLOCK];
static __device__ int deviceArray[MAX_BLOCK];
static __device__ int looping = 1;
static __device__ unsigned long long int overflow = 12345678;

static __device__ __inline__ uint32_t __smid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

// Launch kernel within SM
__global__ void loopKernel(int num_blocks)
{
  unsigned long long int lastTime = clock64();
  int idx = blockIdx.x;
  int currentLoop = 1;
  smidOfBlock[idx] = __smid();

  // Only do few blocks
  if (idx < MAX_BLOCK)
  {
    // Init the value
    deviceArray[idx] = 0;
    while ( looping )
    {
      deviceArray[idx]++;

      if ( deviceArray[idx] >= overflow)
      {
        deviceArray[idx] -= overflow;
        long long int currentTime = clock64();
        //atomicExch(&speed[idx], overflow * 100 * 1000/(currentTime - lastTime));
        speed[idx] = overflow * 100 * 1000/(currentTime - lastTime);
        lastTime = currentTime;
        if (idx == 0)
        {
          printf("Speed: ");
          for (int i = 0; i < num_blocks; i++)
          {
            printf("[%d]%3llu ", smidOfBlock[i], speed[i]);
          }
          printf("\n");

          currentLoop++;
          if (currentLoop > 20) 
          //looping = 0;
          atomicExch(&looping, 0);
        }
        // if no print, the kernel on sm!=0 did not stop
        // if print, all kernel will get the correct looping=0 value.
        // Add volatile to looping works, but quite slow, -60% performance.
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier
        //printf("tid[%d], looping[%d], curloop[%d]\n", idx, looping, currentLoop);
      }
    }
  }
    printf("Finished thread %d\n", idx);
}

int main(int argc, char *argv[])
{
  cudaError_t cudaRet;
  unsigned ranks = 20;
  if (argc > 1) ranks = atoi(argv[1]);
  if ( ranks < 1) ranks = 1;

  int tot_SM = 0;
  cudaRet = cudaDeviceGetAttribute(&tot_SM, cudaDevAttrMultiProcessorCount, 0);
  gpuErrchk("cudaDeviceGetAttribute", cudaRet);
  printf("running %d ranks on device 0, with %d SMs\n", ranks, tot_SM);


  loopKernel<<<ranks, 1>>>(ranks);
  cudaRet = cudaDeviceSynchronize();
  gpuErrchk("kernel", cudaRet);

}

