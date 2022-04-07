// Test run kernel in kernel

#include <stdio.h>
#include "cudaheader.cuh"

// Code is copied from Robert Crovella on stackoverflow:
// https://stackoverflow.com/questions/30361459/what-is-the-behavior-of-thread-block-scheduling-to-specific-sms-after-cuda-kern
// Modified a little to match A100

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define NB 20
// increase array length here if your GPU has more than 32 SMs
#define MAX_SM 192
// set HANG_TEST to 1 to demonstrate a hang for test purposes
#define HANG_TEST 0

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

static __device__ __inline__ uint32_t __smid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;}

__device__ volatile int blocks_completed = 0;
// increase array length here if your GPU has more than 32 SMs
__device__ int first_SM[MAX_SM];
__device__ int device_SM[NB+MAX_SM];
int host_SM[NB+MAX_SM];

// launch with one thread per block only
__global__ void tkernel(int num_blocks, int num_SMs){

  int my_SM = __smid();
  int idx = blockIdx.x;
  if (idx > NB - 10)
    printf("Starting thread[%d] on sm[%d]\n", idx, my_SM);
  int im_not_first = atomicCAS(first_SM+my_SM, 0, 1);
  if (!im_not_first){
    while (blocks_completed < (num_blocks-num_SMs+HANG_TEST));
#ifdef SLEEP
    while(true)
    {
      if ( device_SM[idx] > 12345678 ) 
      {
        atomicSub((int*)&device_SM[idx], 12345679);
        if (idx == 0)
        {
          //int sum = 0;
          //for (int tmpi = 0; tmpi < NB; tmpi++)
          //{
          //  if (device_SM[tmpi] > 0)
          //    sum++;
          //}
          //printf("[0] %d non-zero elements in device_SM array\n", sum);
          //printf("[0] device_SM[1300] = %d\n", device_SM[1300]);
          //printf("List of zeros: ");
          for (int tmpi = 0; tmpi < NB; tmpi++)
          {
            printf("%3d, ", device_SM[tmpi] % 1000);
            if (tmpi % 30 == 29)
            printf("\n");
          }
          printf("\n");
        }
        //break;
      }
      else
      {
        atomicAdd((int *)&device_SM[idx], 1);
      }
    }
#endif
  }
  // This would leads to the array element to be non-zero, why?
  // Compiler optimized?
  //device_SM[idx] = my_SM;
  //atomicExch(device_SM+idx, my_SM);
  
  atomicAdd((int *)&blocks_completed, 1);
}

int main(int argc, char *argv[]){
  unsigned my_dev = 0;
  if (argc > 1) my_dev = atoi(argv[1]);
  cudaSetDevice(my_dev);
  cudaCheckErrors("invalid CUDA device");
  int tot_SM = 0;
  cudaDeviceGetAttribute(&tot_SM, cudaDevAttrMultiProcessorCount, my_dev);
  cudaCheckErrors("CUDA error");
  if (tot_SM > MAX_SM) {printf("program configuration error\n"); return 1;}
  printf("running on device %d, with %d SMs\n", my_dev, tot_SM);
  int temp[MAX_SM];
  for (int i = 0; i < MAX_SM; i++) temp[i] = 0;
  cudaMemcpyToSymbol(first_SM, temp, MAX_SM*sizeof(int));
  cudaMemcpyToSymbol(device_SM, temp, MAX_SM*sizeof(int));
  cudaCheckErrors("cudaMemcpyToSymbol fail");
  tkernel<<<NB, 1>>>(NB, tot_SM);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel error");
  
  cudaMemcpyFromSymbol(host_SM, device_SM, NB*sizeof(int));
  cudaCheckErrors("cudaMemcpy fail");

  int width = 32;
  // Find best width
  for ( width = 20; width < 48; width += 2)
  {
    if (  tot_SM % width == 0)
    {
      printf("Change column width to %d.\n", width);
      break;
    }
  }
  for ( int i = 0; i < NB; i++ )
  {
    printf("%03d ", host_SM[i]);
    if ( i % width == width - 1 ) printf("\n");
  }
  printf("\n");
}

