#include <stdio.h>

// error checking macro
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


const int DSIZE = 4;
const int block_size = 2;  // CUDA maximum is 1024

// display kernel, print the two array
__global__ void vDisp(const float *A, const float *B, int ds)
{
  int idx = blockIdx.x * block_size + threadIdx.x; // create typical 1D thread index from built-in variables
  printf("idx = %d, ds = %d\n", idx, ds);
  if (idx < ds)
    printf("Device: [%d], \t%f\t%f \n", idx, A[idx], B[idx]);         // do the vector (element) add here
}

int main()
{
  // This is for read/write pointer, need to be released.
  float *h_A, *d_A;
  // This is for read, so no need to release.
  float *h_read, *d_read;

  // Allocate space on CPU
  h_A = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++)
  {
    // Initialize with random.
    h_A[i] = rand()/(float)RAND_MAX;
  }

  cudaMalloc(&d_A, DSIZE*sizeof(float));  // allocate device space for vector A
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // Set the read pointer to proper value
  d_read = &d_A[1];
  h_read = &h_A[1];

  // Copy vector to GPU:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // run a loop ask for input address
  long addr = 0;
  do
  {
    // Print the address of each variable
    printf("h_A = %p, d_A = %p, h_read = %p, d_read = %p \n", (void *)h_A, (void *)d_A, (void *)h_read, (void *)d_read);

    // Run the display to show the array d_A
    vDisp<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_read, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Enter another the address !
    printf("Try type another hex value!: \n");
    
  } while(scanf("%p", &d_read));


  // copy vector back from device to host:
  //cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  // No cuda free??
  cudaFree(d_A);
  free(h_A);

  return 0;
}
  

