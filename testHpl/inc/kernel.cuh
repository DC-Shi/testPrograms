#ifndef DCS_HPL_GPU_FUNC
#define DCS_HPL_GPU_FUNC

#ifndef CUDACHECKERROR
#define CUDACHECKERROR
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
      exit(0); \
    }                                                                 \
   }
#endif

__global__ void calc();
__global__ void calcdouble();

__global__ void calcPoly6();
__global__ void calcPoly16();

#endif
