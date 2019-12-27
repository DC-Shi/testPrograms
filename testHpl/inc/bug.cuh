#ifndef DCS_HPL_GPU_BUG
#define DCS_HPL_GPU_BUG

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

#ifdef __cplusplus
  extern "C"
#endif
void bugTester();

#endif
