#ifndef GPUASSERT
#define GPUASSERT

#define gpuErrchk(str, ans) { gpuAssert((str), (ans), __FILE__, __LINE__); }
inline void gpuAssert(const char *prompt, cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUASSERT %s Failed: %s %s %d\n", prompt, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
   else
       printf("success : %s\n", prompt);
}

// Used for C++ functions
#ifdef __cplusplus
#include <iostream>
#define gpuErrchkCpp20(str, ans) { gpuAssertCpp20((str), (ans), __FILE__, __LINE__); }
inline void gpuAssertCpp20(const char *prompt, cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUASSERT "<< prompt << " Failed:" << cudaGetErrorString(code) << file << line << std::endl;;
      if (abort) exit(code);
   }
   else
   // Only print success if prompt is not empty
      if ( prompt[0] != 0 || prompt != nullptr || prompt != NULL)
         std::cout << "success : " << prompt << std::endl;
}
#endif

#endif