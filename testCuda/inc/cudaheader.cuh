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
#endif