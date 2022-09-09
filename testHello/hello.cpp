/*
Compile this code:
    g++ -o hello_cpu.exe \
      -I/usr/local/cuda/targets/x86_64-linux/include hello.cpp \
      -L/usr/local/cuda/targets/x86_64-linux/lib -lnvToolsExt -lm
Run the code:
    ./hello_cpu.exe
Profile the code:
    nsys profile -o profile_cpu --stat=true ./hello_cpu.exe

Profiling it, you can use:
    time (linux command)
    std::chrono (in C++), or clock() function, etc. (in C)
    (not required, profiling tools) Visual studio profiler, vtune, nsys 
*/

#include <iostream>
#include <cmath>
#include "nvToolsExt.h"


int main()
{
  int n = 1048576;  int check_idx = n - 10;
  float *x_h, *y_h; float *x_d, *y_d;
  
  // Allocate data
  x_h = (float*)malloc(n*sizeof(float));
  y_h = (float*)malloc(n*sizeof(float));

nvtxRangePush("Init data");
  // Initialize data
  for ( int i = 0; i < n; i++ )
  { // i^4+i+1
    x_h[i] = ( (i%251)*(i%251)*(i%251)*(i%251) ) + (i%251)+1;
    x_h[i] = fmod(x_h[i], 997.3);
    y_h[i] = -1;
  }
  printf("x[%d] = %f\n", check_idx, x_h[check_idx]);
nvtxRangePop();

nvtxRangePush("SAXPY CPU");
  // Do CPU works
  for ( int i = 0; i < n; i++ )
  {
    y_h[i] = 2.0*x_h[i] + y_h[i];
  }
nvtxRangePop();

  // Check the output
  printf("Result y[%d] = %f\n", check_idx, y_h[check_idx]);
  if ( 2*x_h[check_idx] - 1 == y_h[check_idx] )
    printf("Match! y=2x-1\n");
  else
    printf("Wrong, should be %f\n", 2*x_h[check_idx] + 1);

  // Release all resource
  free(x_h);
  free(y_h);
  
  return 0;
}


