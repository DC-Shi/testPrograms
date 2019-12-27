#include <stdio.h>

__global__ void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main(int argc, char* argv[])
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    for ( int i = 0; i < argc; ++i )
       printf("param[%d] = %s\n", i, argv[i]);
    return 0;
}
