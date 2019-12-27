#include <stdio.h>
#include "bug.cuh"

/// Occurred once for such form, reducd to a8 level and less x*x*x would be work
// But cannot reproduce.
__device__ void too_many_resources_requested_for_launch(int* result,
    int a0 ,
    int a1 ,
    int a2 ,
    int a3 ,
    int a4 ,
    int a5 ,
    int a6 ,
    int a7 ,
    int a8 ,
    int a9 ,
    int a10,
    int a11,
    int a12,
    int a13,
    int a14,
    int a15,
    int x,
    int modulus
)
{
    *result = (
        a0 + x*(
            a1 + x*(
                a2 + x*(
                    a3 + x*(
                        a4 + x*(
                            a5 + x*(
                                a6 + x*(
                                    a7 + x*(
                                        a8 + x*(
                                            0 + x*(
                                                0 + x*(
                                                    0 + x*(
                                                        0 + x*(
                                                            0 + x*(
                                                                0 + x*(
                                                                    0 + x*(1) % modulus
                                                                ) % modulus
                                                            ) % modulus
                                                        ) % modulus
                                                    ) % modulus
                                                ) % modulus
                                            ) % modulus
                                        ) % modulus
                                    ) % modulus
                                ) % modulus
                            ) % modulus
                        ) % modulus
                    ) % modulus
                ) % modulus
            ) % modulus
        ) % modulus
    ) % modulus;
}

__device__ void bugZ(int* result,
    int a0 ,
    int a1 ,
    int a2 ,
    int a3 ,
    int a4 ,
    int a5 ,
    int a6 ,
    int a7 ,
    int a8 ,
    int a9 ,
    int a10,
    int a11,
    int a12,
    int a13,
    int a14,
    int a15,
    int x,
    int modulus
)
{
    *result = (
        a0 + x*(
            a1 + x*(
                a2 + x*(
                    a3 + x*(
                        a4 + x*(
                            a5 + x*(
                                a6 + x*(
                                    a7 + x*(
                                        a8 + x*(
                                            a9 + x*(
                                                a10 + x*(
                                                    a11 + x*(
                                                        a12 + x*(
                                                            a13 + x*(
                                                                a14 + x*(
                                                                    a15 + x*(1) % modulus
                                                                ) % modulus
                                                            ) % modulus
                                                        ) % modulus
                                                    ) % modulus
                                                ) % modulus
                                            ) % modulus
                                        ) % modulus
                                    ) % modulus
                                ) % modulus
                            ) % modulus
                        ) % modulus
                    ) % modulus
                ) % modulus
            ) % modulus
        ) % modulus
    ) % modulus;
}

// blockDim: 16*16*4
// gridDim: 2147483647, 65535, 65535: 16^7, 16^3, 16^3
// 420ms for 256*4096*16*16*4 8+12+4+4+2=30
__global__ void bugArray()
{
    int f_base = threadIdx.z; // is 0-4
    int a1  = threadIdx.y;
    int a2  = threadIdx.x;
 
    int a3  =  blockIdx.z       & 0xF;
    int a4  = (blockIdx.z >> 4) & 0xF;
    int a5  = (blockIdx.z >> 8) & 0xF;
 
    int a6  =  blockIdx.y       & 0xF;
    int a7  = (blockIdx.y >> 4) & 0xF;
    int a8  = (blockIdx.y >> 8) & 0xF;
 
    int a9  =   blockIdx.x        & 0xF;
    int a10 = (blockIdx.x >> 4)  & 0xF;
    int a11 = (blockIdx.x >> 8)  & 0xF;
    int a12 = (blockIdx.x >> 12) & 0xF;
    int a13 = (blockIdx.x >> 16) & 0xF;
    int a14 = (blockIdx.x >> 20) & 0xF;
    int a15 = (blockIdx.x >> 24) & 0xF;

    int MOD = 17; // Compiler seems automatically optimized % 16 to & 0xF

    int a0 = 4*f_base;
    
    // ??? Does this array the same across all threads??
    int Y[10] = {0,0,0,0,0,0,0,0,0,0};
    // Make sure distinct values
    for(int x = 0; x < 8; x++)
    {
        int result = 0;
        too_many_resources_requested_for_launch(&result,
            a0, a1, a2, a3,
            a4, a5, a6, a7,
            a8, a9, a10, a11,
            a12, a13, a14, a15,
            x, MOD);
        int result0 = (result + 0) % MOD;
        int result1 = (result + 1) % MOD;
        int result2 = (result + 2) % MOD;
        int result3 = (result + 3) % MOD;

        //Y[0] |= (1 << result0);
        Y[1] |= (1 << result1);
        //Y[2] |= (1 << result2);
        //Y[3] |= (1 << result3);
    }


    for (int idx_fg=0;idx_fg<4;idx_fg++)
    {
        //                 FEDCBA9876543210
        if (Y[idx_fg] == 0b0011001101010011)
        {
            // Let's check it whether in back row
            for(int x = 8; x < 16; x++)
            {
                int result = 0;
                bugZ(&result,
                    a0+idx_fg, a1, a2, a3,
                    a4, a5, a6, a7,
                    a8, a9, a10, a11,
                    a12, a13, a14, a15,
                    x, MOD);
                int result_b = (result) % MOD;
        
                Y[4+idx_fg] |= (1 << result_b);
            }

            //                    FEDCBA9876543210
            if ( Y[4+idx_fg] == 0b1100110010101100)
            {
                int res[16];
                for(int tmpi = 0; tmpi<16;tmpi++)
                {
                    bugZ(&res[tmpi],
                        a0+idx_fg, a1, a2, a3,
                        a4, a5, a6, a7,
                        a8, a9, a10, a11,
                        a12, a13, a14, a15,
                        tmpi, MOD);
                }
                printf("a=[%2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d,], res=[%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,] Y=[%d %d %d %d %d %d %d %d %d %d]\n",
                        a0+idx_fg, a1, a2, a3,
                        a4, a5, a6, a7,
                        a8, a9, a10, a11,
                        a12, a13, a14, a15,
                    res[0], res[1], res[2], res[3],
                    res[4], res[5], res[6], res[7],
                    res[8], res[9], res[10], res[11],
                    res[12], res[13], res[14], res[15],
                Y[0], Y[1], Y[2], Y[3], Y[4],
                Y[5], Y[6], Y[7], Y[8], Y[9]);
            }
            
        }

    }

}

__global__ void testPrint()
{
    int testY[20] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
    int res[16]={0,1,2,3,4};

    printf("mnopqr16=[%1d,%2d,%3d, %4d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d,], res16=[%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,%2d,] non-meaning16=[%p%p%p%p%p%p%p%p]\n",
res[0], res[1], res[2], res[3],
res[4], res[5], res[6], res[7],
res[8], res[9], res[10], res[11],
res[12], res[13], res[14], 1048576, res[15],
            testY[0], testY[1], testY[2], testY[3], testY[4],
            testY[5], testY[6], testY[7], testY[8], testY[9]);
    
}


void bugTester()
{
    dim3 dimGrid(16,16,16), dimBlock1(16,16,1), dimBlock4(16,16,4);
    //calcPoly6<<<dimGrid,dimBlock4>>>();
    dim3 varGrid(1,16*256,256*16);

    
    //bugArray<<<varGrid,dimBlock4>>>();
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    testPrint<<<1,1>>>();
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}