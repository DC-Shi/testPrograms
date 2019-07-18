#include <stdio.h>
#include "kernel.cuh"

__global__ void sample()
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    int c = threadIdx.x;
    double x = 1;
    
    double result = pow(0.0,x)+a+b*x+c*pow(x,2.0);
    
    if(result == 10)
    printf("a=%d, b=%d, c=%d\n", a,b,c);
}

__global__ void calc()
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    int c = blockIdx.z;
    int d = threadIdx.x;
    int e = threadIdx.y;
    int f_base = threadIdx.z; // is 0-4

    int MOD = 17;
    
    for (int f = 4*f_base; f < 4*f_base+4; f++)
    {
        int Y[20] = {0};
        // Make sure distinct values
        for(int x = 0; x < 8; x++)
        {
            int result = (x==0?1:0)+a+x*(b+x*(c+x*(d+x*(e+x*f % MOD) % MOD) % MOD) % MOD);
            result = result % MOD;
            // We suppose the result should be in 0~16 range, and it should be not shown yet.
            if (Y[result] != 0)
                break;

            // I don't know what changed, but now it can reach 8~9ms for integer-only kernel.
            // 16^5*4 = 2^22
            Y[result] = result==0?1:result;
        }
    
        // Now we have 8 distinct values, check each value
        //if (Y[0]==1 && Y[1]==1 && Y[4]==1 && Y[6]==1 && Y[8]==1 && Y[9]==1 && Y[12]==1 && Y[13]==1)
        if (Y[0] && Y[1] && Y[4] && Y[6] && Y[8] && Y[9] && Y[12] && Y[13])
        {
            // Below method can print inter-cross.
            //for (int i=0;i<20;i++) printf("%d ", Y[i]);
            //printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d\n", a,b,c,d,e,f);
            printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d, Y=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ]\n",
            a,b,c,d,e,f,
    Y[0], Y[1], Y[2], Y[3], Y[4],
    Y[5], Y[6], Y[7], Y[8], Y[9],
    Y[10], Y[11], Y[12], Y[13], Y[14],
    Y[15], Y[16], Y[17], Y[18], Y[19]);
        }

    }
}


__global__ void calcdouble()
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    int c = blockIdx.z;
    int d = threadIdx.x;
    int e = threadIdx.y;
    int f_base = threadIdx.z; // is 0-4

    double MOD = 17.0;
    
    for (int f = 1; f < 16; f++)
    {
        int Y[20] = {0};
        // Make sure distinct values
        for(double x = 0; x < 8; x++)
        {
            int result = (int)fmod(pow(0.0,x)+a+b*x+c*pow(x,2)+d*pow(x,3)+e*pow(x,4)+f*pow(x,5), MOD);
            // We suppose the result should be in 0~16 range, and it should be not shown yet.
            if (Y[result] != 0)
                break;

            // method1: plus only, 258ms
            //Y[result]++;

            // method2: plus when 0, and others show its number, 254ms
            // somehow it changed to ~15ms, maybe I changed print? --- no
            // 
            Y[result] = result==0?1:result;
        }
    
        // Now we have 8 distinct values, check each value
        if (Y[0] && Y[1] && Y[4] && Y[6] && Y[8] && Y[9] && Y[12] && Y[13])
        {
            // Below method can print inter-cross.
            //for (int i=0;i<20;i++) printf("%d ", Y[i]);
            //printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d\n", a,b,c,d,e,f);
            printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d, Y=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ]\n",
            a,b,c,d,e,f,
    Y[0], Y[1], Y[2], Y[3], Y[4],
    Y[5], Y[6], Y[7], Y[8], Y[9],
    Y[10], Y[11], Y[12], Y[13], Y[14],
    Y[15], Y[16], Y[17], Y[18], Y[19]);
        }

    }
}

// Find all power-6 polynomial
// y = x^8 + 13x^7 + a6 x^6 + a5 x^5 + a4 x^4 + a3 x^3 + a2 x^2 + a1 x
// quite fast, <10ms for 16*16*16*16*16*4=2^22
__global__ void calcPoly6()
{
    int a1 = blockIdx.x;
    int a2 = blockIdx.y;
    int a3 = blockIdx.z;
    int a4 = threadIdx.x;
    int a5 = threadIdx.y;
    int f_base = threadIdx.z; // is 0-4

    int MOD = 17;
    
    for (int a6 = 4*f_base; a6 < 4*f_base+4; a6++)
    {
        int Y[20] = {0};
        // Make sure distinct values
        for(int x = 0; x < 8; x++)
        {
            int result = x*(
                a1 + x*(
                    a2 + x*(
                        a3 + x*(
                            a4 + x*(
                                a5 + x*(
                                    a6 + x*(
                                        13 + x % MOD
                                    ) % MOD
                                ) % MOD
                            ) % MOD
                        ) % MOD
                    ) % MOD
                ) % MOD
            ) % MOD;
            result = result % MOD;
            // We suppose the result should be in 0~16 range, and it should be not shown yet.
            if (Y[result] != 0)
                break;

            // I don't know what changed, but now it can reach 8~9ms for integer-only kernel.
            Y[result] = result==0?1:result;
        }
    
        // Now we have 8 distinct values, check each value
        //if (Y[0]==1 && Y[1]==1 && Y[4]==1 && Y[6]==1 && Y[8]==1 && Y[9]==1 && Y[12]==1 && Y[13]==1)
        if (Y[0] && Y[1] && Y[4] && Y[6] && Y[8] && Y[9] && Y[12] && Y[13])
        {
            // Below method can print inter-cross.
            //for (int i=0;i<20;i++) printf("%d ", Y[i]);
            //printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d\n", a,b,c,d,e,f);
            printf("a1=%d, a2=%d, a3=%d, a4=%d, a5=%d, a6=%d, Y=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ]\n",
            a1,a2,a3,a4,a5,a6,
    Y[0], Y[1], Y[2], Y[3], Y[4],
    Y[5], Y[6], Y[7], Y[8], Y[9],
    Y[10], Y[11], Y[12], Y[13], Y[14],
    Y[15], Y[16], Y[17], Y[18], Y[19]);
        }

    }
}


__device__ void calculateZ(int* result,
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


// Find all polynomial
// y = a0
//   + a1 x^1
//   + a2 x^2
//   + a3 x^3
//   + a4 x^4
//   + a5 x^5
//   + a6 x^6
//   + a7 x^7
//   + a8 x^8
//   + a9 x^9
//   + a10 x^10
//   + a11 x^11
//   + a12 x^12
//   + a13 x^13
//   + a14 x^14
//   + a15 x^15

// blockDim: 16*16*4
// gridDim: 2147483647, 65535, 65535: 16^7, 16^3, 16^3
// 420ms for 256*4096*16*16*4 8+12+4+4+2=30
__global__ void calcPoly16()
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
    int Y[10] = {0};
    // Make sure distinct values
    for(int x = 0; x < 8; x++)
    {
        int result = 0;
        calculateZ(&result,
            a0, a1, a2, a3,
            a4, a5, a6, a7,
            a8, a9, a10, a11,
            a12, a13, a14, a15,
            x, MOD);
        int result0 = (result + 0) % MOD;
        int result1 = (result + 1) % MOD;
        int result2 = (result + 2) % MOD;
        int result3 = (result + 3) % MOD;

        Y[0] |= (1 << result0);
        Y[1] |= (1 << result1);
        Y[2] |= (1 << result2);
        Y[3] |= (1 << result3);
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
                calculateZ(&result,
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
                    calculateZ(&res[tmpi],
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
    // Now we have 8 distinct values, check each value
    //if (Y[0]==1 && Y[1]==1 && Y[4]==1 && Y[6]==1 && Y[8]==1 && Y[9]==1 && Y[12]==1 && Y[13]==1)
//    if (Y[0] && Y[1] && Y[4] && Y[6] && Y[8] && Y[9] && Y[12] && Y[13])
//    {
//            // Below method can print inter-cross.
//            //for (int i=0;i<20;i++) printf("%d ", Y[i]);
//            //printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d\n", a,b,c,d,e,f);
//            printf("a=[%2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d], Y=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ]\n",
//            a0, a1, a2, a3,
//            a4, a5, a6, a7,
//            a8, a9, a10, a11,
//            a12, a13, a14, a15,
//
//    Y[0], Y[1], Y[2], Y[3], Y[4],
//    Y[5], Y[6], Y[7], Y[8], Y[9],
//    Y[10], Y[11], Y[12], Y[13], Y[14],
//    Y[15], Y[16], Y[17], Y[18], Y[19]);
//Y[18]++;
//    }

}


