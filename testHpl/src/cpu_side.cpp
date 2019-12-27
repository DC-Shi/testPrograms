#include "cpu_side.h"
#include <stdio.h>


int l(int x)
{
    int ret = (x); ret = (ret + CpuModulus*CpuModulus) % CpuModulus;
    printf(" l(%d)=%d,", x, ret);
    return ret;
}
int k(int x)
{
    int ret = 2+(x-1)*l(x); ret = (ret + CpuModulus*CpuModulus) % CpuModulus;
    printf(" k(%d)=%d,", x, ret);
    return ret;
}
int h(int x)
{
    int ret = 4+(x-5)*k(x); ret = (ret + CpuModulus*CpuModulus) % CpuModulus;
    printf(" h(%d)=%d,", x, ret);
    return ret;
}
int g(int x)
{
    int ret = 2+(x-2)*(x-3)*(x-4)*(x-6)*h(x); ret = (ret + CpuModulus*CpuModulus) % CpuModulus;
    printf(" g(%d)=%d,", x, ret);
    return ret;
}
int f(int x)
{
    int ret = x*g(x); ret = (ret + CpuModulus*CpuModulus) % CpuModulus;
    printf(" f(%d)=%d\n", x, ret);
    return ret;
}


int f_expand_wrong(int x)
{
    int ret = x*(
        5 + x*(
            14 + x*(
                3 + x*(
                    4 + x*(
                        11 + x*(
                            7 + x*(
                                13 + x
                            ) % CpuModulus
                        ) % CpuModulus
                    ) % CpuModulus
                ) % CpuModulus
            ) % CpuModulus
        ) % CpuModulus
    ) % CpuModulus;
    printf(" f_expand_wrong(%d)=%d\n", x, ret);
    return ret;
}



int f_poly8(int x)
{
    int ret = (
        13 + x*(
            14 + x*(
                13 + x*(
                    0 + x*(
                        6 + x*(
                            0 + x*(
                                2 + x*(
                                    4 + x*(
                                        15 + (x*x*x*x%CpuModulus) * (x*x*x*x%CpuModulus)
                                    ) % CpuModulus
                                ) % CpuModulus
                            ) % CpuModulus
                        ) % CpuModulus
                    ) % CpuModulus
                ) % CpuModulus
            ) % CpuModulus
        ) % CpuModulus
    ) % CpuModulus;
    printf(" f_poly8(%d)=%d\n", x, ret);
    return ret;
}