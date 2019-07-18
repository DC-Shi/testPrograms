#ifndef DCS_HPL_CPU_FUNC
#define DCS_HPL_CPU_FUNC

// f(x) that can do mapping 0-7 to front row.
int f(int x);
const int CpuModulus = 17;
int f_expand_wrong(int x);

int f_poly8(int x);

#endif
