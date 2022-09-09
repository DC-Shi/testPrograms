# Compiler behaviours on LOOP 

We've found several cases that the order of computation affects speed (OpenACC or CUDA) although we are sure that those computation can be perfectly paralleled and not blocked by data access.

# Experiments

The experiment will use 3D arrays A,B,C. Every array is N * N * N
1. Initialize A and B with specific value, and `C = 0`.
2. Compute the value `C = A + B`, for each element in every index.
    Two methods used for comparison: i,j,k and k,j,i
3. Check the value for C, the compared value is directly computed from index.

Since we have two index looping method, and we also compare two accelerating method: OpenACC and CUDA Fortran

Hardware:
- Server: DGX-A100(Use 1 GPU for testing)
- Software: pgfortran 22.2-0 in HPC SDK 22.2


# Result

Init the array:

Init Time(ms)  | i,j,k | k,j,i 
-----|--|---
OpenACC | 62.463 | 1.717
CUDA | 145.390 | 156.005


Computation: Add one by one

Add Time(ms)  | i,j,k | k,j,i 
-----|--|---
OpenACC | 33.163 | 1.870
CUDA | 10.214 | 1.837

# Conclusion
Since Fortran stored data in column-major order, and the array `A(i,j,k)` will store `A(9,100,10)` next to `A(10,100,10)`, thus looping `i` first will give a better performance.

For `i,j,k` order, the compiler recognized the `DO` loop, but it will keep the order of loop(`k` the inner, `j` in middle, `i` in outer), hence the data access is not coalesced.

Modify to `k,j,i` order, is a more efficient way, thus you'll see the computation keeps nearly the same time for both OpenACC and Fortran.

The SASS between these 4 cases are not the same as well. The OpenACC shows more instruction than Fortran