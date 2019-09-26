// Reference from https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
// https://stackoverflow.com/questions/36953612/assignment-of-device-function-pointers-in-cuda-from-host-function-pointers

#include <iostream>
#include <stdio.h>
#include "cudaheader.cuh"
// Since C++ 11
//template<typename T>
//using func_t = T (*) (T, T);

typedef int (*funcptr) ();
typedef int (*func2ptr) (int, int);

__device__ int add_func (int x, int y) { return x + y; }
__device__ int mul_func (int x, int y) { return x * y; }
__device__ int ret_func (int x, int y) { return 42; }
__device__ int f () { return 21; }

__device__ funcptr  f_ptr   = f ;
__device__ func2ptr ret_ptr = ret_func ;
__device__ func2ptr add_ptr = add_func ;
__device__ func2ptr mul_ptr = mul_func ;



__global__ void kernel(funcptr op, int* d_x)
{
//    *result = (*op)(*d_x, *d_y);
    int k = op () ;
    printf ("%d\n", k) ;

    funcptr func2 = f ; // does not use a global-scope variable
    printf ("%d\n", func2()) ;
}


cudaError_t runTest1()
{
    funcptr h_funcptr ;
    int *d_a;
    
    cudaMallocManaged(&d_a, 3*sizeof(int));
    d_a[0] = 1;
    d_a[1] = 2;

    if (cudaSuccess != cudaMemcpyFromSymbol (&h_funcptr, f_ptr, sizeof (funcptr)))
        printf ("FAILED to get SYMBOL\n");

    kernel <<<1,1>>> (h_funcptr, d_a) ;

    cudaError_t ret = cudaDeviceSynchronize();
    if (ret == cudaSuccess)
    {
        printf("a = [%d, %d, %d]\n", d_a[0], d_a[1], d_a[2]);
    }

    cudaFree(d_a);

    return ret;
}

// A variable declaration like basic data types 
struct Point 
{ 
   int x, y, z;
   func2ptr PointFunc;
};  

__global__ void kernelPoint(Point *op)
{
    printf("op(%p): %d, %d, %d, %p\n", op, op->x, op->y, op->z, op->PointFunc);
//    *result = (*op)(*d_x, *d_y);
    int k = (op->PointFunc) (op->x, op->y) ;
    printf ("k = %d\n", k) ;

    func2ptr func2 = ret_func ; // does not use a global-scope variable
    printf ("func2(1,1) = %d\n", func2(1,1)) ;

    op->z = k;
}

cudaError_t runTest2()
{
    Point *d_p;
    
    cudaMallocManaged(&d_p, 3*sizeof(int)+sizeof(func2ptr));
    d_p->x = 5; d_p->y = 2; d_p->z = 9;
    d_p->PointFunc = &ret_func;
    func2ptr tmp ;

    printf("Before copy: tmp(%p), %p, d_p->PointFunc(%p), %p\n", &tmp, tmp, &(d_p->PointFunc), (d_p->PointFunc));
    cudaError_t abc = cudaMemcpyFromSymbol (&tmp, ret_ptr, sizeof (func2ptr)); // fail
    //cudaError_t abc = cudaMemcpyFromSymbol (&tmp, add_ptr, sizeof (func2ptr)); // success
    if (cudaSuccess != abc)
        printf ("FAILED to get SYMBOL, %d, %s\n", abc, cudaGetErrorName(abc));

    printf("After  copy: tmp(%p), %p, d_p->PointFunc(%p), %p\n", &tmp, tmp, &(d_p->PointFunc), (d_p->PointFunc));
    //d_p->PointFunc = (func2ptr)0x18;    // a+b
    //d_p->PointFunc = (func2ptr)0x20;    // a*b
    //d_p->PointFunc = (func2ptr)0x10;    // 42
    d_p->PointFunc = (func2ptr)0x10;
    //d_p->PointFunc = tmp;
    printf("After  set : tmp(%p), %p, d_p->PointFunc(%p), %p\n", &tmp, tmp, &(d_p->PointFunc), (d_p->PointFunc));

    kernelPoint <<<1,1>>> (d_p) ;

    cudaError_t ret = cudaDeviceSynchronize();
    if (ret == cudaSuccess)
    {
        printf("a = [%d, %d, %d]\n", d_p->x, d_p->y, d_p->z);
    }

    cudaFree(d_p);

    return ret;
}

__global__ void kernInit(Point *op)
{
    printf("op(%p): %p, %p, %p ===\n", op, &op[0].x, &op[0].y, &op[0].z);
    op[0].x = 3;
    op[0].y = 3;
    op[0].z = 1823;
    //p->PointFunc = &ret_func;
    
    printf("op_new(%p): %d, %d, %d\n", op, op[0].x, op[0].y, op[0].z);
    //printf("op(%p): %d, %d, %d, %p\n", op, op[0].x, op[0].y, op[0].z, op[0].PointFunc);
}
__device__ Point test3Obj[1];
cudaError_t runTest3()
{
    Point* arr_host;
    // We have to convert the pointer.
    gpuErrchk("load from symbol", cudaGetSymbolAddress((void**)&arr_host, test3Obj));

    printf("test3Obj(%p), arr_host(%p)\n", test3Obj, arr_host);
    kernInit<<<1,1>>>(arr_host);
    cudaError_t ret = cudaDeviceSynchronize();
    if (ret == cudaSuccess)
    {
        printf("goodinit\n");
    }
    else
    printf("bad init\n");

    func2ptr tmp ;

    printf("Before copy: tmp(%p), %p, test3Obj->PointFunc(%p), %p\n", &tmp, tmp, &(test3Obj->PointFunc), (test3Obj->PointFunc));
    /// Finally, it's not &tmp and &struct->pointfunc problem.
    /// The reason is we misused device pointer on host side.
    //  Remember: when transfer pointer between host and device, use cudaMemcpyFromSymbol
    //  data in CPU, pointer in CPU. Use as normal.
    //  data in GPU, pointer in CPU. cudaMemcpyFromSymbol
    //  data in GPU, pointer in GPU. Use as normal.
    //  data in CPU, pointer in GPU. not encountered this case
    cudaError_t abc = cudaMemcpyFromSymbol (&(arr_host->PointFunc), add_ptr, sizeof (func2ptr));
    if (cudaSuccess != abc)
        printf ("FAILED to get SYMBOL, %d, %s\n", abc, cudaGetErrorName(abc));

/*
f_ptr  
ret_ptr
add_ptr
mul_ptr
*/
    void* ptr_tmp, *sym_ptr;
    gpuErrchk("cudaGetSymbolAddress : ", cudaGetSymbolAddress((void**)&ptr_tmp, f_ptr));
    gpuErrchk("cudaMemcpyFromSymbol : ", cudaMemcpyFromSymbol (&sym_ptr, f_ptr, sizeof (funcptr)) );
    printf("ptr_tmp(%p) = %p, sym_ptr(%p) = %p,   f_ptr(%p) = %p\n", &ptr_tmp, ptr_tmp, &sym_ptr, sym_ptr, &f_ptr, f_ptr);
    
    gpuErrchk("cudaGetSymbolAddress : ", cudaGetSymbolAddress((void**)&ptr_tmp, ret_ptr));
    gpuErrchk("cudaMemcpyFromSymbol : ", cudaMemcpyFromSymbol (&sym_ptr, ret_ptr, sizeof (func2ptr)) );
    printf("ptr_tmp(%p) = %p, sym_ptr(%p) = %p, ret_ptr(%p) = %p\n", &ptr_tmp, ptr_tmp, &sym_ptr, sym_ptr, &ret_ptr, ret_ptr);
    
    gpuErrchk("cudaGetSymbolAddress : ", cudaGetSymbolAddress((void**)&ptr_tmp, add_ptr));
    gpuErrchk("cudaMemcpyFromSymbol : ", cudaMemcpyFromSymbol (&sym_ptr, add_ptr, sizeof (func2ptr)) );
    printf("ptr_tmp(%p) = %p, sym_ptr(%p) = %p, add_ptr(%p) = %p\n", &ptr_tmp, ptr_tmp, &sym_ptr, sym_ptr, &add_ptr, add_ptr);
    
    gpuErrchk("cudaGetSymbolAddress : ", cudaGetSymbolAddress((void**)&ptr_tmp, mul_ptr));
    gpuErrchk("cudaMemcpyFromSymbol : ", cudaMemcpyFromSymbol (&sym_ptr, mul_ptr, sizeof (func2ptr)) );
    printf("ptr_tmp(%p) = %p, sym_ptr(%p) = %p, mul_ptr(%p) = %p\n", &ptr_tmp, ptr_tmp, &sym_ptr, sym_ptr, &mul_ptr, mul_ptr);

    printf("After  copy: tmp(%p), %p, test3Obj->PointFunc(%p), %p\n", &tmp, tmp, &(test3Obj->PointFunc), (test3Obj->PointFunc));

    kernelPoint <<<1,1>>> (arr_host) ;

    ret = cudaDeviceSynchronize();
    if (ret == cudaSuccess)
    {
        printf("a = [%d, %d, %d]\n", test3Obj->x, test3Obj->y, test3Obj->z);
    }

    return ret;
}

int main()
{
    
    if (runTest1() != cudaSuccess)
        printf ("FAILED Test 1\n");
    else
        printf ("SUCCEEDED Test 1\n");

    printf("=====================================\n");
    
    if (runTest2() != cudaSuccess)
        printf ("FAILED Test 2\n");
    else
        printf ("SUCCEEDED Test 2\n");

    printf("=====================================\n");
    
    if (runTest3() != cudaSuccess)
        printf ("FAILED Test 3\n");
    else
        printf ("SUCCEEDED Test 3\n");
}
