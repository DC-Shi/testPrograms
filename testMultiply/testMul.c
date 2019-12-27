/*
This code is to test whether multiply by 0 is faster than a non-zero number.

Compiler might optimize this.
*/
#include <stdio.h>  // printf
#include <stdlib.h> // rand, malloc, free
#include <time.h>   // clock
#include <string.h> // memset

// How many elements.
// 5 array used, each is float 4bytes, so 20x of memory of count
// 256 -> 5GB, 1->20M
const int ARRSIZE = 128*1024*1024;


// https://stackoverflow.com/questions/1789807/function-pointer-as-an-argument
typedef void (*FUNC_PTR)(float*, float*, float*);
// https://stackoverflow.com/questions/459691/best-timing-method-in-c
// Timing function
void timing(FUNC_PTR calc, float* a, float* b, float* c)
{
  clock_t start = clock(), diff;
  calc(a, b, c);
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
}

// Function to be used.
void normalMul(float* a, float* b, float* c)
{
  int i = 0;
  for ( i = 0; i < ARRSIZE; i++ )
    c[i] = a[i] * b[i];
}

// Preview some elements in array
void Preview(float* a, int length, char* desc)
{
  printf("%s[%d] = %f, ", desc, 0, a[0]);
  printf("%s[%d] = %f, ", desc, length/2, a[length/2]);
  printf("%s[%d] = %f\n", desc, length-1, a[length-1]);
}

int main()
{
  float *a,*b, *c, *b_zero, *b_3;
  int i = 0;
  // Set random seed from time
  int seed = time(NULL);
  srand(seed);

  a = malloc(ARRSIZE*sizeof(float));
  b = malloc(ARRSIZE*sizeof(float));
  c = malloc(ARRSIZE*sizeof(float));
  b_zero = malloc(ARRSIZE*sizeof(float));
  b_3 = malloc(ARRSIZE*sizeof(float));

  // Set the constant number
  memset( b_zero, 0, ARRSIZE*sizeof(float) );
  memset( b_3, 1.0/3, ARRSIZE*sizeof(float) );


  for ( i = 0; i < ARRSIZE; i++ )
  {
    a[i] = (float)rand()/(float)(RAND_MAX);
    b[i] = (float)rand()/(float)(RAND_MAX);
  }

  Preview(a, ARRSIZE, "a");

  printf("b * 0 = c ");
  timing(&normalMul, b, b_zero, c);
  printf("0 * a = c ");
  timing(&normalMul, b_zero, a, c);

  printf("a * b = c ");
  timing(&normalMul, a, b, c);
  printf("a * 0 = c ");
  timing(&normalMul, a, b_zero, c);
  printf("a *1/3= c ");
  timing(&normalMul, a, b_3, c);
  printf("0 * a = c ");
  timing(&normalMul, b_zero, a, c);


  free(a);
  free(b);
  free(c);
  free(b_zero);
  free(b_3);


  return 0;
}

