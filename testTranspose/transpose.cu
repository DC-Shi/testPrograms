#include <stdio.h>
#include <stdlib.h>

  //__global__ void transpose(int* input, int* output, size_t m, size_t n)
__global__ void transpose(int* input, int* output, size_t m, size_t n)
{
  size_t row = threadIdx.x + blockIdx.x * blockDim.x;
  size_t col = threadIdx.y + blockIdx.y * blockDim.y;

  // single element transpose.
  if ( row < m && col < n )
  {
    output[row*n + col] = input[col*m + row];
  }
}

// Reduce 32GB L2 visit to 4GB, 38ms->15ms
__global__ void transpose32(int* input, int* output, size_t m, size_t n)
{
  size_t bigX = blockIdx.x * blockDim.x;
  size_t smallx = threadIdx.x;
  size_t inputRow = bigX + smallx;
  size_t bigY = blockIdx.y * blockDim.y;
  size_t smally = threadIdx.y;
  size_t inputCol = bigY + smally;

  // target: (x,y) -> (y,x)
  // howto: (X+x, Y+y) -> (Y+y, X+x)
  // each block is (X,Y) -> (Y,X)
  // inside one block: (x,y) -> (y,x)

  __shared__ int shareData[32][32];
  // single element transpose.
  if ( inputRow < m && inputCol < n )
  {
    //shareData[smally][smallx] = input[X+x][Y+y]
    shareData[smally][smallx] = input[(bigX+smallx)*n + bigY+smally];
    __syncthreads();
    //output[Y+y][X+x] = shareData[smally][smallx]
    output[(bigY+smally)*m + bigX+smallx] = shareData[smally][smallx];
    // Latter one make things worse, writing is not coalasced.
    //output[(bigY+smallx)*m + bigX+smally] = shareData[smallx][smally];
  }
}

void printMatPart(int* mat_in, int* mat_out, size_t totalRows, size_t totalCols)
{
  size_t row = 42,col = 56;
  //printf("[42][56]: %d, [56,42]: %d\n", input[row*totalCols+col], output[row+totalRows*col]);

  for (size_t i = 0; i < 4; i++)
  {
    for (size_t j = 0; j < 3; j++)
      printf("%3d", mat_in[(row+i)*totalCols + col+j]);
    printf("\n");
  }

  printf("-----------------\n");
  for (size_t j = 0; j < 3; j++)
  {
    for (size_t i = 0; i < 4; i++)
      printf("%3d", mat_out[(col+j)*totalRows + row+i]);
    printf("\n");
  }
}

int main()
{
  // Random or fixed sequence
  //srand (time(NULL));
  srand(42);
  // Use GPU #2
  cudaSetDevice(2);


  size_t totalRows = 32768;
  size_t totalCols = 32768;
  size_t N = totalCols*totalRows; // large matrix
  int *input, *output;
  
  // Allocate
  cudaMallocManaged(&input, N*sizeof(int));
  cudaMallocManaged(&output, N*sizeof(int));

  for (size_t i = 0; i < totalRows; i++)
  {
    int rnd = rand() % 100;
    for (size_t j = 0; j < totalCols; j++)
      input[i*totalCols + j] = rnd;
  }
  cudaMemPrefetchAsync(input, N*sizeof(int), 2);
  cudaMemPrefetchAsync(output, N*sizeof(int), 2);

  
  dim3 block3(32,32,1);
  dim3 numBlocks3((totalRows + 31) / 32, (totalCols + 31) / 32,1);
#ifdef NAIVE
  transpose<<<numBlocks3, block3>>>(input, output, totalRows, totalCols);
#endif

#ifdef OPTIMISE32
  transpose32<<<numBlocks3, block3>>>(input, output, totalRows, totalCols);
#endif
  cudaDeviceSynchronize();

  printMatPart(input, output, totalRows, totalCols);
 
  // Free
  cudaFree(input); cudaFree(output);
  
  return 0;
}

