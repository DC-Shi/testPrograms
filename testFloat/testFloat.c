/*
This is to confirm float cannot represent large int numbers.
Originally I want to verify FP16, but it's not directly supported.
FP32 structure:
s|eeeeeeee|mmmmmmmm mmmmmmmm mmmmmmm

min: 2^-23 * 2^-127

so to represent 2^24+1 = 16777217, it has to be (2^-1+2^-25)*2^25
(2^-1+2^-24)*2^24 = 2^23 + 1 = 8388609

I don't think float can represent 16777217, let's show this!
*/

#include <stdio.h>  // printf
#include <stdlib.h> // rand, malloc, free
#include <time.h>   // clock
#include <string.h> // strcat
#include <stdint.h> // uint32_t

// https://stackoverflow.com/questions/1941307/c-debug-print-macros
#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif


const char *FloatToBinArray(uint32_t x)
{
DEBUG_PRINT(("bin..."));
  static char b[40];
  b[0] = 0;

  uint32_t mask = 0x80000000;
  int shift = 0;
  // mask>>32 == mask, like loops: 8->4->2->1->0x80000000
  for ( ; shift < 32; shift++ )
  {
    strcat(b, ((x & (mask>>shift)) == (mask>>shift) ) ? "1" : "0" );
    // Add space for seperating sign-bit, exp-bit, reminder-bits
    if ( shift == 0 || shift == 8 )
      strcat(b, "|");
    if ( shift == 16 || shift == 24 )
      strcat(b, " ");
DEBUG_PRINT( ("looping... shift=%d, %u & %u = %u\n", shift, x, (mask>>shift), x & (mask>>shift)) );
  }

  return b;
}

void PrintFloatAndHex(float f)
{
DEBUG_PRINT( ("printing...") );
/*
  union
  {
	float flt;
	uint32_t int32;
  } conv;

  conv.flt = f;
*/
  uint32_t int32 = * (uint32_t *) &f;
  char sign = (int32>>31 == 0) ? '+' : '-';
  uint32_t exp = (int32 & 0x7F800000)>>23;
  uint32_t frac = (int32 & 0x007FFFFF);
  printf("%s : %c2^(%u-127)*(1+%.23f) = %f \n", FloatToBinArray(int32), sign, exp, frac/8388608.0, f);
}


int main()
{
  float f23 = 8388608;
// 0.5 resolution
  PrintFloatAndHex(f23-0.5);
  PrintFloatAndHex(f23);
  PrintFloatAndHex(f23+1);
// 1 resolution
  PrintFloatAndHex(f23+0.5);
  PrintFloatAndHex(f23+1);


  float f24 = 16777216.0f;
  PrintFloatAndHex(f24-2);
  PrintFloatAndHex(f24-1);
  PrintFloatAndHex(f24);
// Below would output 16777216.0, since it cannot distinguish.
  PrintFloatAndHex(f24+1);
  PrintFloatAndHex(f24+2);

  printf("Type one float: ");
  float f;
  while (scanf("%f", &f) == 1)
  {
    PrintFloatAndHex(f);
    printf("Next float: ");
  }

  return 0;
}
