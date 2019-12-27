#include <stdio.h>
#include <math.h>

float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

typedef union{
   unsigned int i;
   float f;
}Data; 

void printD(Data a)
{
  printf("%f(0x%X) ", a.f, a.i);
}

void round(Data *ret)
{
  // 1+8+23 -> 1+5+10
  // trim fraction to 10 bits.
  ret->i = ret->i & 0xFFFFE000;
//  printD(ret);
  // trim exponent to 5 bits. -127~128 to -15~16
  // get exp
  int exp = ((ret->i & 0x7F800000)>>23)-127;
//  printf("exp=%d ", exp);
  // clear exp
  ret->i = ret->i & 0x807FFFFF;
//  printD(ret);
  // add exp(only 5 bits)
  exp = ((exp+15)&0x1F) + 112;
//  printf("exp=%d ", exp);
  ret->i = ret->i | (exp<<23);
//  printD(ret);

}

Data hmul(Data a, Data b)
{
  Data ret;
  ret.f = a.f * b.f;

//  printD(ret);
  round(&ret);

  return ret;
}

Data hsub(Data a, Data b)
{
  Data ret;
  ret.f = a.f - b.f;
  round(&ret);
  return ret;
}


int main(int argc, char** argv)
{
  float t = 0.43f;

  float res = Q_rsqrt(t);
  float real = 1.0f / sqrt(t);

  printf("Q_rsqrt(%f) = %f != %f\n", t, res, real);

  Data a, b, c, d;
  a.f = 0.43;
  b.f = 0.2;
  c = hmul(a, b);
  d = hsub(a, b);

  printD(a);
  printD(b);
  printD(c);
  printD(d);

  printf("\n");

  return 0;
}
