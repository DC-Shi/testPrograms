#include <stdio.h>
#define N 1024
int main() {
	int i, a[N], b[N], c[N];
	for (int i = 0; i < N; i++)
	{
		a[i] = 0;
		b[i] = c[i] = i;
	}
#pragma acc kernels
	{
		for (i = 0;i < N;i++) {
			a[i] = b [i]+ c[i];
		}
		for ( i = 0; i < N/2; i++)
		{
			b[i] = 2 * a[i];
		}
		printf("b[N/2]=%d\N",b[N/2]);
	}
		return 0;
}
