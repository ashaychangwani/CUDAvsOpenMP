#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

#include <omp.h>

#define MAX_N 16

int main(int argc, char* argv[])
{
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
	}
    int n=atoi(argv[1]);
    int npown = 1;
    for (int i = 0; i < n; i++)
      npown *= n;
        
    double start, end;
    int sols = 0;
    
    start = omp_get_wtime();
	int iter;

	#pragma omp target teams num_teams(128) map(tofrom:sols)
  {
		#pragma omp distribute
		for (iter = 0; iter < npown; iter++)
		{
			int current = iter;
			int i,j;
			int board = 0;
			for (i = 0; i < n; i++)
			{
                board *= 10;
				board += current % n;
				current /= n;
			}
			int sum = 0;
			int iPower = 1;
			int jPower = 1;
			#pragma distribute parallel for reduction(max:sum) collapse(2)
			for (i = 0; i < n; i++)
			{
                int ithDigit = board / iPower;
				iPower *= 10;
		        ithDigit = ithDigit % 10;
				jPower=1;
				for (j = 0; j < n; j++)
				{
                    int jthDigit = board / jPower;
					jPower *= 10;
		            jthDigit = jthDigit % 10;
					if (i < j && ithDigit == jthDigit) sum += 1;
					if (i < j && (ithDigit - jthDigit == i - j || ithDigit - jthDigit == j - i))
						sum += 1;
				}
			}
			if (sum == 0)
			{
				#pragma omp atomic
				sols++;
			}
		}
	}
	
    end = omp_get_wtime();
    printf("Exec time= %g sec\n", end - start);
    printf("Solutions= %d\n", sols);
    
	return 0;
}