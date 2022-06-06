#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


/*** TODO: insert the declaration of the kernel function below this line ***/

__global__ void isValid(int *board, int total, int N, bool *results);

/**** end of the kernel declaration ***/


int main(int argc, char *argv[]){

	int n = 0; 
	int i;  
	int *cpu_boards; 
	int *gpu_boards;
	bool *cpu_results;
	bool *gpu_results;
	clock_t start, end; 
	cudaSetDevice(1);
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
		}
		
	n = atoi(argv[1]);
	
	int max_iter = 1;
    for (int i = 0; i < n; i++)
      max_iter *= n;

	
	if( !(cpu_boards = (int *)malloc(pow(n,n)*sizeof(int))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	if( !(cpu_results = (bool *)malloc(pow(n,n)*sizeof(bool))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}

	start = clock();
	
	size_t space = pow(n,n)*sizeof(int);
	cudaMallocHost(&gpu_boards, space);

	size_t space2 = pow(n,n)*sizeof(bool);
	cudaMallocHost(&gpu_results, space2);

	long iter = 0;
	int idx;
	int number;
	for(iter = 0;iter < max_iter; iter++){
		idx = iter;
		number = 0;
		for(int i=0;i<n;i++){
			number *= 10;
			number += idx % n;
			idx /= n;
		}
		cpu_boards[iter] = number;
		cpu_results[iter] = false;
	}

	cudaMemcpy(gpu_boards, cpu_boards, space, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_results, cpu_results, space2, cudaMemcpyHostToDevice);
	
	int NUM_THREADS = 512;
	int NUM_BLOCKS = max_iter/NUM_THREADS+1;

	
	isValid<<< NUM_BLOCKS, NUM_THREADS >>>(gpu_boards, max_iter, n, gpu_results);

	cudaMemcpy(cpu_results, gpu_results, space2, cudaMemcpyDeviceToHost);

	cudaFreeHost(gpu_results); cudaFreeHost(gpu_boards); 
	
	end = clock();
	printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

	int counter = 0;
	for(i = 0; i < max_iter; i++)
	  if( cpu_results[i]) 
		counter++;

	printf("Final count: %d\n",counter);
		
	free(cpu_results); free(cpu_boards); 

	return 0;
}


/**** TODO: Write the kernel itself below this line *****/
__global__ void isValid(int *board, int total, int N, bool *results){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= total)
        return;
	int number = board[index];
    int errors = 0;
	int ithDigit;
	int jthDigit;
    for(int i=0;i<N;i++){
		ithDigit = number / pow(10.0,(double)i);
		ithDigit = ithDigit % 10;
        for(int j=0;j<N;j++){
			jthDigit = number / pow(10.0,(double)j);
			jthDigit = jthDigit % 10;
            if(i<j && ithDigit == jthDigit) errors++;
            if (i < j && (ithDigit - jthDigit == i - j || ithDigit - jthDigit == j - i)) errors++;
        }
    }
	if(errors > 0)
        results[index] = false;
    else
        results[index] = true;
    
}
