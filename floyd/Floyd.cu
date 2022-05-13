#include <stdint.h>
#include <bits/stdc++.h>
#include <time.h>
#include "utils.h"

using namespace std;

int random_seed=1234;


__global__ void Floyd_Warshall_CUDA(int V, int i,unsigned long* CUDA_Matrix);

struct timespec start, endtime;

int main(int argc, char** argv){

    cout<<"Floyd Warshall's Algorithm: Sequential vs CUDA Comparison\n\n";
    int N = (int) atoi(argv[1]);

    int graph_size=N*N;
    unsigned long *Matrix=(unsigned long *)calloc(graph_size, sizeof(unsigned long));
    unsigned long *CPU_Matrix=(unsigned long *)calloc(graph_size, sizeof(unsigned long));
    unsigned long *GPU_Matrix=(unsigned long *)calloc(graph_size, sizeof(unsigned long));

    cout<<"N : "<<N<<endl;

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            Matrix[i*N + j] = 0;
        }
    }

    generateRandomGraph(Matrix, N, random_seed);

    
    for(int i=0;i<graph_size;i++){
        CPU_Matrix[i]=Matrix[i];
    }
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         cout<< CPU_Matrix[i*N + j] << " ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"After CPU\n";
    clock_gettime(CLOCK_MONOTONIC, &start);
    Floyd_Warshall_CPU(CPU_Matrix, N);
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    double diff=0;
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         cout<< CPU_Matrix[i*N + j] << " ";
    //     }
    //     cout<<endl;
    // }
    diff=timetaken(start, endtime);;

    cout<<"Time taken for Floyd Warshall on CPU: "<<diff<<endl;

    for(int i=0;i<graph_size;i++){
        GPU_Matrix[i]=Matrix[i];
    }


    unsigned long* CUDA_Matrix;

    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMalloc((void**)&CUDA_Matrix,graph_size*sizeof(unsigned long));

    cudaMemcpy(CUDA_Matrix, GPU_Matrix, graph_size*sizeof(unsigned long), cudaMemcpyHostToDevice);
    int block_size = 512;
    dim3 dimGrid((N+block_size-1)/block_size,N);   

    for(int i=0;i<N;i++){
       Floyd_Warshall_CUDA<<<dimGrid,block_size>>>(N, i, CUDA_Matrix);
       cudaThreadSynchronize();
    }
    
    cudaMemcpy(GPU_Matrix, CUDA_Matrix, graph_size*sizeof(unsigned long), cudaMemcpyDeviceToHost);
    // cout<<endl;
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         cout<< GPU_Matrix[i*N + j] << " ";
    //     }
    //     cout<<endl;
    // }
    
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    double diffGPU=timetaken(start, endtime);;
    
    cout<<"Time taken for Floyd Warshall on CUDA: "<<diffGPU<<endl;

    int match=graph_size;
    for(int i=0;i<graph_size;i++){
        match -= int(CPU_Matrix[i]==GPU_Matrix[i]);
    }
    if(match==0){
       cout<<"The sequential and CUDA outputs match!\n";
    }

    free(Matrix);
    free(CPU_Matrix);
    free(GPU_Matrix);
    cudaFree(CUDA_Matrix);

}

__global__ void Floyd_Warshall_CUDA(int n, int k,unsigned long* CUDA_Matrix){

    int i=blockIdx.x*blockDim.x +threadIdx.x;
    if(i>=n) return;
    __shared__  int min_distance;
 

    if(threadIdx.x==0){
        min_distance=CUDA_Matrix[n*blockIdx.y+k];
    }

    __syncthreads();

    int idx=n*blockIdx.y + i;
    unsigned long currDistance=CUDA_Matrix[k*n+i];
    unsigned long total_distance=min_distance+currDistance;
    if (CUDA_Matrix[idx] > total_distance) CUDA_Matrix[idx] = total_distance;
}



