#include <stdint.h>
#include <bits/stdc++.h>
#include <time.h>
#include <omp.h>
#include "utils.h"
using namespace std;


int random_seed=1234;

void Floyd_Warshall_OpenMP(unsigned long* Matrix, int n);

struct timespec start, endtime;

int main(int argc, char * argv[]){
    
    cout<<"Floyd Warshall's Algorithm: Sequential vs OpenMP Comparison\n\n";
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
    // cout<<omp_get_max_threads()<<endl<<endl;
    omp_set_num_threads(1000);
    for(int i=0;i<graph_size;i++){
        GPU_Matrix[i]=Matrix[i];
    }


    clock_gettime(CLOCK_MONOTONIC, &start);
    Floyd_Warshall_OpenMP(GPU_Matrix, N);
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    double diffGPU=timetaken(start, endtime);
    // for(int i=0;i<N;i++){
    //     for(int j=0;j<N;j++){
    //         cout<< GPU_Matrix[i*N + j] << " ";
    //     }
    //     cout<<endl;
    // }
    cout<<"Time taken for Floyd Warshall on OpenMP: "<<diffGPU<<endl;

    int match=graph_size;
    for(int i=0;i<graph_size;i++){
        match -= int(CPU_Matrix[i]==GPU_Matrix[i]);
    }
    if(match==0){
       cout<<"The sequential and OpenMP outputs match!\n";
    }
    // cout<<match<<endl;
    free(Matrix);
    free(CPU_Matrix);
    free(GPU_Matrix);

}



void Floyd_Warshall_OpenMP(unsigned long *Matrix, int n){
    int current_node;
    for(int x=0;x<n;x++){
    #pragma omp shared(Matrix)
        {
    #pragma omp target teams distribute parallel for private(current_node) map(tofrom:Matrix[0:n*n]) map(to: current_node, n)
        for(int y=0;y<n;y++){
            for(int z=0;z<n;z++){
                current_node=y*n+z;
		        unsigned long currDistance = Matrix[y*n+x]+Matrix[x*n+z];
                if (Matrix[current_node] > currDistance) {
                    Matrix[current_node] = currDistance;
                }
            }
        }
        }
    }
}

