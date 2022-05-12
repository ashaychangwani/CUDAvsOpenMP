#include <bits/stdc++.h>
#include <time.h>

#define INFTY 100000000
void generateRandomGraph(unsigned long* Matrix, int n, int random_seed){
    srand(random_seed);
    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            if(i == j){
                Matrix[i*n + j] = 0;
            }

            else{
                int r = rand() % 1000;
                int val = (r == 5)? INFTY: r;
                Matrix[i*n + j] = val;
                Matrix[j*n + i] = val; 
            }
        }
    }
}

void Floyd_Warshall_CPU(unsigned long *Matrix, int n){
    for(int x=0; x<n; x++){
        for(int y=0; y<n; y++){
            for(int z=0; z<n; z++){
		        int current_node=y*n+z;
                unsigned long currDistance = Matrix[y*n+x]+Matrix[x*n+z];
                if (Matrix[current_node] > currDistance) {
                    Matrix[current_node] = currDistance;
                }
            }
        }
    }
}

double timetaken(timespec start, timespec end){
    double start_s= (double)start.tv_sec*1e9 + (double)start.tv_nsec;
    double end_s= (double)end.tv_sec*1e9 + (double)end.tv_nsec;
    return (end_s-start_s)/1e9;
}