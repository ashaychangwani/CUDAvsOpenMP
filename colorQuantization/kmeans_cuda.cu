#include <cstdio>
#include <random>
#include "utils.h"

#define square(X) X*X
#define THREADS_PER_BLOCK 512

/*****************************************************************/

/*** Kernel Definitions ***/
__global__ void find_cluster(int, int, Pixel*, const Cluster* __restrict__);
__global__ void recenter1(int, int, Pixel*, uint4*);
__global__ void recenter2(Cluster*, uint4*, bool*);
/**** end of the kernel declaration ***/

/*****************************************************************/

int main(int argc, char * argv[]) {
	
	if(argc != 4){
        fprintf(stderr, "usage: kmeans_sequential <IN_PATH> <OUT_PATH> <K_CLUSTERS> \n");
        exit(1);
    }

	const char *inPath, *outPath;
	inPath = argv[1]; outPath = argv[2];
	int K_clusters = atoi(argv[3]);
	unsigned int height, width, channels;
	
	unsigned char* image;

	if (read_png(inPath, &image, height, width, channels)!= 0) {
		exit(1);
	}
	if (channels !=3){
		printf("Three channel PNG only supported as of now");
	}

	int n_pixels = height * width;
	Pixel* pixels = (Pixel*)calloc(n_pixels, sizeof(Pixel));
	int i=0;
	while(i<n_pixels){
		pixels[i].x = image[3*i+ 0];
		pixels[i].y = image[3*i+ 1];
		pixels[i].z = image[3*i+ 2];
		pixels[i].cluster = -1;
		i++;
	}

	Cluster* clusters = (Cluster*)calloc(K_clusters, sizeof(Cluster));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> uniform(0, n_pixels - 1);
	i=0;
	//Initialize Cluster and Assign a Random Pixel to the Cluster
	while (i<K_clusters){
		Pixel *pixel = &pixels[uniform(gen)];
		clusters[i++] = Cluster(pixel->x, pixel->y, pixel->z, 0, (int*)calloc(n_pixels, sizeof(int)));
	}
	//Define Blocks and Threads per block
    dim3 numBlocks((n_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

	// allocate device memory
	Pixel* d_pixels;
	Cluster* d_cluster;
	size_t sz_pixel= sizeof(Pixel), sz_cluster = sizeof(Cluster);
	uint4* d_sum;	//To store sum and count
	bool* d_converged;

	cudaMalloc((void**)&d_pixels, n_pixels * sz_pixel);
	cudaMalloc((void**)&d_cluster, K_clusters * sz_cluster);
	cudaMalloc((void**)&d_sum, sizeof(uint4));
	cudaMalloc((void**)&d_converged, sizeof(bool));

	if(!d_pixels && !d_cluster && !d_sum && !d_converged){
        printf("cannot allocate array\n");
        exit(1);
    }

	cudaMemcpy(d_pixels, pixels, n_pixels * sz_pixel, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster, clusters, K_clusters * sz_cluster, cudaMemcpyHostToDevice);


	bool thread_converged = true;
	bool converged;
	do{
		find_cluster<<<numBlocks , threadsPerBlock>>>(n_pixels, K_clusters, d_pixels, d_cluster);
		thread_converged = true;
		for (int i = 0; i < K_clusters; ++i) {
			cudaMemset(d_sum, 0, 4 * sizeof(int));

			recenter1<<<numBlocks , threadsPerBlock>>>(n_pixels, i, d_pixels, d_sum);
			recenter2<<<1, 1>>>(&d_cluster[i], d_sum, d_converged);

			cudaMemcpy(&converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

			thread_converged &= converged;
		}

	}while (!thread_converged);

	// copy device memory back to host
	cudaMemcpy(pixels, d_pixels, n_pixels * sz_pixel, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, d_cluster, K_clusters * sz_cluster, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_pixels);
	cudaFree(d_cluster);
	cudaFree(d_sum);

	int idx = 0;
	while(idx < n_pixels){
		Cluster* cluster = &clusters[pixels[idx].cluster];
		image[3*idx] = cluster->x;
		image[3*idx+1] = cluster->y;
		image[3*idx+2] = cluster->z;
		idx++;
	}

	if ((write_png(outPath, image, height, width, 3)) != 0) {
		printf("fail to write output png file\n");
		exit(1);
	}

	delete[] clusters;
	delete[] pixels;

	return 0;
}


__global__ void find_cluster(int n_pixels, int K_clusters, Pixel* pixels, const Cluster* __restrict__ clusters) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n_pixels){
		Pixel pixel = pixels[idx];
		int min = INT_MAX, min_cluster, dist, j=0;

		while(j<K_clusters){
			dist = square((pixel.x - clusters[j].x))+ square((pixel.y - clusters[j].y)) + square((pixel.z - clusters[j].z));
			if (dist < min) {
				min = dist;
				min_cluster = j;
			}
			j++;
		}
		pixels[idx].cluster = min_cluster;
	}
}

__global__ void recenter1(int n_pixels, int cluster, Pixel* pixels, uint4* sumc) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n_pixels){
		Pixel pixel = pixels[idx];
		if (pixel.cluster == cluster) {
			atomicAdd(&sumc->x, pixel.x);
			atomicAdd(&sumc->y, pixel.y);
			atomicAdd(&sumc->z, pixel.z);
			atomicAdd(&sumc->w, 1);
		}
	}

}

__global__ void recenter2(Cluster* cluster, uint4* sumc, bool* converged) {
	uint32_t points = sumc->w ;
	*converged = false;
	if (points > 0) {
		Cluster copy = *cluster;
		cluster->x = (sumc->x) / (points);
		cluster->y = (sumc->y) / (points);
		cluster->z = (sumc->z) / (points);
		if (cluster->x == copy.x && 
			cluster->y == copy.y && 
			cluster->z == copy.z)	*converged=true;
	}
}
