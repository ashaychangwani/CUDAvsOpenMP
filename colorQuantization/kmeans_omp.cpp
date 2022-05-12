#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>
#include <omp.h>
#include "utils.h"


// _clusters-Means cluster centeral pixels
// struct Cluster_OMP {
// 	uint8_t x;                // R
// 	uint8_t y;                // G
// 	uint8_t z;                // B
// 	std::vector<bool> pixels; // Bool Array
// };

int recenter(int n, int K_clusters, Pixel* pixels, Cluster* clusters, bool* visited) {
	for (int i = 0; i < K_clusters; ++i) {
		Cluster *tmp = &clusters[i];
		int count = 0;
		int sum_x, sum_y, sum_z;
		sum_x = sum_y = sum_z = 0;
		#pragma omp teams distribute \
			parallel for reduction(+:sum_x, sum_y, sum_z, count)
		for (int j = 0; j < n; ++j) {
			Pixel* pixel = &pixels[j];
			bool flag = visited[i*n + j];
			sum_x += pixel->x*flag;
			sum_y += pixel->y*flag;
			sum_z += pixel->z*flag;
			count+=flag;
		}

		// printf("%d", tmp->size, count);
		if (count > 0) {
			Cluster copy = clusters[i];
			clusters[i].x = sum_x / count;
			clusters[i].y = sum_y / count;
			clusters[i].z = sum_z / count;

			if (copy.x != clusters[i].x || copy.y != clusters[i].y || copy.z != clusters[i].z) {
				return 0;
			}
		}
	}

	return 1;
}

int main(int argc, const char** argv) {
	
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
	Pixel* pixels = new Pixel[n_pixels];
	int i=0;
	
	while(i<n_pixels){
		pixels[i].x = image[3*i+ 0];
		pixels[i].y = image[3*i+ 1];
		pixels[i].z = image[3*i+ 2];
		pixels[i].cluster = -1;
		i++;
	}

	Cluster* clusters = new Cluster[K_clusters];
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(5);
	std::uniform_int_distribution<> uniform(0, n_pixels - 1);
	
	//Initialize Cluster and Assign a Random Pixel to the Cluster
	i=0;
	while (i<K_clusters){
		Pixel *pixel = &pixels[uniform(gen)];
		clusters[i].x = pixel->x; clusters[i].y = pixel->y; clusters[i].z = pixel->z;
		i++;
	}
	
	//Used as a 2d array
	bool visited[K_clusters*n_pixels];

	do {
		// #pragma omp target teams distribute parallel for
		for (int i = 0; i < K_clusters; i++) {
			memset(visited, 0, K_clusters*n_pixels*sizeof(bool));
		}
		//Find the Minimum Distance Cluster
		#pragma omp shared(n_pixels, K_clusters) 
		{
		#pragma omp target teams distribute parallel for map(to:n_pixels, K_clusters)\
														shared(n_pixels, K_clusters)\
														map(tofrom:pixels[0:n_pixels], clusters[0:K_clusters], visited[0:K_clusters*n_pixels])\
														schedule(auto) 
			for (int idx = 0; idx < n_pixels; ++idx) {
				int min_dist = INT_MAX, min_cluster;
				for (int j = 0; j < K_clusters; ++j) {
					int dist = 	square((pixels[idx].x - clusters[j].x))+ 
							square((pixels[idx].y - clusters[j].y)) + 
							square((pixels[idx].z - clusters[j].z));
					
					if (dist <= min_dist) {
						min_dist = dist;
						min_cluster = j;
					}
				}
				pixels[idx].cluster = min_cluster;
				visited[min_cluster*n_pixels + idx] = 1;
			}
		}
	} while (recenter(n_pixels, K_clusters, pixels, clusters, visited) == 0);
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

