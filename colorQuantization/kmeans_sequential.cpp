#include <cstdio>
#include <random>
#include "utils.h"

#define square(X) X*X


int recenter(int n, int k, Pixel* pixels, Cluster* clusters) {

	for (int i = 0; i < k; ++i) {
		Cluster* cluster = &clusters[i];
		int count = cluster->size;
		Point3d sum(0,0,0);

		for (int j = 0; j < count; ++j) {
			Pixel* pixel = &pixels[cluster->pixels[j]];
			sum.x += pixel->x; sum.y += pixel->y; sum.z += pixel->z;
		}
		if (count > 0){
			Cluster copy = clusters[i];
			clusters[i].x = sum.x / count;
			clusters[i].y = sum.y / count;
			clusters[i].z = sum.z / count;

			if (copy.x != clusters[i].x || copy.y != clusters[i].y || copy.z != clusters[i].z) {
				return 0;
			}
		}
	}

	return 1;
}

int main(int argc, char * argv[]) {
	
	if(argc != 4){
        fprintf(stderr, "usage: kmeans_sequential <IN_PATH> <OUT_PATH> <K_CLUSTERS> \n");
        exit(1);
    }

	const char *inPath, *outPath;
	inPath = argv[1]; outPath = argv[2];
	int K = atoi(argv[3]);
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

	Cluster* clusters = (Cluster*)calloc(K, sizeof(Cluster));
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(5);
	std::uniform_int_distribution<> uniform(0, n_pixels - 1);
	i=0;
	//Initialize Cluster and Assign a Random Pixel to the Cluster
	while (i<K){
		Pixel *pixel = &pixels[uniform(gen)];
		clusters[i++] = Cluster(pixel->x, pixel->y, pixel->z, 0, (int*)calloc(n_pixels, sizeof(int)));
	}
	
	do{
		int k=0, idx = 0;
		while (k< K){
			clusters[k].size = 0;
			k++;
		}
		
		//Find the Minimum Distance Cluster
		
		while(idx < n_pixels){
			int min_dist = INT_MAX, min_cluster, dist;
			for (int j = 0; j < K; ++j) {
				dist = square((pixels[idx].x - clusters[j].x))+ square((pixels[idx].y - clusters[j].y)) + square((pixels[idx].z - clusters[j].z));
				if (dist < min_dist) {
					min_dist = dist;
					min_cluster = j;
				}
			}
			clusters[min_cluster].pixels[clusters[min_cluster].size++] = idx;
			pixels[idx].cluster = min_cluster;
			idx++;
		}

	} while (recenter(n_pixels, K, pixels, clusters)==0);	//Loop Until Convergence of Centroids

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

	return 0;
}
