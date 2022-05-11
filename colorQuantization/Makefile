
kmeans_seq.out:kmeans_sequential.cpp
	/ext3/gcc-offload/install/bin/g++ -lpng -std=c++11 $^ -o $@

kmeans_omp.out:kmeans_omp.cpp
	/ext3/gcc-offload/install/bin/g++ -fopenmp -lpng -std=c++11 $^ -o $@

kmeans_cuda.out:kmeans_cuda.cu
	nvcc -lpng -std=c++11 $^ -o $@

clean:
	rm -rf *.out
