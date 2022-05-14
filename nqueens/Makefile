
queen_OMP.out:openmp-queen.cpp
/ext3/gcc-offload/install/bin/g++ -fopenmp $^ -o $@

queen_CUDA.out:cuda-queen.cu
	nvcc $^ -o  $@

clean:
	rm -rf *.out
