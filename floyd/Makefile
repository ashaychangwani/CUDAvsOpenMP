
Floyd_OMP.out:Floyd.cpp
/ext3/gcc-offload/install/bin/g++ -fopenmp $^ -o $@

Floyd_CUDA.out:Floyd.cu
	nvcc $^ -o  $@

clean:
	rm -rf *.out
