// reference
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <stdio.h>

// Device code
__global__ void VecAdd(int* A, int* B, int* C, int offset){
	int i = offset + blockDim.x * blockIdx.x + threadIdx.x;
	//if (i < N)
		C[i] = A[i] + B[i];
}

// Host code
int main() {
    const int N = 4096, nStreams = 8;
	size_t size = N * sizeof(int);

    // strem for `cudaMemcpyAsync()`
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

	//Allocate input vectors h_A and h_B in host memory
	int* h_A = (int*)malloc(size);
	int* h_B = (int*)malloc(size);
	int* h_C = (int*)malloc(size);

	// Initialize input vectors
	for(int i=0; i<N; i++){
		h_A[i] = i+1;
		h_B[i] = 1;
	}
	
	// Allocate vectors in device memory
	int* d_A;
	cudaMalloc(&d_A, size);
	int* d_B;
	cudaMalloc(&d_B, size);
	int* d_C;
	cudaMalloc(&d_C, size);

    // Invoke kernel with `cudaMemcpyAsync()`
    int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;    // N / threadsPerBlock

    for (int i = 0; i < nStreams; ++i) {
        // Copy vectors form host memory to device memory
        cudaMemcpyAsync(d_A + i * N, h_A + i * N, size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_B + i * N, h_B + i * N, size, cudaMemcpyHostToDevice, stream[i]);
        // Invoke kernel
        VecAdd <<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>> (d_A + i * N, d_B + i * N, d_C + i * N, i*N);
        // Copy result from device memory to host memory
	    // h_C contains the result in host memory
        cudaMemcpyAsync(h_C + i * N, d_C + i * N, size, cudaMemcpyDeviceToHost, stream[i]);
    }

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	for (int i=0; i<N; i++) {
		if (i%(threadsPerBlock)==0)   printf("\n");
		printf("%d ", h_C[i]);
	}
	printf("\n");
	free(h_A);
	free(h_B);
	free(h_C);

    // Free stream
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]);
	
}