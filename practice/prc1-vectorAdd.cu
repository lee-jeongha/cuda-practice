// reference
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <stdio.h>

// Device code
__global__ void VecAdd(int *A, int *B, int *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main() {
    int N = 4096;    // 4*1024
    size_t size = N * sizeof(int);

    //Allocate input vectors h_A and h_B in host memory
    int *h_A = (int *) malloc(size);
    int *h_B = (int *) malloc(size);
    int *h_C = (int *) malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1;
        h_B[i] = 1;
    }

    // Allocate vectors in device memory
    int *d_A;
    cudaMalloc(&d_A, size);
    int *d_B;
    cudaMalloc(&d_B, size);
    int *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors form host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    for (int i = 0; i < N; i++) {
        if (i % (threadsPerBlock) == 0) printf("\n");
        printf("%d ", h_C[i]);
    }
    printf("\n");
    free(h_A);
    free(h_B);
    free(h_C);

}
