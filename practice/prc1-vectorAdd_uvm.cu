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

    //Allocate input vectors A and B to device & host
    int *A;
    int *B;
    int *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
        B[i] = 1;
    }

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        if (i % (threadsPerBlock) == 0) printf("\n");
        printf("%d ", C[i]);
    }
    printf("\n");

    // Free device & host memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
