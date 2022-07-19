// sample code with 2 devices (using CudaMemAdvice)
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html


//cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
//cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, myGpuId);

#include <stdio.h>

// Device code
__global__ void write(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
    ret[threadIdx.x] += a + b + threadIdx.x;
}

// Host code
int main() {
    // **get device count**
    int deviceCount = 0;
    cudaError_t err_id = cudaGetDeviceCount(&deviceCount);
    if (err_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) err_id, cudaGetErrorString(err_id));
        return false;
    }
    printf("\ndevice count: %d\n", deviceCount);

    // **set cuda device**
    cudaSetDevice(0);

    // **memory access**
    int *ret;

    cudaMallocManaged(&ret, 1000 * sizeof(int));
    cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);    //set direct access hint

    write<<< 1, 1000 >>>(ret, 10, 100);    //pages populated in GPU memory
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++) {
        printf("%d: A+B = %d\n", i, ret[i]);
        //directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
        //directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
        //directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
        //directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
    }

    append<<< 1, 1000 >>>(ret, 10, 100);
    cudaDeviceSynchronize();
#if 0
    for (int i = 0; i < 1000; i++) {
        printf("%d: A + B = %d\n", i, ret[i]);
    }
#endif
    cudaFree(ret);
    return 0;
}

/*
   device count: 2
   0: A+B = 110
   1: A+B = 111
   2: A+B = 112
   3: A+B = 113
   4: A+B = 114
   5: A+B = 115
   6: A+B = 116
   7: A+B = 117
   8: A+B = 118
   9: A+B = 119
   10: A+B = 120
   11: A+B = 121
   12: A+B = 122
   13: A+B = 123
   14: A+B = 124
*/
