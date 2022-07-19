// sample code for getting devices info (using CudaGetDeviceCount, cudaGetDeviceProperties)
// https://cpp.hotexamples.com/examples/-/-/cudaGetDeviceCount/cpp-cudagetdevicecount-function-examples.html

#include <stdio.h>

// Host code
int main() {
    // get device count
    int deviceCount = 0;
    cudaError_t err_id = cudaGetDeviceCount(&deviceCount);
    if (err_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) err_id, cudaGetErrorString(err_id));
        return false;
    }
    printf("\ndevice count: %d\n", deviceCount);

    // get device properties
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000,
               (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
    }
    return 0;
}

/*
   device count: 2

   Device 0: "TITAN V"
    CUDA Driver Version / Runtime Version          11.0 / 11.0
    CUDA Capability Major/Minor version number:    7.0

   Device 1: "TITAN V"
    CUDA Driver Version / Runtime Version          11.0 / 11.0
    CUDA Capability Major/Minor version number:    7.0
 */
