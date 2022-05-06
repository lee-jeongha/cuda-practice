// reference
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <stdio.h>

__global__ void AplusB(int *ret, int a, int b) {
	ret[threadIdx.x] = a + b + threadIdx.x;
}

int main(){
	/*
	int *ret;
	cudaMalloc(&ret, 1000*sizeof(int));
	AplusB<<< 1, 1000 >>>(ret, 10, 100);
	int *host_rest = (int*)malloc(1000*sizeof(int));
	cudaMemcpy(host_ret, ret, 1000*sizeof(int), cudaMemcpyDefault);
	for(int i=0; i<1000; i++)
		printf("%d: A+B = %d\n", i, host_ret[i]);
	free(host_rest);
	cudaFree(ret);
	return 0;
	*/

	int *ret;
	cudaMallocManaged(&ret, 1000*sizeof(int));
	AplusB<<< 1, 1000 >>>(ret, 10, 100);
	cudaDeviceSynchronize();
	for(int i=0; i<1000; i++)
		printf("%d: A+B = %d\n", i, ret[i]);
	cudaFree(ret);
	return 0;
}
