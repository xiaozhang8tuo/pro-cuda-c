// nvcc -O2 -o hello hello.cu
#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

// export PATH=/usr/local/cuda/bin:$PATH
// export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
    // printf("thread %d\n", threadIdx.x);
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    // cudaDeviceSynchronize();
    return 0;
}


