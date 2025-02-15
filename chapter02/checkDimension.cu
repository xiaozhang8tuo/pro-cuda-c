// nvcc -o check checkDimension.cu
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

// threadIdx 对应着block的dim
// blockIdx  对应着grid的dim

// grid.x 2 grid.y 1 grid.z 1
// block.x 3 block.y 1 block.z 1
// threadIdx:(0,0,0)blockIdx:(1,0,0)blockDim:(3,1,1)gridDim:(2,1,1)
// threadIdx:(1,0,0)blockIdx:(1,0,0)blockDim:(3,1,1)gridDim:(2,1,1)
// threadIdx:(2,0,0)blockIdx:(1,0,0)blockDim:(3,1,1)gridDim:(2,1,1)
// threadIdx:(0,0,0)blockIdx:(0,0,0)blockDim:(3,1,1)gridDim:(2,1,1)
// threadIdx:(1,0,0)blockIdx:(0,0,0)blockDim:(3,1,1)gridDim:(2,1,1)
// threadIdx:(2,0,0)blockIdx:(0,0,0)blockDim:(3,1,1)gridDim:(2,1,1)

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 6;

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}
