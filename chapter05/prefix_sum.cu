// nvcc prefix_sum.cu -o sum
#include <cuda_runtime.h>
#include <iostream>

// CUDA核函数：计算线程束内的前缀和
__global__ void inclusive_scan(int *data) {
    int tid = threadIdx.x;
    int lane = tid % warpSize; // 计算线程在warp内的位置

    // 假设线程束大小是32
    int value = data[tid];

    // 进行线程束内的前缀和计算
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int shfl = __shfl_up(value, offset);
        if (lane >= offset) {
            value += shfl;
        }
        // 注意：__shfl_up 在这个环境中是不必要的 __syncthreads()
        // 因为 __shfl_up 本身就是线程束内部的同步操作
    }

    // 将计算结果写回全局内存
    data[tid] = value;
}

int main() {
    const int SIZE = 32;
    const int ARRAY_BYTES = SIZE * sizeof(int);

    // 生成输入数据
    int h_data[SIZE];
    for (int i = 0; i < SIZE; ++i) h_data[i] = 1; // 使用简单的数据：每个元素都是1

    // 分配GPU内存
    int *d_data;
    cudaMalloc((void **)&d_data, ARRAY_BYTES);
    
    // 将输入数据拷贝到GPU
    cudaMemcpy(d_data, h_data, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // 执行核函数
    inclusive_scan<<<1, SIZE>>>(d_data);

    // 将结果拷贝回主机
    cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < SIZE; ++i) {
        std::cout << "Value at index " << i << " is " << h_data[i] << std::endl;
    }

    // 释放GPU内存
    cudaFree(d_data);

    return 0;
}
