// nvcc -o graph cuda_graph.cu -lcudart -std=c++11
#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void debug_print(int *a) {
    if (a != NULL) {
        printf("in kernel func, a = %d\n", *a);
    } else {
        printf("in kernel func, received NULL pointer\n");
    }
}

int main() {
    int a = 0;
    int *d_a = NULL;
    cudaError_t cudaStatus;

    // CUDA 流和图
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 创建 CUDA 流
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // 执行 CPU 指令
    a = 1;
    printf("1 graph a: %d\n", a);

    // 开始捕获图
    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaStreamDestroy(stream);
        return 1;
    }
    a = 3;
    printf("2 graph a: %d\n", a);


    // 将变量a拷贝到设备内存
    cudaStatus = cudaMallocAsync((void**)&d_a, sizeof(int), stream);
    // cudaStatus = cudaMalloc((void**)&d_a, sizeof(int));//cudaMalloc failed: operation not permitted when stream is capturing
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStreamDestroy(stream);
        return 1;
    }

    // 将变量a拷贝到设备内存
    cudaStatus = cudaMemcpyAsync(d_a, &a, sizeof(int), cudaMemcpyHostToDevice, stream);
    // cudaStatus = cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice); // cudaMemcpy failed: operation would make the legacy stream depend on a capturing blocking stream
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaStreamDestroy(stream);
        return 1;
    }

    // 在捕获图期间不要使用 cudaMemcpy，将其移到捕获图之后
    debug_print<<<1, 1, 0, stream>>>(d_a);

    // 结束捕获图
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaStreamDestroy(stream);
        return 1;
    }

    a = 2;
    printf("3 graph a: %d\n", a);

    // 实例化并执行图
    cudaStatus = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaStreamDestroy(stream);
        return 1;
    }
    cudaStatus = cudaGraphLaunch(instance, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaGraphExecDestroy(instance);
        cudaFree(d_a);
        cudaStreamDestroy(stream);
        return 1;
    }
    cudaStreamSynchronize(stream);
    printf("4 graph a: %d\n", a);

    // 清理资源
    cudaGraphExecDestroy(instance);
    cudaFree(d_a);
    cudaStreamDestroy(stream);


    return 0;
}
