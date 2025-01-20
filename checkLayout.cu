#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void checkIndex()
{
    printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    dim3 dimBlock(3, 1, 3);
    dim3 dimGrid(2, 1, 2);

    printf("dimGrid.x = %d, dimGrid.y = %d, dimGrid.z = %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock.x = %d, dimBlock.y = %d, dimBlock.z = %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    int totalThreads = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;
    printf("Total number of threads = %d\n", totalThreads);

    checkIndex<<<dimGrid, dimBlock>>>();
    // CUDA 오류 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch checkIndex kernel (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    // 커널 실행 완료 대기
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}