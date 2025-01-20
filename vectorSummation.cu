#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include <iostream>
#include "DS_timer.h"

#define NUM_DATA 1024 * 1024

__global__ void vectorAdd(int *da, int *db, int *dc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > NUM_DATA)
    {
        return;
    }

    dc[i] = da[i] + db[i];
}

int main()
{
    DS_timer timer(5);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation(Kernel)");
    timer.setTimerName(2, (char *)"Data Trans. : Host -> Device");
    timer.setTimerName(3, (char *)"Data Trans. : Device -> Host");
    timer.setTimerName(4, (char *)"VecAdd on Host");
    timer.initTimers();
    int *a, *b, *c, *hc;
    int *da, *db, *dc;
    long memSize = NUM_DATA * sizeof(int);
    int numBlocks = (NUM_DATA + 1024 - 1) / 1024;

    printf("%d elements, memSize = %ld bytes\n", NUM_DATA, memSize);

    a = new int[NUM_DATA];
    memset(a, 0, memSize);
    b = new int[NUM_DATA];
    memset(b, 0, memSize);
    c = new int[NUM_DATA];
    memset(c, 0, memSize);
    hc = new int[NUM_DATA];
    memset(hc, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++)
    {
        hc[i] = a[i] + b[i];
    }
    timer.offTimer(4);

    cudaMalloc(&da, memSize);
    cudaMalloc(&db, memSize);
    cudaMalloc(&dc, memSize);

    timer.onTimer(0);

    timer.onTimer(2);
    cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    timer.onTimer(1);
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(1024, 1, 1);
    vectorAdd<<<dimGrid, dimBlock>>>(da, db, dc);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    timer.onTimer(3);
    cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);

    timer.offTimer(0);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    timer.printTimer();

    bool result = true;
    for (int i = 0; i < NUM_DATA; i++)
    {
        if (c[i] != hc[i])
        {
            printf("Error: c[%d] = %d, hc[%d] = %d\n", i, c[i], i, hc[i]);
            result = false;
        }
    }

    if (result)
    {
        printf("Success!\n");
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] hc;

    return 0;
}