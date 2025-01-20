#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include <iostream>
#include "DS_timer.h"

#define NUM_DATA 1024

__global__ void vectorAdd(int *da, int *db, int *dc, int numElements)
{
    int i = threadIdx.x;
    if (i < numElements)
    {
        dc[i] = da[i] + db[i];
    }
}

int main()
{
    // DS_timer timer;
    // timer.setTimerName(0, (char *)"CUDA Total");
    // timer.setTimerName(1, (char *)"Computation(Kernel)");
    // timer.setTimerName(2, (char *)"Data Trans. : Host -> Device");
    // timer.setTimerName(3, (char *)"Data Trans. : Device -> Host");
    // timer.setTimerName(4, (char *)"VecAdd on Host");
    int *a, *b, *c, *hc;
    int *da, *db, *dc;
    int memSize = NUM_DATA * sizeof(int);

    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

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

    // timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++)
    {
        hc[i] = a[i] + b[i];
    }
    // timer.offTimer(4);

    cudaMalloc(&da, memSize);
    cudaMalloc(&db, memSize);
    cudaMalloc(&dc, memSize);

    // timer.onTimer(0);

    // timer.onTimer(2);
    cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
    // timer.offTimer(2);

    // timer.onTimer(1);
    vectorAdd<<<1, NUM_DATA>>>(da, db, dc, NUM_DATA);
    cudaDeviceSynchronize();
    // timer.offTimer(1);

    // timer.onTimer(3);
    cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);
    // timer.offTimer(3);

    // timer.offTimer(0);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // timer.printTimer();

    bool result = true;
    for (int i = 0; i < NUM_DATA; i++)
    {
        if (c[i] != hc[i])
        {
            printf("Error: c[%d] = %d, hc[%d] = %d\n", i, c[i], i, hc[i]);
            result = false;
            break;
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