#include "matadd.cuh"
#include "DS_timer.h"
#include "DS_definitions.h"

#define ROW_SIZE (1920)
#define COL_SIZE (1080)
#define MAT_SIZE (ROW_SIZE * COL_SIZE)

#define ID2INDEX(_row, _col) (_row * COL_SIZE + _col)

bool MatAddGPU_2D2D(float *_dA, float *_dB, float *_dC);
bool MatAddGPU_1D1D(float *_dA, float *_dB, float *_dC);
bool MatAddGPU_2D1D(float *_dA, float *_dB, float *_dC);

int main(void)
{
    DS_timer timer(8);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation(Kernel)");
    timer.setTimerName(2, (char *)"Data Trans. : Host -> Device");
    timer.setTimerName(3, (char *)"Data Trans. : Device -> Host");
    timer.setTimerName(4, (char *)"Host Performance");

    timer.setTimerName(5, (char *)"Kernel 2D-2D");
    timer.setTimerName(6, (char *)"Kernel 1D-1D");
    timer.setTimerName(7, (char *)"Kernel 2D-1D");
    timer.initTimers();

    float *A, *B, *C[4], *hC;
    float *dA, *dB, *dC;

    //// host memory allocation
    allocNinitMem<float>(&A, MAT_SIZE);
    allocNinitMem<float>(&B, MAT_SIZE);
    LOOP_I(NUM_LAYOUTS)
    {
        allocNinitMem<float>(&C[i], MAT_SIZE);
    }
    allocNinitMem<float>(&hC, MAT_SIZE);

    // device memory allocation
    cudaMalloc(&dA, sizeof(float) * MAT_SIZE);
    cudaMemset(dA, 0, sizeof(float) * MAT_SIZE);
    cudaMalloc(&dB, sizeof(float) * MAT_SIZE);
    cudaMemset(dB, 0, sizeof(float) * MAT_SIZE);
    cudaMalloc(&dC, sizeof(float) * MAT_SIZE);
    cudaMemset(dC, 0, sizeof(float) * MAT_SIZE);

    // input matrix generation
    for (int i = 0; i < MAT_SIZE; i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Check host performamce
    timer.onTimer(4);
    for (int row = 0; row < ROW_SIZE; row++)
    {
        for (int col = 0; col < COL_SIZE; col++)
        {
            hC[ID2INDEX(row, col)] = A[ID2INDEX(row, col)] + B[ID2INDEX(row, col)];
        }
    }
    timer.offTimer(4);

    // copy the input matrices from host memory to device memory
    timer.onTimer(2);
    cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    // *** Kernel call
    timer.onTimer(5);
    MatAddGPU_2D2D(dA, dB, dC);
    cudaDeviceSynchronize();
    timer.offTimer(5);
    timer.onTimer(3);
    cudaMemcpy(C[G2D_B2D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);
    timer.offTimer(3);

    timer.onTimer(6);
    MatAddGPU_1D1D(dA, dB, dC);
    cudaDeviceSynchronize();
    timer.offTimer(6);
    cudaMemcpy(C[G1D_B1D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

    timer.onTimer(7);
    MatAddGPU_2D1D(dA, dB, dC);
    cudaDeviceSynchronize();
    timer.offTimer(7);
    cudaMemcpy(C[G2D_B1D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

    // ***

    // validation
    bool isCorrect = true;
    for (int layout = 0; layout < NUM_LAYOUTS; layout++)
    {
        isCorrect = true;
        for (int i = 0; i < MAT_SIZE; i++)
        {
            if (hC[i] != C[layout][i])
            {
                isCorrect = false;
                break;
            }
        }

        switch (layout)
        {
        case G1D_B1D:
            printf("G1D_B1D");
            break;
        case G2D_B1D:
            printf("G2D_B1D");
            break;
        case G2D_B2D:
            printf("G2D_B2D");
            break;
        }
        if (isCorrect)
            printf(" kernel works well!\n");
        else
            printf(" kernel fails to make correct result(s)..\n");
    }

    timer.printTimer();

    SAFE_DELETE(A);
    SAFE_DELETE(B);
    SAFE_DELETE(hC);
    LOOP_I(4) { SAFE_DELETE(C[i]); }
    return 0;
}

/******************************************************************
 * Complete following three functions
 ******************************************************************/

bool MatAddGPU_2D2D(float *_dA, float *_dB, float *_dC)
{
    //** Set the block and grid layout for your 2D2D kernel **//
    dim3 blockDim(32, 32);
    int numCols = (COL_SIZE + blockDim.x - 1) / blockDim.x;
    int numRows = (ROW_SIZE + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(numCols, numRows, 1);

    return kernelCall(_dA, _dB, _dC, ROW_SIZE, COL_SIZE, ThreadLayout::G2D_B2D, gridDim, blockDim);
}

bool MatAddGPU_1D1D(float *_dA, float *_dB, float *_dC)
{
    dim3 blockDim(32);
    int numBlocks = (COL_SIZE + blockDim.x - 1) / blockDim.x;
    dim3 gridDim(numBlocks);

    return kernelCall(_dA, _dB, _dC, ROW_SIZE, COL_SIZE, ThreadLayout::G1D_B1D, gridDim, blockDim);
}

bool MatAddGPU_2D1D(float *_dA, float *_dB, float *_dC)
{
    //** Set the block and grid layout for your 2D1D kernel **//
    dim3 blockDim(1 /* block dimenstions */);
    dim3 gridDim(1 /* grid dimensions */);
    /***********************************************************/

    return kernelCall(_dA, _dB, _dC, ROW_SIZE, COL_SIZE, ThreadLayout::G2D_B1D, gridDim, blockDim);
}