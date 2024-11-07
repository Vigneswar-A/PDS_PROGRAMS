#include <cuda.h>
#include <stdio.h>

#define N 25

__constant__ int stencil[] = {0.5, 1, 1.5};

__global__ void computeStencil(int *A)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    A[idx] = (idx-1 ? A[idx-1] : 0)*stencil[0] + A[idx]*stencil[1] + ((idx+1)<N ? A[idx+1] : 0);
}

int main()
{
    // Create host and device variables
    int A[N];

    for (int i = 0; i < 25; i++)
    {
        A[i] = i;
    }

    int *devA;

    // Allocate device memory
    cudaMalloc(&devA, sizeof(A));

    // Copy A to device
    cudaMemcpy(devA, A, sizeof(A), cudaMemcpyHostToDevice);

    // Initialize events to compute execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel to compute stencil
    cudaEventRecord(start);
    computeStencil<<<1, N>>>(devA); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result to host memory
    cudaMemcpy(A, devA, sizeof(A), cudaMemcpyDeviceToHost);

    // Compute execution time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Stencil Result: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");
    
    printf("Execution time (CUDA): %fms\n", ms);

    // Free the device memory
    cudaFree(devA);

    return 0;
}

/*
Stencil Result: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 24 
Execution time (CUDA): 0.037952ms
*/
