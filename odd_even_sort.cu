#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 20

__global__ void odd_even_sort(int *A)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int phase = 0; phase < N; phase++)
    {
        if ((phase & 1) == (idx & 1) && (idx + 1 < N) && A[idx] > A[idx + 1])
        {
            int temp = A[idx + 1];
            A[idx + 1] = A[idx];
            A[idx] = temp;
        }
        __syncthreads();
    }
}

int main()
{
    // Initialize host array
    int A[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        A[i] = rand()%100;
    }
    printf("Before sorting: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");

    // Allocate device memory
    int *devA;
    cudaMalloc(&devA, sizeof(A));

    // Copy the host array to the device
    cudaMemcpy(devA, A, sizeof(A), cudaMemcpyHostToDevice);

    // Initialize events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sort the array
    cudaEventRecord(start);
    odd_even_sort<<<1, N>>>(devA);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the sorted array back to the host
    cudaMemcpy(A, devA, sizeof(A), cudaMemcpyDeviceToHost);

    // Compute the execution time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");
    printf("Execution time (CUDA): %f ms\n", ms);

    // Free device memory and destroy events
    cudaFree(devA);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/*
Before sorting: 40 3 64 3 33 79 77 68 13 4 90 30 83 89 9 77 55 60 29 88 
Sorted array: 3 3 4 9 13 29 30 33 40 55 60 64 68 77 77 79 83 88 89 90 
Execution time (CUDA): 0.038688 ms
*/
