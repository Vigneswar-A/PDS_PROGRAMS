#include <cuda.h>
#include <stdio.h>

#define N 16

__global__ void dot_product(int *A, int *B, int *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] * B[idx];   
    }
}

__global__ void reduce(int *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Parallel reduction technique
    for (int s = N / 2; s > 0; s >>= 1) 
    {
        if (idx+s < N)
        {
            C[idx] += C[idx + s];
        }
        __syncthreads();
    }
}

int main()
{
    // Create host and device variables
    int A[N];
    int B[N];
    int C[N];
    int *devA, *devB, *devC;

    // Initialize array with sample input values
    for (int i = 0; i < N; i++)
    {
        A[i] = 1;
        B[i] = 2;
    }

    // Allocate device memory
    cudaMalloc(&devA, sizeof(A));
    cudaMalloc(&devB, sizeof(B));
    cudaMalloc(&devC, sizeof(C));

    // Copy arrays to device
    cudaMemcpy(devA, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, sizeof(B), cudaMemcpyHostToDevice);

    // Compute the dot product
    dot_product<<<1, N>>>(devA, devB, devC);
    reduce<<<1, N>>>(devC);

    // Copy result to host memory
    cudaMemcpy(C, devC, sizeof(C), cudaMemcpyDeviceToHost);

    printf("Dot Product = %d\n", C[0]);

    // Free the device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
