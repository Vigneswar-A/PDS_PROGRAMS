#include <cuda.h>
#include <stdio.h>

#define N 25

__global__ void dot_product(int *A, int *B, int *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    C[idx] = A[idx] * B[idx];  
}

__global__ void reduce(int *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // Parallel reduction technique
    int stride = 1;
    while (stride < N && idx%stride == 0){
        stride *= 2;
        int parent = idx/stride * stride;
        if (parent != idx){
            C[parent] += C[idx];
        }
        __syncthreads();
    }
}

int main()
{
    // Create host and device variables
    int A[N], B[N], C[N];
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

    // Initialize events to compute execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Compute the dot product
    cudaEventRecord(start);
    dot_product<<<1, N>>>(devA, devB, devC);
    reduce<<<1, N>>>(devC);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result to host memory
    cudaMemcpy(C, devC, sizeof(C), cudaMemcpyDeviceToHost);

    // Compute execution time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Dot product : %d\n", C[0]);
    printf("Execution time (CUDA): %fms\n", ms);

    // Free the device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

/*
Dot product : 50
Execution time (CUDA): 0.037312ms
*/
