#include <cuda.h>
#include <stdio.h>

__global__ void transpose(int *matrix)
{
    __shared__ int temp[3][3];  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    temp[idx/3][idx%3] = matrix[idx];
    __syncthreads();
    matrix[idx] = temp[idx%3][idx/3];  
}

int main()
{
    // Create host and device variables
    int matrix[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    int *devMatrix;

    // Allocate device memory
    cudaMalloc(&devMatrix, sizeof(matrix));

    // Copy matrix to device
    cudaMemcpy(devMatrix, matrix, sizeof(matrix), cudaMemcpyHostToDevice);

    // Initialize events to compute execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Transpose the matrix
    cudaEventRecord(start);
    transpose<<<1, 9>>>(devMatrix); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result to host memory
    cudaMemcpy(matrix, devMatrix, sizeof(matrix), cudaMemcpyDeviceToHost);

    // Compute execution time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Transposed matrix: \n");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%d ", matrix[i*3 + j]);
        }
        printf("\n");
    }
    printf("Execution time (CUDA): %fms\n", ms);

    // Free the device memory
    cudaFree(devMatrix);

    return 0;
}

/*
Transposed matrix: 
1 4 7 
2 5 8 
3 6 9 
Execution time (CUDA): 0.041696ms
*/
