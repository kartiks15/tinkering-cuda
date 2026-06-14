#include <cstdio>

__global__
void dot_product(float* A, float* B, float* result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    int N = 3;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    result[row * N + col] = sum;
}

int main()
{
    int N = 3;
    float A[3][3] = 
    {
        {1,1,1},
        {1,2,3},
        {1,3,6}
    };

    float B[3][3] = 
    {
        {1,1,1},
        {1,2,3},
        {1,3,6}
    };

    float result[3][3] = {};

    // Device pointers
    float *d_A, *d_B, *d_result;

    // Allocate device memory
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_result, N * N * sizeof(float));

    // Copy data to device (cast 2D arrays to float*)
    cudaMemcpy(d_A, (float*)A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (float*)B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(2, 2);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    dot_product<<<gridDim, blockDim>>>(d_A, d_B, d_result);

    // Copy result back to host
    cudaMemcpy((float*)result, d_result, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    // Print result
    printf("Result matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}