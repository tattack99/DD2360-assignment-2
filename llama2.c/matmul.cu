#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(float* xout, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

extern "C" void matmul(float* xout, float* x, float* w, int n, int d) {
    float *d_xout, *d_x, *d_w;
    size_t size_xout = d * sizeof(float);
    size_t size_x = n * sizeof(float);
    size_t size_w = d * n * sizeof(float);

    cudaMalloc((void**)&d_xout, size_xout);
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_w, size_w);

    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, size_w, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, size_xout, cudaMemcpyDeviceToHost);

    cudaFree(d_xout);
    cudaFree(d_x);
    cudaFree(d_w);
}
