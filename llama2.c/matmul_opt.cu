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

static float *d_xout, *d_x;

extern "C" void matmul(float* xout, float* x, float* w, int n, int d) {
    float *d_w;
    size_t size_xout = d * sizeof(float);
    size_t size_x = n * sizeof(float);
    size_t size_w = d * n * sizeof(float);

    // d_x and d_xout are allocated only once, so they must fit the largest call:
    // d_xout: largest d = vocab_size = 32000 (classifier matmul)
    // d_x:    largest n = hidden_dim = 768   (w2 matmul)
    if (d_xout == NULL) { cudaMalloc((void**)&d_xout, 32000 * sizeof(float)); }
    if (d_x == NULL)    { cudaMalloc((void**)&d_x,    768 * sizeof(float)); }

    cudaMalloc((void**)&d_w, size_w);

    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, size_w, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, size_xout, cudaMemcpyDeviceToHost);

    //cudaFree(d_xout);
    //cudaFree(d_x);
    cudaFree(d_w);
}
