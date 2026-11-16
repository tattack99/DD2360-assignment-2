#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

/* Prototypes */
void fill_random(float *array, int array_size);
void fill_array(float *array, int array_size, float value);
void copy_array(float *dest, const float *src, int array_size);
int validate(const float *N, const float *M, const float *P, int width);

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start()
{
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info)
{
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %-55s %8.0f us\n", info, time);
}

#define MASK_WIDTH 5

// Threads per block (= tile size). The Makefile can override it: make run TILE=128
#ifndef TILE_WIDTH
#define TILE_WIDTH 256
#endif


__global__ void convolution_1D_basic(float *N, float *M, float *P)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int width = gridDim.x * blockDim.x;
  int r = MASK_WIDTH / 2;   // mask radius: neighbours on each side

  if(i<0||i>=width) return;

  float sum = 0.0f;
  for(int j=-r;j<=r;j++)
  {
    int currentId = i + j;
    if(currentId<0 || currentId>=width) continue;
    sum += N[currentId] * M[j+r];
  }
  P[i] = sum;
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P)
{
  __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1];   // tile + halo on both sides

  int i = blockIdx.x * blockDim.x + threadIdx.x;   // global position
  int width = gridDim.x * blockDim.x;              // array length
  int r = MASK_WIDTH / 2;                          // radius = 2

  // 1. Every thread copies its own element, shifted right by r to leave room for the left halo
  input_tile[threadIdx.x + r] = N[i];

  // 2. The first r threads also load the halo: one value on the left, one on the right.
  //    Positions outside the array get 0 (ghost cells).
  if (threadIdx.x < r) {
    int left  = i - r;                 // r positions before this thread
    int right = i + blockDim.x;        // just past the end of the block
    input_tile[threadIdx.x] = (left >= 0) ? N[left] : 0.0f;
    input_tile[threadIdx.x + blockDim.x + r] = (right < width) ? N[right] : 0.0f;
  }

  // 3. Wait until the whole tile (including halo) is in shared memory
  __syncthreads();

  // 4. Same weighted sum as the basic kernel, but reading from shared memory.
  //    No edge check needed: out-of-bounds neighbours are already 0 in the tile.
  float sum = 0.0f;
  for (int j = -r; j <= r; j++) {
    sum += input_tile[threadIdx.x + r + j] * M[j + r];
  }
  P[i] = sum;
}

int main(int argc, char *argv[])
{

  // Read the arguments from the command line
  int N = atoi(argv[1]);
  if (N % TILE_WIDTH != 0) {
    printf("N must be a multiple of TILE_WIDTH (%d)\n", TILE_WIDTH);
    return 1;
  }

  int size_N = N * sizeof(float);
  int size_M = MASK_WIDTH * sizeof(float);
  int size_P = N * sizeof(float);

  //@@ Allocate the host memory
  float *hostN;
  float *hostM;
  float *hostP;
  float *hostP_tiled;
  cputimer_start();
  hostN = (float *) malloc(size_N); // The input array N of length N
  hostM = (float *) malloc(size_M); // The 1D mask M of length MASK_WIDTH
  hostP = (float *) malloc(size_P); // The output array P of length N (basic kernel)
  hostP_tiled = (float *) malloc(size_P); // Output of the tiled kernel
  cputimer_stop("Allocated host memory");


  //@@ Allocate the device memory
  float *deviceN;
  float *deviceM;
  float *deviceP;
  float *deviceP_tiled;   
  cputimer_start();
  gpuCheck(cudaMalloc((void **)&deviceN,size_N));
  gpuCheck(cudaMalloc((void **)&deviceM,size_M));
  gpuCheck(cudaMalloc((void **)&deviceP,size_P));
  gpuCheck(cudaMalloc((void **)&deviceP_tiled,size_P));
  cputimer_stop("Allocated device memory");


  //@@ Initialize N with random values
  //@@ Initialize P with 0.0
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  float mask[MASK_WIDTH] = {-0.25f, 0.5f, 1.0f, 0.5f, 0.25f};

  cputimer_start();
  fill_random(hostN,N);
  fill_array(hostP,N,0);
  copy_array(hostM, mask, MASK_WIDTH);
  cputimer_stop("Initialize arrays with values");


  cputimer_start();
  gpuCheck(cudaMemcpy(deviceN,hostN,size_N,cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM,hostM,size_M,cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU");


  /* Call the basic kernel */
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (basic)
  cputimer_start();
  convolution_1D_basic<<<N / TILE_WIDTH, TILE_WIDTH>>>(deviceN, deviceM, deviceP);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());   // wait for the kernel, so the timer measures it
  cputimer_stop("Finished 1D convolution(basic)");


  cputimer_start();
  gpuCheck(cudaMemcpy(hostP, deviceP, size_P, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 5; i++) printf("P[%d] = %f\n", i, hostP[i]);
  cputimer_stop("Copying output P to the CPU and print out the results");

  /* Call the tiled kernel */
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)
  cputimer_start();
  convolution_1D_tiled<<<N / TILE_WIDTH, TILE_WIDTH>>>(deviceN, deviceM, deviceP_tiled);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(tiled)");

  cputimer_start();
  gpuCheck(cudaMemcpy(hostP_tiled, deviceP_tiled, size_P, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 5; i++) printf("P_tiled[%d] = %f\n", i, hostP_tiled[i]);
  cputimer_stop("Copying output P to the CPU and print out the results");


  //@@ Validate the results from the two implementations
  int errors = validate(hostN, hostM, hostP, N);
  int errors_tiled = validate(hostN, hostM, hostP_tiled, N);
  printf("Basic kernel vs CPU: %s (%d mismatches)\n", errors ? "FAIL" : "PASS", errors);
  printf("Tiled kernel vs CPU: %s (%d mismatches)\n", errors_tiled ? "FAIL" : "PASS", errors_tiled);

  cputimer_start();
  gpuCheck(cudaFree(deviceN));
  gpuCheck(cudaFree(deviceM));
  gpuCheck(cudaFree(deviceP));
  gpuCheck(cudaFree(deviceP_tiled));
  free(hostN);
  free(hostM);
  free(hostP);
  free(hostP_tiled);
  cputimer_stop("Free memory resources");

  return 0;
}


/* Implementation of Prototypes */
void fill_random(float *array, int array_size) {
    for (int i = 0; i < array_size; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }
}

void fill_array(float *array, int array_size, float value){
   for(int i=0;i<array_size;i++){
      array[i] = value;
   }
}

void copy_array(float *dest, const float *src, int array_size) {
    for (int i = 0; i < array_size; i++) {
        dest[i] = src[i];
    }
}

// Recomputes the convolution on the CPU and counts how many elements of P differ.
int validate(const float *N, const float *M, const float *P, int width) {
    int errors = 0;
    for (int i = 0; i < width; i++) {
        float ref = 0.0f;
        for (int j = -MASK_WIDTH/2; j <= MASK_WIDTH/2; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < width) ref += N[idx] * M[j + MASK_WIDTH/2];
        }
        if (fabsf(ref - P[i]) > 1e-5f) errors++;
    }
    return errors;
}
