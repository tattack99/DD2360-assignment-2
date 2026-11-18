#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <cstdio>
#include <sys/time.h>

/* CUDA */
#define CHECK(call) do {                                            \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",            \
                     cudaGetErrorString(err), __FILE__, __LINE__);  \
        std::exit(1);                                               \
    }                                                               \
} while (0)

#ifndef REPS
#define REPS 5          // timed launches per kernel (after one warm-up launch); compile with -DREPS=1 when profiling
#endif
#define MAX_TILES 16

// ---------------------------------------------------------------------------
// Basic kernel (Assignment 1, renamed gemm).
// One thread computes one C[row, col]. A is M x N, B is N x P, C is M x P.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gemm(const T *A, const T *B, T *C, int M, int N, int P) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= P) return;

    T sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[(size_t)row * N + k] * B[(size_t)k * P + col];
    }
    C[(size_t)row * P + col] = sum;
}

// ---------------------------------------------------------------------------
// Tiled kernel. Launch with blockDim = (tileX, tileY) and dynamic shared
// memory of (tileY*tileX + tileX*tileX) elements of T.
//   thread (tx, ty) computes C[blockIdx.y*tileY + ty, blockIdx.x*tileX + tx]
//   The shared dimension N is walked in steps of tileX:
//     As: tileY x tileX piece of A      Bs: tileX x tileX piece of B
//   Tiles are filled with strided loops so any (tileX, tileY) works.
//   Elements outside the matrices are loaded as 0, so M, N, P need not be
//   multiples of the tile size. Every thread reaches __syncthreads().
// ---------------------------------------------------------------------------
template <typename T>
__global__ void tiled_gemm(const T *A, const T *B, T *C, int M, int N, int P,
                           int tileX, int tileY) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T *As = reinterpret_cast<T *>(smem_raw);       // tileY x tileX
    T *Bs = As + (size_t)tileY * tileX;            // tileX x tileX

    const int step = tileX;                        // k-step
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * tileX + tx;
    const int nthreads = tileX * tileY;
    const int row = blockIdx.y * tileY + ty;
    const int col = blockIdx.x * tileX + tx;

    T sum = 0;
    const int steps = (N + step - 1) / step;
    for (int t = 0; t < steps; t++) {
        for (int idx = tid; idx < tileY * step; idx += nthreads) {   // load A tile
            int r = idx / step, c = idx % step;
            int gr = blockIdx.y * tileY + r, gc = t * step + c;
            As[idx] = (gr < M && gc < N) ? A[(size_t)gr * N + gc] : (T)0;
        }
        for (int idx = tid; idx < step * tileX; idx += nthreads) {   // load B tile
            int r = idx / tileX, c = idx % tileX;
            int gr = t * step + r, gc = blockIdx.x * tileX + c;
            Bs[idx] = (gr < N && gc < P) ? B[(size_t)gr * P + gc] : (T)0;
        }
        __syncthreads();

        for (int k = 0; k < step; k++) sum += As[ty * step + k] * Bs[k * tileX + tx];
        __syncthreads();
    }
    if (row < M && col < P) C[(size_t)row * P + col] = sum;
}

double cpu_ms() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec * 1000.0 + (double)tp.tv_usec / 1000.0;
}

template <typename T>
void fill_random(T *array, size_t count) {
    for (size_t i = 0; i < count; i++) {
        array[i] = (T)rand() / (T)RAND_MAX;
    }
}

// Dot products for the listed rows of C. partial[r * P + col] holds row rows[r].
template <typename T>
void cpu_rows(const T *A, const T *B, T *partial, int N, int P,
              const int *rows, int nrows) {
    for (int r = 0; r < nrows; r++) {
        int row = rows[r];
        for (int col = 0; col < P; col++) {
            T sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[(size_t)row * N + k] * B[(size_t)k * P + col];
            }
            partial[(size_t)r * P + col] = sum;
        }
    }
}

template <typename T>
bool results_match(const T *gpu, const T *partial, int N, int P,
                   const int *rows, int nrows, long long *exact, double *max_abs) {
    const double eps = (sizeof(T) == sizeof(float)) ? 1.2e-7 : 2.3e-16;
    bool ok = true;
    bool printed = false;
    *exact = 0;
    *max_abs = 0.0;

    for (int r = 0; r < nrows; r++) {
        int row = rows[r];
        for (int col = 0; col < P; col++) {
            T cpu_v = partial[(size_t)r * P + col];
            T gpu_v = gpu[(size_t)row * P + col];
            double diff = std::fabs((double)cpu_v - (double)gpu_v);
            if (diff > *max_abs) *max_abs = diff;
            if (cpu_v == gpu_v) (*exact)++;

            double scale = std::fabs((double)cpu_v) + 1.0;
            double tol = eps * (double)N * scale * 8.0;
            if (diff > tol) {
                ok = false;
                if (!printed) {
                    printf("  first mismatch at C[%d, %d]: cpu %.9g gpu %.9g\n",
                           row, col, (double)cpu_v, (double)gpu_v);
                    printed = true;
                }
            }
        }
    }
    return ok;
}

// One warm-up launch, then the mean of REPS launches, timed with CUDA events.
template <typename Launch>
float time_launch(Launch launch) {
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));
    launch();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < REPS; r++) launch();
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    return ms / REPS;
}

template <typename T>
int run(int M, int N, int P, const char *type_name, int verify,
        const int tiles[][2], int num_tiles) {
    const int tx = 16;   // block shape of the basic kernel
    const int ty = 16;
    const int grid_x = (P + tx - 1) / tx;
    const int grid_y = (M + ty - 1) / ty;
    const long long num_blocks = (long long)grid_x * grid_y;
    const long long threads_launched = num_blocks * tx * ty;
    const long long threads_compute = (long long)M * P;
    const long long threads_idle = threads_launched - threads_compute;
    const long long flops = 2LL * M * N * P;
    const long long global_reads = flops; // basic kernel: one A load and one B load per multiply-add

    const size_t nA = (size_t)M * (size_t)N;
    const size_t nB = (size_t)N * (size_t)P;
    const size_t nC = (size_t)M * (size_t)P;
    const size_t bytes_h2d = (nA + nB) * sizeof(T);
    const size_t bytes_d2h = nC * sizeof(T);

    printf("data type: %s\n", type_name);
    printf("matrix A: (%d x %d)\n", M, N);
    printf("matrix B: (%d x %d)\n", N, P);
    printf("matrix C: (%d x %d)\n", M, P);
    printf("floating-point operations: 2*M*N*P = %lld\n", flops);
    printf("basic gemm: %d x %d threads per block, %lld blocks, %lld threads launched, %lld exit idle\n",
           tx, ty, num_blocks, threads_launched, threads_idle);
    printf("basic gemm global memory reads: 2*M*N*P = %lld\n", global_reads);
    printf("bytes copied host to device: %zu (A and B)\n", bytes_h2d);
    printf("bytes copied device to host: %zu (C)\n", bytes_d2h);
    printf("Achieved Occupancy / shared memory (one tile per run): make occupancy PROG=matmul ARGS=\"%d %d %d %s 0 %d %d\"\n",
           M, N, P, type_name, tiles[0][0], tiles[0][1]);
    fflush(stdout);

    T *A = (T *)malloc(nA * sizeof(T));
    T *B = (T *)malloc(nB * sizeof(T));
    T *C_gpu = (T *)malloc(nC * sizeof(T));
    if (!A || !B || !C_gpu) {
        fprintf(stderr, "host malloc failed\n");
        free(A);
        free(B);
        free(C_gpu);
        return 1;
    }
    fill_random(A, nA);
    fill_random(B, nB);

    T *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nA * sizeof(T)));
    CHECK(cudaMalloc((void **)&d_B, nB * sizeof(T)));
    CHECK(cudaMalloc((void **)&d_C, nC * sizeof(T)));

    double t_h2d = cpu_ms();
    CHECK(cudaMemcpy(d_A, A, nA * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, nB * sizeof(T), cudaMemcpyHostToDevice));
    double t_h2d_e = cpu_ms() - t_h2d;
    printf("copy data from host to device: %.3f ms\n", t_h2d_e);

    // ---- CPU reference (computed once, used to check every kernel) ----
    int status = 0;
    int *rows = NULL;
    T *partial = NULL;
    int nrows = 0;
    double cpu_total_ms = 0.0;   // time of the CPU version for the WHOLE C
    int cpu_estimated = 0;       // 1 if cpu_total_ms was scaled up from sampled rows
    if (verify) {
        // A full CPU pass is one thread doing M*N*P multiply-adds. Past 1024^3
        // that takes much longer than the GPU, so check the first, middle, and
        // last rows. Those rows include every column, including the partial block.
        const long long macs = 1LL * M * N * P;
        const long long full_limit = 1024LL * 1024 * 1024;
        if (macs <= full_limit) {
            nrows = M;
            rows = (int *)malloc((size_t)M * sizeof(int));
            if (rows) for (int i = 0; i < M; i++) rows[i] = i;
        } else {
            int candidates[3] = {0, M / 2, M - 1};
            rows = (int *)malloc(3 * sizeof(int));
            if (rows) {
                for (int i = 0; i < 3; i++) {
                    bool seen = false;
                    for (int j = 0; j < nrows; j++) if (rows[j] == candidates[i]) seen = true;
                    if (!seen) rows[nrows++] = candidates[i];
                }
            }
        }
        partial = (T *)malloc((size_t)(nrows > 0 ? nrows : 1) * (size_t)P * sizeof(T));
        if (!rows || !partial) {
            fprintf(stderr, "host malloc failed\n");
            status = 1;
        } else {
            if (nrows == M) printf("CPU reference: computing all %lld elements\n", threads_compute);
            else printf("CPU reference: checking %d rows (%lld elements); a full CPU pass is slower than the GPU here\n",
                        nrows, (long long)nrows * P);
            fflush(stdout);
            double t0 = cpu_ms();
            cpu_rows(A, B, partial, N, P, rows, nrows);
            double t_rows = cpu_ms() - t0;
            if (nrows == M) {
                cpu_total_ms = t_rows;
                printf("CPU version: %.3f ms\n", cpu_total_ms);
            } else {
                cpu_estimated = 1;
                cpu_total_ms = t_rows * (double)M / (double)nrows;   // every row costs the same
                printf("CPU version: %.3f ms for %d rows -> %.3f ms estimated for all %d rows\n",
                       t_rows, nrows, cpu_total_ms, M);
            }
        }
    } else {
        printf("CPU reference: skipped\n");
    }

    // ---- result table ----
    printf("\n%-24s %12s %8s %14s %20s  %s\n",
           "kernel", "time (ms)", "check", "shared/block", "global reads", "details");
    char csv[2048];
    int pos = snprintf(csv, sizeof(csv), "CSV,%s,%d,%d,%d,%.3f,%d",
                       type_name, M, N, P, cpu_total_ms, cpu_estimated);

    // ---- basic kernel ----
    {
        dim3 grid(grid_x, grid_y, 1);
        dim3 tpb(tx, ty, 1);
        CHECK(cudaMemset(d_C, 0, nC * sizeof(T)));
        float ms = time_launch([&]() { gemm<T><<<grid, tpb>>>(d_A, d_B, d_C, M, N, P); });
        CHECK(cudaMemcpy(C_gpu, d_C, nC * sizeof(T), cudaMemcpyDeviceToHost));
        const char *verdict = "n/a";
        long long exact = 0; double max_abs = 0.0;
        if (verify && partial) {
            bool ok = results_match(C_gpu, partial, N, P, rows, nrows, &exact, &max_abs);
            verdict = ok ? "PASS" : "FAIL";
            if (!ok) status = 1;
        }
        printf("%-24s %12.3f %8s %14s %20lld  max abs err %g\n",
               "gemm (16x16)", ms, verdict, "0", global_reads, max_abs);
        pos += snprintf(csv + pos, sizeof(csv) - pos, ",%.4f", ms);
    }

    // ---- tiled kernel, one run per tile size ----
    for (int i = 0; i < num_tiles; i++) {
        const int tX = tiles[i][0], tY = tiles[i][1];
        char name[48];
        snprintf(name, sizeof(name), "tiled_gemm (%dx%d)", tX, tY);
        const size_t smem = ((size_t)tY * tX + (size_t)tX * tX) * sizeof(T);
        if (tX < 1 || tY < 1 || tX * tY > 1024 || smem > 48 * 1024) {
            printf("%-24s %12s %8s   (needs %d threads and %zu bytes of shared memory: limit is 1024 and 49152)\n",
                   name, "-", "SKIPPED", tX * tY, smem);
            continue;
        }
        dim3 grid((P + tX - 1) / tX, (M + tY - 1) / tY, 1);
        dim3 tpb(tX, tY, 1);
        CHECK(cudaMemset(d_C, 0, nC * sizeof(T)));
        float ms = time_launch([&]() {
            tiled_gemm<T><<<grid, tpb, smem>>>(d_A, d_B, d_C, M, N, P, tX, tY);
        });
        CHECK(cudaMemcpy(C_gpu, d_C, nC * sizeof(T), cudaMemcpyDeviceToHost));
        const char *verdict = "n/a";
        long long exact = 0; double max_abs = 0.0;
        if (verify && partial) {
            bool ok = results_match(C_gpu, partial, N, P, rows, nrows, &exact, &max_abs);
            verdict = ok ? "PASS" : "FAIL";
            if (!ok) status = 1;
        }
        // reads = M*N*P*(1/tileX + 1/tileY), ignoring the zero-padded edge tiles
        long long reads = (long long)((double)M * N * P * (1.0 / tX + 1.0 / tY));
        printf("%-24s %12.3f %8s %12zu B %20lld  max abs err %g\n",
               name, ms, verdict, smem, reads, max_abs);
        if (pos < (int)sizeof(csv))
            pos += snprintf(csv + pos, sizeof(csv) - pos, ",%d,%d,%.4f", tX, tY, ms);
    }
    printf("\n%s\n", csv);

    free(rows);
    free(partial);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_gpu);
    return status;
}

static void usage() {
    fprintf(stderr,
            "Usage: matmul [M N P] [float|double] [0|1] [tileX tileY]...\n"
            "       matmul N\n"
            "  A is M x N, B is N x P, C is M x P.\n"
            "  One number N means a square N x N multiply.\n"
            "  Type defaults to double. Verify 1 checks the CPU (default), 0 skips it.\n"
            "  Tile sizes default to 8 8 16 16 32 32; pass pairs to override (max %d pairs).\n",
            MAX_TILES);
}

int main(int argc, char **argv) {
    int M, N, P;
    const char *type_name = "double";
    int verify = 1;
    int tiles[MAX_TILES][2] = {{8, 8}, {16, 16}, {32, 32}};
    int num_tiles = 3;

    if (argc == 1) {
        M = N = P = 512;
    } else if (argc == 2) {
        M = N = P = atoi(argv[1]);
    } else if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        P = atoi(argv[3]);
        if (argc >= 5) type_name = argv[4];
        if (argc >= 6) verify = atoi(argv[5]);
        if (argc >= 7) {
            if ((argc - 6) % 2 != 0 || (argc - 6) / 2 > MAX_TILES) {
                usage();
                return 1;
            }
            num_tiles = (argc - 6) / 2;
            for (int i = 0; i < num_tiles; i++) {
                tiles[i][0] = atoi(argv[6 + 2 * i]);
                tiles[i][1] = atoi(argv[7 + 2 * i]);
            }
        }
    } else {
        usage();
        return 1;
    }

    if (M <= 0 || N <= 0 || P <= 0) {
        usage();
        return 1;
    }
    if (strcmp(type_name, "float") != 0 && strcmp(type_name, "double") != 0) {
        usage();
        return 1;
    }
    if (verify != 0 && verify != 1) {
        usage();
        return 1;
    }

    if (strcmp(type_name, "float") == 0) {
        return run<float>(M, N, P, type_name, verify, tiles, num_tiles);
    }
    return run<double>(M, N, P, type_name, verify, tiles, num_tiles);
}
