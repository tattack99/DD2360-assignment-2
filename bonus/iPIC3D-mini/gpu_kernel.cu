/* ================= GPU PARTICLE MOVER: kernel ================= */
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define MOVER_BLOCK_SIZE 256

// all the small numbers, passed to the kernel in one struct
struct moverParams {
    long   nop;
    int    n_sub_cycles, NiterMover;
    int    nyn, nzn;                 // grid sizes, used as strides by get_idx()
    FPpart dt_sub_cycling, dto2, qomdt2;
    double xStart, yStart, zStart;   // double: same cell index as the CPU
    double Lx, Ly, Lz;
    FPfield invdx, invdy, invdz, invVOL;
    bool   periodicX, periodicY, periodicZ;
};

/* Flat indexing uses the helper already provided in include/Alloc.h:
 *   get_idx(i, j, k, nyn, nzn) = (i*nyn + j)*nzn + k   ==   array[i][j][k] of a *_flat array
 * It is marked __host__ __device__, so it can be called inside the kernel. */

__global__ void mover_PC_kernel(
    FPpart* __restrict__ x, FPpart* __restrict__ y, FPpart* __restrict__ z,
    FPpart* __restrict__ u, FPpart* __restrict__ v, FPpart* __restrict__ w,
    const FPfield* __restrict__ XN, const FPfield* __restrict__ YN,
    const FPfield* __restrict__ ZN,
    const FPfield* __restrict__ Ex, const FPfield* __restrict__ Ey,
    const FPfield* __restrict__ Ez, const FPfield* __restrict__ Bxn,
    const FPfield* __restrict__ Byn, const FPfield* __restrict__ Bzn,
    const moverParams P)
{
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.nop) return;                    // extra threads do nothing

    FPpart px = x[i], py = y[i], pz = z[i];    // load once into registers
    FPpart pu = u[i], pv = v[i], pw = w[i];
    const int nyn = P.nyn, nzn = P.nzn;

    for (int i_sub = 0; i_sub < P.n_sub_cycles; i_sub++) {
        FPpart xptilde = px, yptilde = py, zptilde = pz;
        FPpart uptilde = 0, vptilde = 0, wptilde = 0;

        for (int innter = 0; innter < P.NiterMover; innter++) {
            // which cell is the particle in?
            int ix = 2 + int((px - P.xStart) * P.invdx);
            int iy = 2 + int((py - P.yStart) * P.invdy);
            int iz = 2 + int((pz - P.zStart) * P.invdz);

            // distances to the cell corners
            FPfield xi[2], eta[2], zeta[2];
            xi[0]   = px - XN[get_idx(ix - 1, iy, iz, nyn, nzn)];
            eta[0]  = py - YN[get_idx(ix, iy - 1, iz, nyn, nzn)];
            zeta[0] = pz - ZN[get_idx(ix, iy, iz - 1, nyn, nzn)];
            xi[1]   = XN[get_idx(ix, iy, iz, nyn, nzn)] - px;
            eta[1]  = YN[get_idx(ix, iy, iz, nyn, nzn)] - py;
            zeta[1] = ZN[get_idx(ix, iy, iz, nyn, nzn)] - pz;

            // weighted sum of the 8 corner fields
            FPfield Exl = 0, Eyl = 0, Ezl = 0, Bxl = 0, Byl = 0, Bzl = 0;
            for (int ii = 0; ii < 2; ii++)
              for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++) {
                    FPfield wgt = xi[ii] * eta[jj] * zeta[kk] * P.invVOL;
                    int n = get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn);
                    Exl += wgt * Ex[n];  Eyl += wgt * Ey[n];  Ezl += wgt * Ez[n];
                    Bxl += wgt * Bxn[n]; Byl += wgt * Byn[n]; Bzl += wgt * Bzn[n];
                }

            // push the velocity, then the position
            FPpart omdtsq = P.qomdt2 * P.qomdt2 * (Bxl*Bxl + Byl*Byl + Bzl*Bzl);
            FPpart denom  = 1.0f / (1.0f + omdtsq);
            FPpart ut = pu + P.qomdt2 * Exl;
            FPpart vt = pv + P.qomdt2 * Eyl;
            FPpart wt = pw + P.qomdt2 * Ezl;
            FPpart udotb = ut*Bxl + vt*Byl + wt*Bzl;
            uptilde = (ut + P.qomdt2*(vt*Bzl - wt*Byl + P.qomdt2*udotb*Bxl)) * denom;
            vptilde = (vt + P.qomdt2*(wt*Bxl - ut*Bzl + P.qomdt2*udotb*Byl)) * denom;
            wptilde = (wt + P.qomdt2*(ut*Byl - vt*Bxl + P.qomdt2*udotb*Bzl)) * denom;
            px = xptilde + uptilde * P.dto2;
            py = yptilde + vptilde * P.dto2;
            pz = zptilde + wptilde * P.dto2;
        }

        // final velocity and position of this sub-cycle
        pu = 2.0f * uptilde - pu;  pv = 2.0f * vptilde - pv;  pw = 2.0f * wptilde - pw;
        px = xptilde + uptilde * P.dt_sub_cycling;
        py = yptilde + vptilde * P.dt_sub_cycling;
        pz = zptilde + wptilde * P.dt_sub_cycling;

        // boundary conditions: the same six rules as mover_PC()
        if (px > P.Lx) { if (P.periodicX) px -= P.Lx; else { pu = -pu; px = 2*P.Lx - px; } }
        if (px < 0)    { if (P.periodicX) px += P.Lx; else { pu = -pu; px = -px; } }
        if (py > P.Ly) { if (P.periodicY) py -= P.Ly; else { pv = -pv; py = 2*P.Ly - py; } }
        if (py < 0)    { if (P.periodicY) py += P.Ly; else { pv = -pv; py = -py; } }
        if (pz > P.Lz) { if (P.periodicZ) pz -= P.Lz; else { pw = -pw; pz = 2*P.Lz - pz; } }
        if (pz < 0)    { if (P.periodicZ) pz += P.Lz; else { pw = -pw; pz = -pz; } }
    }

    x[i] = px; y[i] = py; z[i] = pz;           // store once
    u[i] = pu; v[i] = pv; w[i] = pw;
}