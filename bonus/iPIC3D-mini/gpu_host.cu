/* ================= GPU PARTICLE MOVER: host function ================= */

// allocate GPU memory and copy a host array into it
template <class T>
static T* to_device(const T* host, size_t bytes)
{
    T* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice));
    return d;
}

static void to_host(void* host, const void* d, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(host, d, bytes, cudaMemcpyDeviceToHost));
}

int mover_PC_gpu(struct particles* part, struct EMfield* field,
                 struct grid* grd, struct parameters* param)
{
    std::cout << "***  GPU MOVER with SUBCYCLYING " << param->n_sub_cycles
              << " - species " << part->species_ID << " ***" << std::endl;

    // same setup values as the CPU version
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double) part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;

    moverParams P;
    P.nop = part->nop;  P.n_sub_cycles = part->n_sub_cycles;
    P.NiterMover = part->NiterMover;
    P.nyn = grd->nyn;   P.nzn = grd->nzn;
    P.dt_sub_cycling = dt_sub_cycling;  P.dto2 = dto2;  P.qomdt2 = qomdt2;
    P.xStart = grd->xStart; P.yStart = grd->yStart; P.zStart = grd->zStart;
    P.Lx = grd->Lx; P.Ly = grd->Ly; P.Lz = grd->Lz;
    P.invdx = grd->invdx; P.invdy = grd->invdy; P.invdz = grd->invdz;
    P.invVOL = grd->invVOL;
    P.periodicX = param->PERIODICX; P.periodicY = param->PERIODICY;
    P.periodicZ = param->PERIODICZ;

    size_t npBytes   = (size_t) part->nop * sizeof(FPpart);
    size_t gridBytes = (size_t) grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);

    // allocate on the GPU and copy CPU -> GPU (use the *_flat arrays for grid and fields)
    FPpart *d_x = to_device(part->x, npBytes), *d_y = to_device(part->y, npBytes);
    FPpart *d_z = to_device(part->z, npBytes), *d_u = to_device(part->u, npBytes);
    FPpart *d_v = to_device(part->v, npBytes), *d_w = to_device(part->w, npBytes);
    FPfield *d_XN = to_device(grd->XN_flat, gridBytes);
    FPfield *d_YN = to_device(grd->YN_flat, gridBytes);
    FPfield *d_ZN = to_device(grd->ZN_flat, gridBytes);
    FPfield *d_Ex = to_device(field->Ex_flat, gridBytes);
    FPfield *d_Ey = to_device(field->Ey_flat, gridBytes);
    FPfield *d_Ez = to_device(field->Ez_flat, gridBytes);
    FPfield *d_Bx = to_device(field->Bxn_flat, gridBytes);
    FPfield *d_By = to_device(field->Byn_flat, gridBytes);
    FPfield *d_Bz = to_device(field->Bzn_flat, gridBytes);

    // launch: one thread per particle
    int block  = MOVER_BLOCK_SIZE;
    int blocks = (int) ((part->nop + block - 1) / block);
    mover_PC_kernel<<<blocks, block>>>(d_x, d_y, d_z, d_u, d_v, d_w,
        d_XN, d_YN, d_ZN, d_Ex, d_Ey, d_Ez, d_Bx, d_By, d_Bz, P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back only the particles (the fields did not change)
    to_host(part->x, d_x, npBytes); to_host(part->y, d_y, npBytes);
    to_host(part->z, d_z, npBytes); to_host(part->u, d_u, npBytes);
    to_host(part->v, d_v, npBytes); to_host(part->w, d_w, npBytes);

    // free everything
    cudaFree(d_x);  cudaFree(d_y);  cudaFree(d_z);
    cudaFree(d_u);  cudaFree(d_v);  cudaFree(d_w);
    cudaFree(d_XN); cudaFree(d_YN); cudaFree(d_ZN);
    cudaFree(d_Ex); cudaFree(d_Ey); cudaFree(d_Ez);
    cudaFree(d_Bx); cudaFree(d_By); cudaFree(d_Bz);
    return 0;
}