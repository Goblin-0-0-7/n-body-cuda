#include <cuda_runtime.h>
#include <windowsnumerics.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

#include "calc.h"
#include "visuals.h"

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;
    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}


__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
    int i;
    extern __shared__ float4 shPosition[];
    for (i = 0; i < blockDim.x; i++) {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}


__device__ void updatePos(float4* pos, float4* vel, float3* acc, float dt, float dt2)
{
    pos->x += 0.5 * acc->x * dt2 + vel->x * dt;
    pos->y += 0.5 * acc->y * dt2 + vel->y * dt;
    pos->z += 0.5 * acc->z + dt2 + vel->z * dt;
}

__device__ void updateVel(float4* vel, float3* acc, float dt)
{
    vel->x += acc->x * dt;
    vel->y += acc->y * dt;
    vel->z += acc->z * dt;
}


__global__ void calculate_forces(void* devX, void*devV, void* devA, int N, int p, float dt, float dt2)
{
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4* globalV = (float4 *)devV;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    myPosition = globalX[gtid];
    for (i = 0, tile = 0; i < N; i += p, tile++) { // N needs to be divisible by p
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx]; // TODO: understand this call
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step. // Note: for the next step, the integration is already done here
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
    // Already update position here when acceleration is already in shared memory
    updateVel(&globalV[gtid], &acc, dt); // TODO: does this really speed things up, what memory calls are really done?
    updatePos(&globalX[gtid], &globalV[gtid], &acc, dt, dt2);
    }

NBodyCalc::NBodyCalc() {
    N = 10;
    p = 2;
    partWeight = 1;
    posRange.xMin = 0;
    posRange.xMax = 0;
    posRange.yMin = 0;
    posRange.yMax = 0;
    posRange.zMin = 0;
    posRange.zMax = 0;
    velRange.xMin = 0;
    velRange.xMax = 0;
    velRange.yMin = 0;
    velRange.yMax = 0;
    velRange.zMin = 0;
    velRange.zMax = 0;
    accRange.xMin = 0;
    accRange.xMax = 0;
    accRange.yMin = 0;
    accRange.yMax = 0;
    accRange.zMin = 0;
    accRange.zMax = 0;
    h_pos = nullptr;
    h_vel = nullptr;
    h_acc = nullptr;
    d_pos = nullptr;
    d_vel = nullptr;
    d_acc = nullptr;
}

NBodyCalc::~NBodyCalc() {
    free(h_pos);
    free(h_acc);
    free(d_pos);
    free(d_acc);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

int NBodyCalc::initCalc(int N, int p, float partWeight, range3 posRange, range3 velRange, range3 accRange)
{
    int failure;

    /* Check if N and p are reasonable values */
    if (p > N) {
        std::cout << "ERROR::CALC::P_VALUE_TO_LARGE\n" << std::endl;
        return 1;
    }
    if (N % p != 0) {
        std::cout << "ERROR::CALC::N_VALUE_NOT_DEVISABLE_BY_P_VALUE\n" << std::endl;
        return 1;
    }

    /* Set values */
    this->N = N;
    this->p = p;
    this->partWeight = partWeight;
    this->posRange = posRange;
    this->accRange = accRange;

    /* Allocate space for particles on host */
    h_pos = (float4*)malloc(N * sizeof(float4));
    h_vel = (float4*)malloc(N * sizeof(float4));
    h_acc = (float4*)malloc(N * sizeof(float4));

    /* Allocate space for particles on device */
    cudaError_t d_pos_err = cudaMalloc(&d_pos, N * sizeof(float4));
    cudaError_t d_vel_err = cudaMalloc(&d_vel, N * sizeof(float4));
    cudaError_t d_acc_err = cudaMalloc(&d_acc, N * sizeof(float4));

    if (!h_pos || !h_vel || !h_acc || d_pos_err != cudaSuccess || d_vel_err != cudaSuccess || d_acc_err != cudaSuccess) {
        std::cout << "ERROR::CALC::MALLOC_FAILED\n" << std::endl;
        return 1;
    }

    failure = initParticlesHost(); // TODO: run with cuda
    if (failure) {
        std::cout << "ERROR::CALC::INIT_PARTICLES_FAILED\n" << std::endl;
        return 1;
    }

    /* copy particles form host to device */
    cudaError_t cpy_pos = cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaError_t cpy_vel = cudaMemcpy(d_vel, h_vel, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaError_t cpy_acc = cudaMemcpy(d_acc, h_acc, N * sizeof(float4), cudaMemcpyHostToDevice);
    if (cpy_pos != cudaSuccess || cpy_acc != cudaSuccess) {
        std::cout << "ERROR::CALC::CUDA::MEMCPY_FAILED\n" << std::endl;
        return 1;
    }

    return 0;
}

int NBodyCalc::initParticlesHost()
{
    if (h_pos == nullptr || h_acc == nullptr) {
        return 1;
    }

    /* Initialize particles on host ? (for now yes, but probably faster on gpu) */
    srand(std::time({})); // initialise seed
    for (int i = 0; i < N; i++) {
        h_pos[i].w = partWeight;
        h_pos[i].x = posRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.xMax - posRange.xMin)));
        h_pos[i].y = posRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.yMax - posRange.yMin)));
        h_pos[i].z = posRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.zMax - posRange.zMin)));

        h_vel[i].x = velRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.xMax - velRange.xMin)));
        h_vel[i].y = velRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.yMax - velRange.yMin)));
        h_vel[i].z = velRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.zMax - velRange.zMin)));

        h_acc[i].x = accRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.xMax - accRange.xMin)));
        h_acc[i].y = accRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.yMax - accRange.yMin)));
        h_acc[i].z = accRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.zMax - accRange.zMin)));
    }

    /* print values for debugging */
    printf("Initial particles:\n");
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Position (%f, %f, %f), Velocity (%f, %f, %f), Acceleration (%f, %f, %f)\n",
            i,
            h_pos[i].x, h_pos[i].y, h_pos[i].z,
            h_vel[i].x, h_vel[i].y, h_vel[i].z,
            h_acc[i].x, h_acc[i].y, h_acc[i].z);
    }

    return 0;
}

int NBodyCalc::runSimulation(int steps, float dt, Visualizer* vis)
{
    float dt2 = dt * dt;

    int grid_dim = N / p; // TODO: probably fix type
    for (int i = 0; i < steps; i++) { // NOTE: this is currently also our render cycle
        calculate_forces<<<grid_dim,p>>>(d_pos, d_vel, d_acc, N, p, dt, dt2); // Note: also inculdes integration step

        cudaDeviceSynchronize();

        // TODO: update Screen, if updated ->
        vis->updateScreen(); // TODO: call before calculations, calculations probably take longer than rendering
        // TODO: swap buffers (Note: use PBO - Pixel Buffer Object in OpenGL)
    }

    /* Copy particles form device to host */ // TODO: add error handling
    cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel, d_vel, N * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_acc, d_acc, N * sizeof(float4), cudaMemcpyDeviceToHost);

    /* print particle values after simulation */
    printf("Simulated particles:\n");
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Position (%f, %f, %f), Velocity (%f, %f, %f), Acceleration (%f, %f, %f)\n",
            i,
            h_pos[i].x, h_pos[i].y, h_pos[i].z,
            h_vel[i].x, h_vel[i].y, h_vel[i].z,
            h_acc[i].x, h_acc[i].y, h_acc[i].z);
    }
    return 0;
}