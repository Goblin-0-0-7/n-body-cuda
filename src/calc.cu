#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <cassert>

#include "calc.h"
//#include "visuals.h"

INTEGRATION_METHODS str2IntegMethod(std::string name)
{
    if (name == "EULER") {
        return EULER;
    }
    else if (name == "EULER_IM") {
        return EULER_IM;
    }
    else if (name == "LEAPFROG") {
        return LEAPFROG;
    }
    else if (name == "VERLET") {
        return VERLET;
    }
    else {
        std::cout << "ERROR::CALC::UNKNOWN_INTEGRATION_METHOD: " << name << std::endl;
        return EULER; // default
	}
}



__device__ void updatePosImmediat(float4* pos, float4* vel, float dt);
__device__ void updateVelImmediat(float4* vel, float3* acc, float dt);


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


__global__ void calculate_forces(void* devX, void* devA, int N, int p, float dt)
{
    extern __shared__ float4 shPosition[];
    float4* globalX = (float4*)devX;
    float4* globalA = (float4*)devA;
    float4 myPosition;
    int i, tile;
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    myPosition = globalX[gtid];
    for (i = 0, tile = 0; i < N; i += p, tile++) { // N needs to be divisible by p
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
    globalA[gtid] = acc4;
}


__global__ void calculate_forcesNintegrate(void* devX, void* devV, void* devA, int N, int p, float dt)
{
    extern __shared__ float4 shPosition[];
    float4* globalX = (float4*)devX;
    float4* globalV = (float4*)devV;
    float4* globalA = (float4*)devA;
    float4 myPosition, myVelocity;
    int i, tile;
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    myPosition = globalX[gtid];
    myVelocity = globalV[gtid];
    for (i = 0, tile = 0; i < N; i += p, tile++) { // N needs to be divisible by p
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
    globalA[gtid] = acc4;
    // Already update position here when acceleration is already in local memory
    updateVelImmediat(&myVelocity, &acc, dt);
    updatePosImmediat(&globalX[gtid], &myVelocity, dt);
    // Save velocity to global memory
    globalV[gtid] = myVelocity;
}

/*  -----------------------------------------------  *\
**                Energy Calculations                **
\*  -----------------------------------------------  */

__device__ float calcVelEnergy(float4 pos, float4 vel)
{
    float vel2 = vel.x + vel.y + vel.z;
    return 0.5f * pos.w * vel2;
}


__device__ float calcPotEnergy(float4 myPos, float4 posj, float myEnergy)
{
    float dx = myPos.x - posj.x;
    float dy = myPos.y - posj.y;
    float dz = myPos.z - posj.z;
    float dr = sqrtf(dx * dx + dy * dy + dz * dz); // add softening factor?
    if (dr != 0) { // skip i == j this way
        myEnergy += (myPos.w * myPos.w) / dr;
    }
    return myEnergy;
}


__device__ float energyTileCalculation(float4 myPosition, float myEnergy)
{
    extern __shared__ float4 shPosition[];
    for (int i = 0; i < blockDim.x; i++) {
        myEnergy = calcPotEnergy(myPosition, shPosition[i], myEnergy);
    }
    return myEnergy;
}


__global__ void calcEnergy(float4* pos, float4* vel, float* energy, int N, int p)
{
    extern __shared__ float4 shPosition[];
    float4* globalX = (float4*)pos;
    float4* globalV = (float4*)vel;
    float* globalE = (float*)energy;
    float4 myPosition;
    int i, tile;
    float myEnergy = 0;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    myPosition = globalX[gtid];

    myEnergy = calcVelEnergy(myPosition, globalV[gtid]);

    for (i = 0, tile = 0; i < N; i += p, tile++) { // N needs to be divisible by p
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        myEnergy = energyTileCalculation(myPosition, myEnergy);
        __syncthreads();
    }
    // Save the result in global memory
    globalE[gtid] = myEnergy;
}


__global__ void reduce1(float* energy, int start_step_size = 1)
{
    int tid = threadIdx.x;

    auto step_size = start_step_size;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            energy[fst] += energy[snd];
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    }
}


__global__ void reduce2(float* energy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            energy[fst] += energy[snd];
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    }
}


/*  -----------------------------------------------  *\
**                Integration Methods                **
\*  -----------------------------------------------  */

__device__ void updatePos(float4* pos, float4* vel, float dt)
{
    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;
}

__device__ void updateVel(float4* vel, float4* acc, float dt)
{
    vel->x += acc->x * dt;
    vel->y += acc->y * dt;
    vel->z += acc->z * dt;
}

__device__ void updatePosImmediat(float4* pos, float4* vel, float dt)
{
    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;
}

__device__ void updateVelImmediat(float4* vel, float3* acc, float dt)
{
    vel->x += acc->x * dt;
    vel->y += acc->y * dt;
    vel->z += acc->z * dt;
}


__device__ void updateVelVerlet(float4* vel, float4* acc, float dt)
{
    vel->x += 0.5f + acc->x * dt;
    vel->y += 0.5f + acc->y * dt;
    vel->z += 0.5f + acc->z * dt;
}

__device__ void updatePosVerlet(float4* pos, float4* vel, float4* acc, float dt)
{
    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;
}


__global__ void integrateEuler(int N, float4* pos, float4* vel, float4* acc, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // TODO: verify idx counting
    if (idx < N) {
        float4* globalX = (float4*)pos;
        float4* globalV = (float4*)vel;
        float4* globalA = (float4*)acc;

        updateVel(&globalV[idx], &globalA[idx], dt);
        updatePos(&globalX[idx], &globalV[idx], dt);
    }
}


//__global__ void integrateLeapfrog(int N, float4* pos, float4* vel, float4* acc, float dt, float dt2)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x; // TODO: verify idx counting
//    if (idx < N) {
//        float4* globalX = (float4*)pos;
//        float4* globalV = (float4*)vel;
//        float4* globalA = (float4*)acc;
//
//        updateVel(&globalV[idx], &globalA[idx], dt);
//        updatePos(&globalX[idx], &globalV[idx], &globalA[idx], dt, dt2);
//    }
//}

// TODO: fix Verlet algorithm
__global__ void integrateVerlet1(int N, float4* pos, float4* vel, float4* acc, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // TODO: verify idx counting
    if (idx < N) {
        float4* globalX = (float4*)pos;
        float4* globalV = (float4*)vel;
        float4* globalA = (float4*)acc;

        updateVelVerlet(&globalV[idx], &globalA[idx], dt);
        updatePosVerlet(&globalX[idx], &globalV[idx], &globalA[idx], dt);
    }
}

__global__ void integrateVerlet2(int N, float4* vel, float4* acc, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // TODO: verify idx counting
    if (idx < N) {
        float4* globalV = (float4*)vel;
        float4* globalA = (float4*)acc;

        updateVel(&globalV[idx], &globalA[idx], dt);
    }
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

    energyInterval = 50;
    h_energy = 0;
    d_energy = nullptr;
}

NBodyCalc::~NBodyCalc() {
    free(h_pos);
    free(h_acc);
    cudaFree(d_pos);
    cudaFree(d_acc);
    cudaFree(d_energy);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

int NBodyCalc::initCalc(int N, int p, float partWeight, range3 posRange, range3 velRange, range3 accRange, INTEGRATION_METHODS integMethod, int seed)
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
    cudaError_t d_energy_err = cudaMalloc(&d_energy, N * sizeof(float));

    if (!h_pos || !h_vel || !h_acc || d_pos_err != cudaSuccess || d_vel_err != cudaSuccess || d_acc_err != cudaSuccess || d_energy_err != cudaSuccess) {
        std::cout << "ERROR::CALC::MALLOC_FAILED\n" << std::endl;
        return 1;
    }

    /* Initialise energy */
    cudaMemset(d_energy, 0, N);

    failure = initParticlesHost(seed); // TODO: run with cuda
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

    setIntegrationMethod(integMethod);

    return 0;
}


void NBodyCalc::setIntegrationMethod(INTEGRATION_METHODS integMethod)
{
    switch (integMethod) {
    case EULER:
        integFunc = &NBodyCalc::runEuler;
        break;
    case EULER_IM:
        integFunc = &NBodyCalc::runEulerImmediat;
    case LEAPFROG:
        integFunc = &NBodyCalc::runLeapfrog;
        break;
    case VERLET:
        integFunc = &NBodyCalc::runVerlet;
        break;
    }
}


void NBodyCalc::runEuler(float dt, float dt2)
{
    // TODO: calculate_forces does not jet have block/thread oversaturation handling
    calculate_forces<<<grid_dim, p, sharedBytes >>>(d_pos, d_acc, N, p, dt);
    cudaDeviceSynchronize();
    integrateEuler<<<block_num_integrate, blocksize_integrate>>>(N, d_pos, d_vel, d_acc, dt);
    cudaDeviceSynchronize();
}

void NBodyCalc::runEulerImmediat(float dt, float dt2)
{
    // TODO: calculate_forces does not jet have block/thread oversaturation handling
    calculate_forcesNintegrate << <grid_dim, p, sharedBytes >> > (d_pos, d_vel, d_acc, N, p, dt);
    cudaDeviceSynchronize();
}

void NBodyCalc::runLeapfrog(float dt, float dt2)
{
    // TODO: impelmentation
}

void NBodyCalc::runVerlet(float dt, float dt2)
{
    // TODO: calculate_forces does not jet have block/thread oversaturation handling
    integrateVerlet1 << <block_num_integrate, blocksize_integrate >> > (N, d_pos, d_vel, d_acc, dt);
    cudaDeviceSynchronize();
    calculate_forces<<<grid_dim, p, sharedBytes>>>(d_pos, d_acc, N, p, dt);
    cudaDeviceSynchronize();
    integrateVerlet2<<<block_num_integrate, blocksize_integrate>>>(N, d_vel, d_acc, dt);
    cudaDeviceSynchronize();
}


int NBodyCalc::initParticlesHost(int seed)
{
    if (h_pos == nullptr || h_acc == nullptr) {
        return 1;
    }

    /* Initialize particles on host ? (for now yes, but probably faster on gpu) */
    if (seed < 0) {
        srand((unsigned int)std::time({})); // initialise seed
    }
    else {
        srand(seed);
    }
    for (int i = 0; i < N; i++) {
        h_pos[i].w = partWeight;
        h_pos[i].x = posRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.xMax - posRange.xMin)));
        h_pos[i].y = posRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.yMax - posRange.yMin)));
        h_pos[i].z = posRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (posRange.zMax - posRange.zMin)));

        h_vel[i].w = 0;
        h_vel[i].x = velRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.xMax - velRange.xMin)));
        h_vel[i].y = velRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.yMax - velRange.yMin)));
        h_vel[i].z = velRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (velRange.zMax - velRange.zMin)));

        h_acc[i].w = 0;
        h_acc[i].x = accRange.xMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.xMax - accRange.xMin)));
        h_acc[i].y = accRange.yMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.yMax - accRange.yMin)));
        h_acc[i].z = accRange.zMin + static_cast <float>(rand()) / (static_cast <float>(RAND_MAX / (accRange.zMax - accRange.zMin)));
    }

    return 0;
}

int NBodyCalc::runSimulation(int steps, float dt)
{
    this->dt = dt;
    dt2 = dt * dt;

    /* Progessbar params */
    float progress;
    int barWidth = 50;
    int pos;

    /* ------------------------------- Threads and Blocks ------------------------------- *\
    ** - 1 block == max. 1024 threads                                                     **
    ** - 1 warp == 32 or 64 threads (threads get executed in groups/warps)                **
    ** - 1 SM (streaming multiprocessor) = can run 1024 or 2048 threads at any given time **
    ** - SM: depend on graphics card                                                      **
    \* ---------------------------------------------------------------------------------- */

    // TODO: thread block size should always be a multiple of 32
    grid_dim = N / p; // TODO: probably fix type
    
    block_num_integrate = (int)ceil(N / 1024.0f);
    blocksize_integrate = (N >= 1024) ? 1024 : N;
    sharedBytes = p * sizeof(float4);

    /* Check for hardware parameters */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 = device ID
    assert(p <= prop.maxThreadsPerBlock);
    assert(blocksize_integrate <= prop.maxThreadsPerBlock);

    for (int i = 0; i < steps; i++) {
        /* Save Energy before first calculation */
        if (bsaveEnergy) {
            if (i % energyInterval == 0) {
                calcEnergy<<<grid_dim, p, sharedBytes>>>(d_pos, d_vel, d_energy, N, p);
                cudaDeviceSynchronize();
                sumEnergy();
                cudaMemcpy(&h_energy, d_energy, sizeof(float), cudaMemcpyDeviceToHost);
                saveEnergy(i, h_energy);
                cudaMemset(d_energy, 0, N);
            }
        }

        (this->*integFunc)(dt, dt2);

        /* Check for any cuda errors */
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        if (bsaveConfig) {
            if (i % configInterval == 0) {
                saveConfiguration(i);
            }
        }

        if (bsaveGPU) {
            if (i % gpuInterval == 0) {
                saveGPU(i, 0.0f);
            }
        }

        if (isPerformanceTest) { // Print status to know programm is running fine
            if (i % perfInterval == 0) {
                std::cout << "\r[";
                for (int j = 0; j < barWidth; ++j) {
                    progress = 100.0f * i / steps;
                    pos = static_cast<int>(barWidth * progress / 100.0f);
                    if (j < pos) std::cout << "=";
                    else if (j == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress) << " %";
                std::cout.flush();
            }
        }

        // TODO: update Screen, if updated ->
        //vis->updateScreen(); // TODO: call before calculations, calculations probably take longer than rendering
        // TODO: swap buffers (Note: use PBO - Pixel Buffer Object in OpenGL)
    }

    /* Copy particles form device to host */ // TODO: add error handling
    cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel, d_vel, N * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_acc, d_acc, N * sizeof(float4), cudaMemcpyDeviceToHost);

    return 0;
}

/* Note: works probably only for even N, N <= 2048 or N multiple of 2048 */
void NBodyCalc::sumEnergy()
{
    if (N <= 2048) {
        reduce1<<<1, N / 2>>>(d_energy);
        cudaDeviceSynchronize();
    }
    else {
        int grid_size = ceil(N / 2048.0f);
        int block_size = 1024;
        reduce2<<<grid_size, block_size>>>(d_energy);
        cudaDeviceSynchronize();
        reduce1<<<1, grid_size>>>(d_energy, block_size * 2);
        cudaDeviceSynchronize();

    }
}


void NBodyCalc::saveFileConfig(std::string name, int interval, std::filesystem::path outFolder)
{
    bsaveConfig = true;
    configInterval = interval;
    
    std::string fileName = name + ".txt";
    configFilePath = outFolder / fileName;
    std::ofstream saveFile(configFilePath.string());

    saveConfiguration(-1);
}

void NBodyCalc::saveConfiguration(int step)
{
    std::ofstream saveFile;
    saveFile.open(configFilePath.string(), std::ios::app);

    if (saveFile.is_open()) {

        saveFile << "Iteration Step: " << std::dec << step << "\n";

        cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++) {

            saveFile << std::to_string(h_pos[i].x) << " " << std::to_string(h_pos[i].y) << " " << std::to_string(h_pos[i].z) << " " << "\n";
        }

        saveFile.close();

        std::cout << "Successfully saved iteration: " << step << std::endl;
    }
    else {
        std::cout << "Error: Failed to create or open the file:" << configFilePath << std::endl;
    }
}

void NBodyCalc::saveFileEnergy(std::string name, int interval, std::filesystem::path outFolder)
{
    bsaveEnergy = true;
    energyInterval = interval;

    std::string fileName = name + ".txt";
    energyFilePath = outFolder / fileName;
    std::ofstream saveFile(energyFilePath.string());

    if (saveFile.is_open()) {
        saveFile << "Step,Energy" <<  std::endl;
        saveFile.close();
    }
    else {
        std::cout << "Error: Failed to open the energy file: " << energyFilePath << std::endl;
    }
}

void NBodyCalc::saveEnergy(int step, float energy)
{
    std::ofstream saveFile;
    saveFile.open(energyFilePath.string(), std::ios::app);

    if (saveFile.is_open()) {
        saveFile << step << "," << energy << std::endl;
        saveFile.close();


        std::cout << "Successfully saved energy: " << step << std::endl;
    }
    else {
        std::cout << "Error: Failed to open the energy file: " << energyFilePath << std::endl;
    }
}


void NBodyCalc::saveFileGPU(std::string name, int interval, std::filesystem::path outFolder)
{
    bsaveGPU = true;
    energyInterval = interval;

    std::string fileName = name + ".txt";
    gpuFilePath = outFolder / fileName;
    std::ofstream saveFile(gpuFilePath.string());

    if (saveFile.is_open()) {
        saveFile << "Step,GPU-util" << std::endl;
        saveFile.close();
    }
    else {
        std::cout << "Error: Failed to open the gpu file: " << gpuFilePath << std::endl;
    }
}

void NBodyCalc::saveGPU(int step, float gpuUtil)
{
    std::ofstream saveFile;
    saveFile.open(gpuFilePath.string(), std::ios::app);

    if (saveFile.is_open()) {
        saveFile << step << "," << gpuUtil << std::endl;
        saveFile.close();


        std::cout << "Successfully saved gpu-util: " << step << std::endl;
    }
    else {
        std::cout << "Error: Failed to open the gpu file: " << gpuFilePath << std::endl;
    }
}

void NBodyCalc::setIsPerformanceTest(int steps)
{
    isPerformanceTest = true;
    perfInterval = steps / 10;
}