#include <cuda_runtime.h>
#include <windowsnumerics.h>
#include <cstdlib>
#include <ctime>
#include <random>

#define EPS 0.01 // softening factor
#define EPS2 EPS * EPS
#define G = 6.67430 // gravitational const. (10^-11 m³/kgs³)

// Configuration
#define N 10 // Number of particles
#define dt 1 // Time step (second?)
#define dt2 (dt * dt)
float steps = 10000;
#define p 2 // Threads per block / Block dimension (how many?)
float L = 3; // box width (in meter?)

// ranges for the initial position of the particles
int posXMax = 1;
int posXMin = -posXMax;
int posYMax = posXMax;
int posYMin = posXMin;
int posZMax = posXMax;
int posZMin = posXMin;
// ranges for the initial position of the particles
int accXMax = 0;
int accXMin = -accXMax;
int accYMax = accXMax;
int accYMin = accXMin;
int accZMax = accXMax;
int accZMin = accXMin;
// TODO: ranges for the initial weight of the particles
float parWeight = 1;

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


__device__ void update_pos(float4* pos, float3* acc)
{
    pos->x += 0.5 * acc->x * dt2 + acc->x * dt;
    pos->y += 0.5 * acc->y * dt2 + acc->y * dt;
    pos->z += 0.5 * acc->z + dt2 + acc->z * dt;
}

__global__ void calculate_forces(void *devX, void *devA)
{
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
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
    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    // Already update position here when acceleration is already in shared memory
    update_pos(&globalX[gtid], &acc);
    globalA[gtid] = acc4;
    }


int main()
{
    // Allocate space for particles on host
    float4* h_pos = (float4*) malloc(N * sizeof(float4));
    float4* h_acc = (float4*) malloc(N * sizeof(float4));

    // Allocate space for particles on device
    float4* d_pos;
    float4* d_acc;
    cudaMalloc(&d_pos, N);
    cudaMalloc(&d_acc, N);

    // Initialize particles on host? (for now yes, but probably faster on gpu)
    std::srand(std::time({})); // initialise seed
    for (int i = 0; i < N; i++) {
        h_pos[i].w = parWeight;
        h_pos[i].x = posXMin + ( std::rand() % ( posXMax - posXMin + 1 ) );
        h_pos[i].y = posYMin + ( std::rand() % ( posYMax - posYMin + 1 ) );
        h_pos[i].z = posZMin + ( std::rand() % ( posZMax - posZMin + 1 ) );
        h_acc[i].x = accXMin + ( std::rand() % ( accXMax - accXMin + 1 ) );
        h_acc[i].y = accYMin + ( std::rand() % ( accYMax - accYMin + 1 ) );
        h_acc[i].z = accZMin + ( std::rand() % ( accZMax - accZMin + 1 ) );
    }

    // TODO: visualize particles
    printf("Initial particles:\n");
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Position (%f, %f, %f), Acceleration (%f, %f, %f)\n",
               i, h_pos[i].x, h_pos[i].y, h_pos[i].z,
               h_acc[i].x, h_acc[i].y, h_acc[i].z);
    }

    // Copy particles from host to device
    cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc, h_acc, N * sizeof(float4), cudaMemcpyHostToDevice);

    // TODO: Run calculation for n time steps
    int grid_dim = N / p; // TODO: probably fix type
    for (int i = 0; i < steps; i++) {
        calculate_forces<<<grid_dim,p>>>(d_pos, d_acc);
    }

    // Copy particles form device to host
    cudaMemcpy(h_pos, d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_acc, d_acc, N * sizeof(float4), cudaMemcpyDeviceToHost);

    // TODO: Update visualization
    printf("Initial particles:\n");
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Position (%f, %f, %f), Acceleration (%f, %f, %f)\n",
               i, h_pos[i].x, h_pos[i].y, h_pos[i].z,
               h_acc[i].x, h_acc[i].y, h_acc[i].z);
    }
}