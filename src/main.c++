#include <cuda_runtime.h>
#include <windowsnumerics.h>
#include <cstdlib>

#define EPS = 0.01; // softening factor
#define G = 6.67430; // gravitational const. (10^-11 m³/kgs³)

// Configuration
float N = 1000; // Number of particles
float dt = 0.01; // Time step (second?)
float L = 3; // box width (in meter?)

// ranges for the initial position of the particles
int posXMax = 2;
int posXMin = -posXMax;
int posYMax = posXMax;
int posYMin = posXMin;
int posZMax = posXMax;
int posZMin = posXMin;
// ranges for the initial position of the particles
int accXMax = 1;
int accXMin = -accXMax;
int accYMax = accXMax;
int accYMin = accXMin,
int accZMax = accXMax;
int accZMin = accXMin;
// TODO: ranges for the initial weight of the particles


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
    extern __shared__ float4[] shPosition;
    for (i = 0; i < blockDim.x; i++) {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

int main()
{
    // Allocate space for particles on host
    float4* h_pos = (float4*) malloc(N * sizeof(float4));
    float3* h_acc = (float3*) malloc(N * sizeof(float3));

    // Allocate space for particles on device
    float4* d_pos;
    float3* d_acc;
    cudaMalloc(&d_pos, N);
    cudaMalloc(&d_acc, N);

    // Initialize particles on host? (for now yes, but probably faster on gpu)
    std::srand(std::time({})); // initialise seed
    for (int i = 0; i < N; i++) {
        h_pos[i].w = 0.05;
        h_pos[i].x = std::rand(posXMin, posXMax);
        h_pos[i].y = std::rand(posYMin, posYMax);
        h_pos[i].z = std::rand(posZMin, posZMax);
        h_acc[i].x = std::rand(accXMin, accXMax);
        h_acc[i].y = std::rand(accYMin, accYMax);
        h_acc[i].z = std::rand(accZMin, accZMax);
    }

    // TODO: Copy particles from device to host

    // TODO: Run calculation for n time steps

    // TODO: Copy particles form host to device

    // TODO: Update visualization
}