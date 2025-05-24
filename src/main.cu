#include <cuda_runtime.h>
#include <windowsnumerics.h>
#include <cstdlib>
#include <ctime>
#include <random>

#define EPS 0.01 // softening factor
#define EPS2 EPS * EPS
#define G = 6.67430 // gravitational const. (10^-11 m³/kgs³)

// Configuration
float N = 1000; // Number of particles
float dt = 0.01; // Time step (second?)
float steps = 1000;
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
    extern __shared__ float4 shPosition[];
    for (i = 0; i < blockDim.x; i++) {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

__global__ void calculate_forces(void *devX, void *devA)
{
  extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  myPosition = globalX[gtid];
  for (i = 0, tile = 0; i < N; i += p, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  // Save the result in global memory for the integration step.
   float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;
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
        h_pos[i].x = posXMin + ( std::rand() % ( posXMax - posXMin + 1 ) );
        h_pos[i].y = posYMin + ( std::rand() % ( posYMax - posYMin + 1 ) );
        h_pos[i].z = posZMin + ( std::rand() % ( posZMax - posZMin + 1 ) );
        h_acc[i].x = accXMin + ( std::rand() % ( accXMax - accXMin + 1 ) );
        h_acc[i].y = accYMin + ( std::rand() % ( accYMax - accYMin + 1 ) );
        h_acc[i].z = accZMin + ( std::rand() % ( accZMax - accZMin + 1 ) );
    }

    // Copy particles from device to host
    cudaMemcpy(d_pos, h_pos, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc, h_acc, N * sizeof(float4), cudaMemcpyHostToDevice);

    // TODO: Run calculation for n time steps
    for (int i = 0; i < steps; i++;) {
        calculate_forces(d_pos, d_acc);
    }
    // TODO: Copy particles form host to device

    // TODO: Update visualization
}