#include <cuda_runtime.h>

/* const definitions */
#define EPS 0.01 // softening factor
#define EPS2 EPS * EPS
#define G = 6.67430 // gravitational const. (10^-11 m³/kgs³)

typedef struct range3 {
	float xMin;
	float xMax;
	float yMin;
	float yMax;
	float zMin;
	float zMax;
}range3;

class NBodyCalc {
private:
	int N;
	int p;
	float partWeight;
	range3 posRange;
	range3 velRange;
	range3 accRange;
	float4* h_pos;
	float4* h_vel;
	float4* h_acc;
	float4* d_pos;
	float4* d_vel;
	float4* d_acc;

	int initParticlesHost();
public:
	NBodyCalc();
	~NBodyCalc();
	int initCalc(int N, int p, float partWeight, range3 posRange, range3 velRange, range3 accRange);
	int runSimulation(int steps, float dt, void (*updateFunc)());
};