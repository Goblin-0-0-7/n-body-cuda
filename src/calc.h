#ifndef CALC_H
#define CALC_H

#include <cuda_runtime.h>
#include <filesystem>

#include "visuals.h"

/* const definitions */
#define EPS 0.01 // softening factor
#define EPS2 EPS * EPS
#define G = 6.67430 // gravitational const. (10^-11 m³/kgs³)

/* Position calculation methods */
enum INTEGRATION_METHODS { EULER, LEAPFROG, VERLET };

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

	float dt;
	float dt2;
	int grid_dim;
	int block_num_integrate;
	int blocksize_integrate;
	int sharedBytes;
	void (NBodyCalc::* integFunc)(float, float);

	/* Performance Test variables */
	bool isPerformanceTest = true;
	int perfInterval;

	/* Energy Test variables */
	bool bsaveEnergy = false;
	int energyInterval;
	std::filesystem::path energyFilePath;
	float h_energy;
	float* d_energy;

	/* Configuration Test variables */
	bool bsaveConfig = false;
	int configInterval;
	std::filesystem::path configFilePath;

	/* GPU Test variables */
	bool bsaveGPU = false;
	int gpuInterval;
	std::filesystem::path gpuFilePath;

	int initParticlesHost();
public:
	NBodyCalc();
	~NBodyCalc();
	int initCalc(int N, int p, float partWeight, range3 posRange, range3 velRange, range3 accRange, INTEGRATION_METHODS integMethod);
	int runSimulation(int steps, float dt);
	void setIntegrationMethod(INTEGRATION_METHODS integMethod);
	void runEuler(float dt, float dt2);
	void runLeapfrog(float dt, float dt2);
	void runVerlet(float dt, float dt2);
	void sumEnergy();
	void saveFileConfig(std::string name, int interval, std::filesystem::path outFolder);
	void saveConfiguration(int step);
	void saveFileEnergy(std::string name, int interval, std::filesystem::path outFolder);
	void saveEnergy(int step, float energy);
	void saveFileGPU(std::string name, int interval, std::filesystem::path outFolder);
	void saveGPU(int step, float gpuUtil);
	void setIsPerformanceTest(int steps);
};

#endif	