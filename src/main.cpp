#include <iostream>

#include "calc.h"
#include "visuals.h"


int main() {
	int failure;

	// TODO: load from config file
	/* Default Configuration */
	int Ns[11] = { 1024, 2048, 3072, 4069, 5120, 6144, 8192, 10240, 15360, 16384, 20480 }; // multiple of 1024
	int Ps[] = { 2, 4, 8, 16, 32, 64, 128 }; // multiple of 2
	int N = Ns[0]; // Number of particles ()
	int p = Ps[3]; // Threads per block / Block dimension
	float dt = 0.00001f; // Time step (second?)
	int steps = 100000;
	float L = 3; // box width (in meter?)

	range3 posRange, velRange, accRange;
	/* ranges for the initial position of the particles */
	posRange.xMin = -1;
	posRange.xMax = -posRange.xMin;
	posRange.yMin = posRange.xMin;
	posRange.yMax = posRange.xMax;
	posRange.zMin = posRange.xMin;
	posRange.zMax = posRange.xMax;
	/* ranges for the initial velocity of the particles */
	velRange.xMin = 0;
	velRange.xMax = -velRange.xMin;
	velRange.yMin = velRange.xMin;
	velRange.yMax = velRange.xMax;
	velRange.zMin = velRange.xMin;
	velRange.zMax = velRange.xMax;
	/* ranges for the initial acceleration of the particles */
	accRange.xMin = 0;
	accRange.xMax = -accRange.xMin;
	accRange.yMin = accRange.xMin;
	accRange.yMax = accRange.xMax;
	accRange.zMin = accRange.xMin;
	accRange.zMax = accRange.xMax;

	// TODO: ranges for the initial weight of the particles
	float partWeight = 1;

	NBodyCalc* cudaSim = new NBodyCalc();
	Visualizer* viziepop = new Visualizer();

	failure = viziepop->initGL(); // TODO: add error handling to initGL
	if (failure) {
		std::cout << "ERROR::VISUALIZER::INITIALIZATION_FAILED\n" << std::endl;
		return 1;
	}

	failure = cudaSim->initCalc(N, p, partWeight, posRange, velRange, accRange);
	if (failure) {
		std::cout << "ERROR::CALC::INITIALIZATION_FAILED\n" << std::endl;
		return 1;
	}

	int num_configValues = 500;
	cudaSim->saveFileConfig("lastSimulation", steps / num_configValues);
	int num_energyValues = 50;
	cudaSim->saveFileEnergy("lastEnergy", steps / num_energyValues);

	cudaSim->runSimulation(steps, dt, viziepop);
	
	std::cin.ignore(); // TODO: improve closing

	delete cudaSim;
	delete viziepop;
	return 0;
}