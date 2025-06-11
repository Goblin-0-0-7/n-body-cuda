#include <iostream>

#include "calc.h"
#include "visuals.h"


int main() {
	int failure;

	/* Default Configuration */
	int N = 10; // Number of particles
	int p = 2; // Threads per block / Block dimension (how many?)
	float dt = 1.0f; // Time step (second?)
	float steps = 10000;
	float L = 3; // box width (in meter?)

	range3 posRange, accRange;
	/* ranges for the initial position of the particles */
	posRange.xMin = -1;
	posRange.xMax = -posRange.xMin;
	posRange.yMin = posRange.xMin;
	posRange.yMax = posRange.xMax;
	posRange.zMin = posRange.xMin;
	posRange.zMax = posRange.xMax;
	/* ranges for the initial position of the particles */
	accRange.xMin = 0;
	accRange.xMax = -accRange.xMin;
	accRange.yMin = accRange.xMin;
	accRange.yMax = accRange.xMax;
	accRange.zMin = accRange.xMin;
	accRange.zMax = accRange.xMax;

	// TODO: ranges for the initial weight of the particles
	float partWeight = 1;

	NBodyCalc* cudaSim = new NBodyCalc();

	failure = cudaSim->initCalc(N, p, partWeight, posRange, accRange);
	if (failure) {
		std::cout << "ERROR::CALC::INIsTIALIZATION_FAILED\n" << std::endl;
		return 1;
	}

	cudaSim->runSimulation(steps, dt, &updateScreen);


	initGL();
	return 0;
}