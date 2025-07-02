#ifndef AUTOTEST_H
#define AUTOTEST_H

#include <string>

#include "calc.h"

class Test {
private:
	NBodyCalc* sim;
	/* Simulation parameters (TODO: make a prop structure) */
	int N = 0;
	int p = 0;
	float dt = 0.0f;
	int steps = 0;
	INTEGRATION_METHODS integMethod = EULER;

	float partWeight = 1;
	range3 posRange;
	range3 velRange;
	range3 accRange;

	/* Test parameters (currently set but not used) */
	std::string testName = "";
	bool evalEnergy = false;
	bool evalGPU = false;
	bool evalConfig = false;

	/* Test output parameters */
	std::filesystem::path outPath;
	std::filesystem::path outFilePath;

	clock_t startTime = 0;
	clock_t endTime = 0;
	double testTime = 0;

public:
	Test(int N, int p, float dt, int steps, float clusterCubeWidth, INTEGRATION_METHODS integMethod, int seed, std::string testname);
	~Test();
	void deleteSim();
	void runTest();
	void addEnergyEval(int num_values);
	void addGPUEval(int num_values);
	void addConfigEval(int num_values);
	void saveTestEval();
	//void saveTestCase(); //TODO: save test values first to buffer than to file

	/* Getter */
	std::string getName() { return testName; };
};

#endif