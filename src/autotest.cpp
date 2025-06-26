#include <string.h>
#include <filesystem>
#include <iostream>
#include <fstream>

#include "autotest.h"
#include "calc.h"

// TODO: add func to read test case config file and generate testcases

Test::Test(int N, int p, float dt, int steps, std::string testname)
{
	sim = new NBodyCalc();

	this->testName = testname;

	this->N = N;
	this->p = p;
	this->dt = dt;
	this->steps = steps;
	
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

	sim->initCalc(N, p, partWeight, posRange, velRange, accRange);

	outPath = std::filesystem::path("../testresults") / testname;
	std::filesystem::create_directories(outPath);

	outFilePath = (outPath / "test.txt");
	std::ofstream saveFile(outFilePath.string());
}

Test::~Test()
{

}


void Test::runTest()
{
	sim->runSimulation(steps, dt);
}

void Test::addEnergyEval(int num_values)
{
	evalEnergy = true;

	sim->saveFileEnergy("energy", steps / num_values, outPath);
}


void Test::addGPUEval(int num_values)
{
	evalGPU = true;

	//sim->saveFileGPU("gpu", steps / num_values, outPath);
}


void Test::addConfigEval(int num_values)
{
	evalConfig = true;

	sim->saveFileConfig("config", steps / num_values, outPath);
}