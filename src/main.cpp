#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "calc.h"
#include "visuals.h"
#include "autotest.h"

//#define USEGL


int main(int argc, char* argv[]) {
	int failure;
	int N, p, steps;
	float dt;

	int num_energyValues, num_configValues, num_gpuValues;
	bool testing = false;
	int num_tests = 0;
	std::string testFilePath;
	Test* test;
	std::vector<Test> tests;

	/* Handle arguments */
	for (int i = 0; i < argc; i++) {
		if (std::strcmp(argv[i], "-t") == 0) {
			if (i + 1 < argc) {
				testFilePath = argv[i + 1];
				testing = true;
			}
		}
	}

	// TODO: load from config file
	/* Default Configuration */
	int Ns[11] = { 1024, 2048, 3072, 4069, 5120, 6144, 8192, 10240, 15360, 16384, 20480 }; // multiple of 1024
	int Ps[7] = { 2, 4, 8, 16, 32, 64, 128 }; // multiple of 2
	N = Ns[0]; // Number of particles ()
	p = Ps[3]; // Threads per block / Block dimension
	dt = 0.00001f; // Time step (second?)
	steps = 100000;

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
#ifdef USEGL
	Visualizer* viziepop = new Visualizer();

	failure = viziepop->initGL(); // TODO: add error handling to initGL
	if (failure) {
		std::cout << "ERROR::VISUALIZER::INITIALIZATION_FAILED\n" << std::endl;
		return 1;
	}
#endif

	// TODO: check for testcases file -> testing = true
	if (testing) {
		std::cout << "Reading test file" << std::endl;

		/* Read test-cases file */
        json testData;
		std::ifstream testFile(testFilePath);
        if (testFile.is_open()) {
			try {
				testFile >> testData;
			}
			catch (const json::parse_error& e) {
				std::cerr << "JSON parse error at byte " << e.byte << ": " << e.what() << std::endl;
				testFile.close();
				return 1;
			}
			
			/* Generate Tests */
			for (const auto& [testName, params] : testData["test_cases"].items()) {
				N = params["N"];
				p = params["p"];
				dt = params["dt"];
				steps = params["steps"];
				num_energyValues = params["energy values"];
				num_configValues = params["configuration values"];
				num_gpuValues = params["GPU values"];

				test = new Test(N, p, dt, steps, testName);
				if (num_energyValues) {
					test->addEnergyEval(num_energyValues);
				}
				if (num_configValues) {
					test->addConfigEval(num_configValues);
				}
				if (num_gpuValues) {
					test->addGPUEval(num_gpuValues);
				}
				tests.push_back(*test);
			}
			testFile.close();
        } else {
            std::cerr << "Error: Could not open " << testFilePath << std::endl;
        }

		if (tests.empty()) {
			std::cerr << "Error: No tests found in file" << std::endl;
			return 1;
		}
		
		num_tests = tests.size();
		std::cout << num_tests << " tests in queue." << std::endl;

		std::cout << "Running Tests" << std::endl;
		/* Run tests */
		int testID = 1;
		for (Test testCase : tests) {
			std::cout << "Start test " << testID << "/" << num_tests << ": \"" << testCase.getName() << "\"" << std::endl;
			testCase.runTest();
			std::cout << "Finished test " << testID << "/" << num_tests << ": \"" << testCase.getName() << " \" successfully." << std::endl;
			testID++;
		}
	}
	else {
		failure = cudaSim->initCalc(N, p, partWeight, posRange, velRange, accRange, EULER);
		if (failure) {
			std::cout << "ERROR::CALC::INITIALIZATION_FAILED\n" << std::endl;
			return 1;
		}

		num_configValues = 500;
		//TODO: create out folder 
		cudaSim->saveFileConfig("lastSimulation", steps / num_configValues, "../log/");
		num_energyValues = 50;
		cudaSim->saveFileEnergy("lastEnergy", steps / num_energyValues, "../log/");

		cudaSim->runSimulation(steps, dt);
	}
	
	std::cin.ignore(); // TODO: improve closing

	delete cudaSim;
#ifdef USEGL
	delete viziepop;
#endif
	return 0;
}