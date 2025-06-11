#ifndef VISUALS_H
#define VISUALS_H

#include <GLFW/glfw3.h>

class Visualizer {
private:
	GLFWwindow* window;
	unsigned int shaderProgram;
	unsigned int VAO; // TODO: remove this variabe, find better solution

public:
	Visualizer();
	~Visualizer();
	int initGL();
	void terminateGL();
	void updateScreen();
};

#endif