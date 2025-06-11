#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "visuals.h"

/* source code of vertex shader */
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

/* source code of fragment shader */
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor; \n"
"void main()\n"
"{\n"
"    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); \n"
"}\0";

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void createNattachShaders(unsigned int* shaderProgram)
{
    /* create vertex shader */
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    /* attach shader source code to shader object and compile */
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    /* check for compiling errors of shader */
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    /* create fragment shader */
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    /* create shader program */
    *shaderProgram = glCreateProgram();

    /* attach shaders to shader program */
    glAttachShader(*shaderProgram, vertexShader);
    glAttachShader(*shaderProgram, fragmentShader);
    glLinkProgram(*shaderProgram);
    /* check if linking failed */
    glGetProgramiv(*shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    /* delete shader objects after linking them (no longer needed) */
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

Visualizer::Visualizer()
{
    window = nullptr;
};

Visualizer::~Visualizer()
{
    /* properly clean/delete all GLFW resources */
    glfwTerminate();
}

int Visualizer::initGL()
{
    /* instantiate GLFW window */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* create window object */
    window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    /* initialize GLAD before calling any OpenGL functions */
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /* tell OpenGL location and size of window */
    glViewport(0, 0, 800, 600); // left, bottom, widht, height

    /* register callback function for resizing (update viewport on resize) */
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    createNattachShaders(&shaderProgram);

    ///* define vertices for triangle */
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.0f, -0.5f, 0.0f,
        -0.25f,  0.5f, 0.0f,
        0.0f, 0.5f, 0.0f,
         0.25f, -0.5f, 0.0f,
        0.5f,  0.5f, 0.0f,
    };

    /* define vertices for a rectangle */
    //float vertices[] = {
    // 0.5f,  0.5f, 0.0f,  // top right
    // 0.5f, -0.5f, 0.0f,  // bottom right
    //-0.5f, -0.5f, 0.0f,  // bottom left
    //-0.5f,  0.5f, 0.0f   // top left 
    //};
    //unsigned int indices[] = {  // note that we start from 0!
    //    0, 1, 3,   // first triangle
    //    1, 2, 3    // second triangle
    //};


    /* create element buffer object */
    unsigned int EBO;
    glGenBuffers(1, &EBO);
    /* generate vertex buffer object */
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    /* generate vertex array object */
    //unsigned int VAO;
    glGenVertexArrays(1, &VAO);

    /* bind the Vertex Array Object first, then bindand set vertex buffer(s), and then configure vertex attributes(s). */
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // GL_STREAM_DRAW vs. GL_STATIV_DRAW vs. GL_DYNAMIC_DRAW (for nbody prob. dynamic)

    /* bind element buffer */
    /*glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);*/

    /* tell OpenGL how to interpret vertex data (per vertex attribute) */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); // position of the vertex attr, size of vertex attr (here vec3), type of data, normalize data,
    // stride == space between censecutive vertex attrs, offset of the data in the buffer
    glEnableVertexAttribArray(0);

    
    ///* render loop */
    //while (!glfwWindowShouldClose(window))
    //{
    //    /* input */
    //    processInput(window);

    //    /* rendering comands here */
    //    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    //    glClear(GL_COLOR_BUFFER_BIT);

    //    /* drawing */
    //    glUseProgram(shaderProgram);
    //    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
    //    /* draw triangle */
    //    //glPointSize(10);
    //    glDrawArrays(GL_TRIANGLES, 0, 6); // GL_POINTS
    //    /* draw rectangle */
    //    //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    //    glBindVertexArray(0);

    //    /* check and call events and swap the buffers */
    //    glfwSwapBuffers(window);
    //    glfwPollEvents();
    //}
    return 0;
}

void Visualizer::updateScreen()
{
    if (!glfwWindowShouldClose(window))
    {
        /* input */
        processInput(window);

        /* rendering comands here */
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        /* drawing */
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        /* draw triangle */
        //glPointSize(10);
        glDrawArrays(GL_TRIANGLES, 0, 6); // GL_POINTS
        /* draw rectangle */
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        /* check and call events and swap the buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    else
    {
        /* properly clean/delete all GLFW resources */
        glfwTerminate();
    }
}