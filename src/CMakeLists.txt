cmake_minimum_required(VERSION 3.18)
project(nbody LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(nbody main.cpp calc.cu autotest.cpp )

include_directories(../include)
#include_directories(../lib)
#target_link_libraries(nbody ../lib)

