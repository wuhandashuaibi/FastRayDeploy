cmake_minimum_required(VERSION 3.14)
# set(CMAKE_BUILD_TYPE "release")
set(CMAKE_BUILD_TYPE "Debug")
project(new_parking)

add_definitions(-std=c++17)
add_definitions(-DDEBUG_INFER)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Release)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 86)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

# set(OpenCV_DIR   "/home/wuhan/opencv4.5.2/include/opencv4/")

set(CUDA_DIR     "/usr/local/cuda")
set(CUDNN_DIR    "/usr/local/cuda")

set(TENSORRT_DIR "/usr/lib/aarch64-linux-gnu")
set(TENSORRT_PLUGIN "./src")
# opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(CUDA REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}
    ${PROJECT_SOURCE_DIR}/src/include
    ${TENSORRT_PLUGIN}
)
MESSAGE(${TENSORRT_DIR}/include)

link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lstdc++fs -std=c++17 -Wall -O0 -Wfatal-errors -pthread -w -g -O0")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lstdc++fs -std=c++17 -O0 -Xcompiler -fPIC -g -G -w ${CUDA_GEN_CODE}")
enable_language(CUDA)
add_subdirectory(src)
add_subdirectory(testcase)
# target_link_libraries(trt_plugin ./lib/lib_plugins.so)
