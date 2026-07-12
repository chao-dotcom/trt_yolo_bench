cmake_minimum_required(VERSION 3.20)
project(trt_yolo_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIB nvinfer)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Install TensorRT C++ SDK and ensure NvInfer.h/libnvinfer are discoverable.")
endif()

add_executable(trt_yolo_bench main.cpp)

target_include_directories(trt_yolo_bench PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_yolo_bench PRIVATE ${OpenCV_LIBS} CUDA::cudart ${TENSORRT_LIB})

if(MSVC)
  target_compile_options(trt_yolo_bench PRIVATE /W4 /permissive-)
else()
  target_compile_options(trt_yolo_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()
