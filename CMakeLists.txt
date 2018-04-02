cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA-wrapper LANGUAGES CXX CUDA)

# Look for CUDA
find_package(CUDA REQUIRED)
#include(FindCUDA)

# Actual library
cuda_add_library(cuda-wrapper STATIC
	wrapper/device.hpp
	wrapper/error.hpp
	wrapper/kernel.cu
	wrapper/memory.hpp
)

# Test executable
cuda_add_executable(wrapper-test
	test/main.cu
	OPTIONS -arch compute_50 -code sm_50)
	
target_include_directories(wrapper-test PRIVATE wrapper)

# Need at least C++ 11 (will propagate to executables)
target_compile_features(cuda-wrapper PUBLIC cxx_std_11)

# CUDA files need special compilation
set_target_properties(cuda-wrapper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(wrapper-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Tell CUDA about our desired compute capability
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

target_link_libraries(wrapper-test cuda-wrapper cuda cudart)