cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA-wrapper LANGUAGES CXX CUDA)

set(DESIRED_CUDA_VERSION 8.0)
set(ARCHITECTURE "compute_20")
set(CODE_ARCH "compute_20")

# Set the prefix path
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/glew")
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/GLFW")

# Look for CUDA
find_package(CUDA ${DESIRED_CUDA_VERSION} EXACT QUIET)

if(CUDA_FOUND)
	message(STATUS "Found CUDA version " ${CUDA_VERSION})
else(CUDA_FOUND)
	# Explicitly set the toolkit directory (TODO: this is for windows only)
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${DESIRED_CUDA_VERSION}")
	find_package(CUDA ${DESIRED_CUDA_VERSION} EXACT QUIET REQUIRED)
	message(STATUS "Found CUDA version " ${CUDA_VERSION} " (manual path)")
endif()

# Look for OpenGL libs
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Set the desired codegen parameters
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=${ARCHITECTURE},code=${CODE_ARCH}")

# Actual library
cuda_add_library(cuda-wrapper STATIC
	wrapper/common.hpp
	wrapper/runtime/device.hpp
	wrapper/runtime/error.hpp
	wrapper/runtime/event.hpp
	wrapper/runtime/kernel.cu
	wrapper/runtime/memory.hpp
	wrapper/runtime/properties.hpp
	wrapper/runtime/stream.hpp
	wrapper/runtime/version.hpp
	wrapper/runtime/utils.hpp
	wrapper/driver/context.hpp
	wrapper/driver/device.hpp
	wrapper/driver/error.hpp
	wrapper/driver/init.hpp
	wrapper/driver/version.hpp
	OPTIONS -gencode=arch=${ARCHITECTURE},code=${CODE_ARCH}
)

# Test executables
cuda_add_executable(runtime-test
	runtime-test/main.cu runtime-test/timer.hpp)
cuda_add_executable(driver-test
	driver-test/main.cu)
cuda_add_executable(ogl-test
	ogl-test/main.cu ogl-test/timer.hpp ogl-test/surface.hpp)
	
target_include_directories(cuda-wrapper PRIVATE wrapper)
target_include_directories(runtime-test PRIVATE wrapper)
target_include_directories(driver-test PRIVATE wrapper)
# Library for OpenGL stuff
target_include_directories(ogl-test PRIVATE wrapper oglwrap
			${OPENGL_INCLUDE_DIRS} ${GLU_INCLUDE_DIRS}
			${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS}
			${GLM_INCLUDE_DIRS})

# Need at least C++ 11 (will propagate to executables)
target_compile_features(cuda-wrapper PUBLIC cxx_std_14)

# CUDA files need special compilation
set_target_properties(cuda-wrapper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(runtime-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(driver-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ogl-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(runtime-test cuda-wrapper cuda cudart)
target_link_libraries(driver-test cuda-wrapper cuda cudart)
target_link_libraries(ogl-test cuda-wrapper cuda cudart ${OPENGL_LIBRARIES} ${GLU_LIBRARIES} ${GLEW_LIBRARIES} glfw glm)