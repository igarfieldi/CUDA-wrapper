
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device.hpp"
#include "error.hpp"
#include "kernel.cu"
#include "memory.hpp"

#include <stdio.h>

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
	try {
		addWithCuda(c, a, b, arraySize);

		printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4]);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cuda::dev::device<true>::get().reset();
	}
	catch (const std::exception &e) {
		fprintf(stderr, "addWithCuda failed! %s\n", e.what());
	}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	auto kernel = cuda::launch::make_kernel(addKernel);
	cuda::dev::device<true> &dev = cuda::dev::device<true>::get();

	cuda::mem::host::input_buffer<int> dev_a(size);
	cuda::mem::host::input_buffer<int> dev_b(size);
	cuda::mem::host::output_buffer<int> dev_c(size);

	dev_a.copy_from(a);
	dev_b.copy_from(b);

	// Launch a kernel on the GPU with one thread for each element.
	kernel.launch(dim3(1, 1, 1), dim3(size, 1, 1), dev_c.data(), dev_a.data(), dev_b.data());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	dev.synchronize();
	dev_c.copy_into(c);
}
