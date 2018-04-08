
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device.hpp"
#include "error.hpp"
#include "kernel.cu"
#include "memory.hpp"
#include "version.hpp"

#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

template < class T >
void print_array(const T *a, size_t size) {
	if (size == 0)
		return ;

	std::cout << "{";
	for (size_t i = 1; i < size; ++i) {
		std::cout << a[i-1] << ",";
	}
	std::cout << a[size - 1] << "}";
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
	try {

		auto kernel = cuda::launch::make_kernel(addKernel);
		cuda::device<true> &dev = cuda::device<true>::get();

		cuda::stream s;

		std::cout << "CUDA device: " << dev.get_name() << '\n';
		std::cout << "Compute capability: " << dev.get_properties().compute_capability() << '\n';
		std::cout << "Driver version: " << cuda::driver::version() << '\n';
		std::cout << "Runtime version: " << cuda::runtime::version() << '\n';
		std::cout << "Binary version: " << kernel.get_attributes().binary_version() << '\n';
		std::cout << "PTX version: " << kernel.get_attributes().binary_version() << '\n';

		cuda::host::input_buffer<int> dev_a(arraySize);
		cuda::host::input_buffer<int> dev_b(arraySize);
		cuda::host::output_buffer<int> dev_c(arraySize);

		dev_a.copy_from(a);
		dev_b.copy_from(b);

		// Launch a kernel on the GPU with one thread for each element.
		//kernel.launch(dim3(1, 1, 1), dim3(arraySize, 1, 1), dev_c.data(), dev_a.data(), dev_b.data());
		s.enqueue(kernel.stream_launch(dim3(1, 1, 1), dim3(arraySize, 1, 1), dev_c.data(), dev_a.data(), dev_b.data()));

		// Synchronize and copy the results back from the GPU
		dev.synchronize();
		dev_c.copy_into(c);

		print_array(a, arraySize);
		std::cout << " + ";
		print_array(b, arraySize);
		std::cout << " = ";
		print_array(c, arraySize);
		std::cout << std::endl;
	}
	catch (const std::exception &e) {
		std::cout.flush();
		std::cerr << e.what() << std::endl;
	}

	cuda::device<true>::get().reset();

    return 0;
}