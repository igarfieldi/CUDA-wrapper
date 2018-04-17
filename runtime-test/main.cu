
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "runtime/device.hpp"
#include "runtime/error.hpp"
#include "runtime/event.hpp"
#include "runtime/kernel.cu"
#include "runtime/memory.hpp"
#include "runtime/version.hpp"
#include "timer.hpp"

#include <iostream>
#include <functional>
#include <thread>

#define CU_SAFE_CALL(x)									\
	do {												\
		auto res = x;									\
		if(res != CUDA_SUCCESS) {						\
			const char *name, *msg;						\
			cuGetErrorName(res, &name);					\
			cuGetErrorString(res, &msg);				\
			std::cerr << "CUDA driver call failed: "	\
					<< msg << " (" << name				\
					<< ")" << std::endl;				\
			exit(EXIT_FAILURE);							\
		}												\
	} while(0)

#define CUDA_SAFE_CALL(x)								\
	do {												\
		auto res = x;									\
		if(res != cudaSuccess) {						\
			std::cerr << "CUDA runtimecall failed: "	\
					<< cudaGetErrorString(res)			\
					<< " ("								\
					<< cudaGetErrorName(res)			\
					<< ")" << std::endl;				\
			exit(EXIT_FAILURE);							\
		}												\
	} while(0)

__global__ void addKernel(int *c, const int *a, const int *b, const unsigned int elems, const unsigned int elems_per_thread)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	for (size_t i = idx * elems_per_thread; i < (idx + 1) * elems_per_thread; ++i) {
		if (i < elems)
			c[i] = a[i] + b[i];
	}
}

void test_device(const cuda::device &dev, const cuda::host::pinned_buffer<int> &a,
		const cuda::host::pinned_buffer<int> &b, const unsigned int array_size) {
	constexpr unsigned int block_size = 1024;
	constexpr unsigned int grid_size = 1024;
	constexpr unsigned int elems_per_thread = 5;

	try {
		auto c = cuda::host::pinned_buffer<int>(array_size, false, false, true);

		dev.make_current();
		auto kernel = cuda::launch::make_kernel(addKernel);
		//auto &s1 = cuda::stream::default_stream();
		auto s1 = cuda::stream(dev, true);
		auto s2 = cuda::stream(dev, true);

		auto evt0 = cuda::event(dev);
		evt0.record(s1);
		cuda::host::input_buffer<int> dev_a1(array_size);
		cuda::host::input_buffer<int> dev_b1(array_size);
		cuda::host::output_buffer<int> dev_c1(array_size);
		cuda::host::input_buffer<int> dev_a2(array_size);
		cuda::host::input_buffer<int> dev_b2(array_size);
		cuda::host::output_buffer<int> dev_c2(array_size);
		auto evt1 = cuda::event(s1);

		s1.enqueue(dev_a1.copy_h2d_async(a));
		s1.enqueue(dev_b1.copy_h2d_async(b));
		auto evt2 = cuda::event(s1);
		s1.enqueue(kernel.stream_launch(dim3(grid_size, 1, 1), dim3(block_size, 1, 1),
			dev_c1.data(), dev_a1.data(), dev_b1.data(), array_size, elems_per_thread));
		auto evt3 = cuda::event(s1);
		s1.enqueue(dev_c1.copy_d2h_async(c));
		auto evt4 = cuda::event(s1);

		s2.enqueue(dev_a2.copy_h2d_async(a));
		s2.enqueue(dev_b2.copy_h2d_async(b));
		s2.enqueue(kernel.stream_launch(dim3(grid_size, 1, 1), dim3(block_size, 1, 1),
			dev_c2.data(), dev_a2.data(), dev_b2.data(), array_size, elems_per_thread));
		s2.enqueue(dev_c2.copy_d2h_async(c));


		// Synchronize (wait for everything to finish executing)
		dev.synchronize();
		std::cout << '\t' << dev.get_properties().name << " (" << dev.used_mem() / 0x100000 << "MB / "
			<< dev.total_mem() / 0x100000 << "MB): "
			<< evt4.elapsed_time(evt0) << "ms ("
			<< evt1.elapsed_time(evt0) << "ms|"
			<< evt2.elapsed_time(evt1) << "ms|"
			<< evt3.elapsed_time(evt2) << "ms|"
			<< evt4.elapsed_time(evt3) << "ms)" << std::endl;
	}
	catch (const std::exception &e) {
		std::cout.flush();
		std::cerr << e.what() << std::endl;
	}
}

void test_device_native(int devno, int *a, int *b, const unsigned int array_size) {
	const unsigned int block_size = 1024;
	const unsigned int grid_size = 1024;
	const unsigned int elems_per_thread = 5;
	int *c;
	CUDA_SAFE_CALL(cudaHostAlloc(&c, array_size * sizeof(*c), cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined));

	CUdevice dev;
	CU_SAFE_CALL(cuDeviceGet(&dev, devno));
	CUcontext ctx;
	CU_SAFE_CALL(cuCtxCreate(&ctx, 0, dev));

	CUstream s1 = {}, s2;
	CU_SAFE_CALL(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));

	CUdeviceptr dev_a1, dev_b1, dev_c1;
	CUdeviceptr dev_a2, dev_b2, dev_c2;

	CU_SAFE_CALL(cuMemAlloc(&dev_a1, array_size * sizeof(*a)));
	CU_SAFE_CALL(cuMemAlloc(&dev_b1, array_size * sizeof(*b)));
	CU_SAFE_CALL(cuMemAlloc(&dev_c1, array_size * sizeof(*c)));
	CU_SAFE_CALL(cuMemAlloc(&dev_a2, array_size * sizeof(*a)));
	CU_SAFE_CALL(cuMemAlloc(&dev_b2, array_size * sizeof(*b)));
	CU_SAFE_CALL(cuMemAlloc(&dev_c2, array_size * sizeof(*c)));

	CU_SAFE_CALL(cuMemcpyHtoDAsync(dev_a1, a, array_size * sizeof(*a), s1));
	CU_SAFE_CALL(cuMemcpyHtoDAsync(dev_b1, b, array_size * sizeof(*b), s1));
	addKernel << <grid_size, block_size, 0, s1 >> > ((int *)dev_c1, (const int *)dev_a1, (const int *)dev_b1, array_size, elems_per_thread);
	CU_SAFE_CALL(cuMemcpyDtoHAsync(c, dev_c1, array_size * sizeof(*c), s1));

	CU_SAFE_CALL(cuMemcpyHtoDAsync(dev_a2, a, array_size * sizeof(*a), s2));
	CU_SAFE_CALL(cuMemcpyHtoDAsync(dev_b2, b, array_size * sizeof(*b), s2));
	addKernel << <grid_size, block_size, 0, s2 >> > ((int *)dev_c2, (const int *)dev_a2, (const int *)dev_b2, array_size, elems_per_thread);
	CU_SAFE_CALL(cuMemcpyDtoHAsync(c, dev_c2, array_size * sizeof(*c), s2));

	CU_SAFE_CALL(cuCtxSynchronize());
	CU_SAFE_CALL(cuMemFree(dev_a1));
	CU_SAFE_CALL(cuMemFree(dev_b1));
	CU_SAFE_CALL(cuMemFree(dev_c1));
	CU_SAFE_CALL(cuMemFree(dev_a2));
	CU_SAFE_CALL(cuMemFree(dev_b2));
	CU_SAFE_CALL(cuMemFree(dev_c2));

	CUDA_SAFE_CALL(cudaFreeHost(c));
	CU_SAFE_CALL(cuStreamDestroy(s2));
	CU_SAFE_CALL(cuCtxDestroy(ctx));
}

void test() {
	constexpr unsigned int array_size = 12000000;
	auto a = cuda::host::pinned_buffer<int>(array_size, true, false, false);
	auto b = cuda::host::pinned_buffer<int>(array_size, true, false, false);
	for (unsigned int i = 0; i < array_size; ++i) {
		a[i] = 2 * i;
		b[i] = array_size - i;
	}

	std::vector<std::thread> threads;
	for (int i = 0; i < cuda::device::count(); ++i) {
		threads.push_back(std::thread(test_device, std::cref(cuda::device::get(i)), std::cref(a), std::cref(b), array_size));
	}

	for (auto &t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}
}

void test_native() {
	constexpr unsigned int array_size = 12000000;
	int *a, *b;

	CUDA_SAFE_CALL(cudaHostAlloc(&a, array_size * sizeof(*a), cudaHostAllocPortable));
	CUDA_SAFE_CALL(cudaHostAlloc(&b, array_size * sizeof(*b), cudaHostAllocPortable));
	for (unsigned int i = 0; i < array_size; ++i) {
		a[i] = 2 * i;
		b[i] = array_size - i;
	}

	int devcount;
	CU_SAFE_CALL(cuDeviceGetCount(&devcount));

	std::vector<std::thread> threads;
	for (int i = 0; i < devcount; ++i) {
		threads.push_back(std::thread(test_device_native, i, a, b, array_size));
	}

	for (auto &t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}

	CUDA_SAFE_CALL(cudaFreeHost(a));
	CUDA_SAFE_CALL(cudaFreeHost(b));
}

int main()
{
	// Warm-up run
	CU_SAFE_CALL(cuInit(0));
	//std::cout << "Warming up CUDA..." << std::endl;
	//test_native();

	std::cout << "Running benchmark..." << std::endl;
	auto timer = cpputils::Timer<>();
	test_native();
	std::cout << "Native completion time: " << timer.duration<>() << "ms" << std::endl;
	timer.reset();
	test();
	std::cout << "Library completion time: " << timer.duration<>() << "ms" << std::endl;
    return 0;
}