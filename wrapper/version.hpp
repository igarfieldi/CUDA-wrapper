#pragma once

#include <cuda_runtime.h>
#include "common.hpp"
#include "error.hpp"

namespace cuda {

	namespace runtime {

		struct runtime_version_tag {};
		using runtime_version = cuda::common::version<runtime_version_tag>;

	} // namespace runtime

	namespace driver {
		
		struct driver_version_tag {};
		using driver_version = cuda::common::version<driver_version_tag>;

	} // namespace driver

	// This is for VC++ which can only explicitly specialize templates at namespace scope
	namespace common {

		template < >
		inline version<cuda::runtime::runtime_version_tag>::version(int combined) : major(combined / 1000), minor((combined % 100) / 10) {
		}

		template < >
		inline version<cuda::driver::driver_version_tag>::version(int combined) : major(combined / 1000), minor((combined % 100) / 10) {
		}

	} // namespace common

	namespace runtime {

		runtime_version version() {
			int v;
			CUDA_TRY(cudaRuntimeGetVersion(&v), "Failed to get runtime version");
			return runtime_version(v);
		}

	} // namespace runtime

	namespace driver {

		driver_version version() {
			int v;
			CUDA_TRY(cudaDriverGetVersion(&v), "Failed to get runtime version");
			return driver_version(v);
		}

	} // namespace driver

} // namespace cuda