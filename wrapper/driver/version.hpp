#pragma once

#include "common.hpp"
#include "error.hpp"
#include <cuda.h>

namespace cuda {

	namespace driver {

		struct driver_version_tag {};
		using driver_version = cuda::common::version<driver_version_tag>;

	} // namespace driver

	  // This is for VC++ which can only explicitly specialize templates at namespace scope
	namespace common {

		template < >
		inline version<cuda::driver::driver_version_tag>::version(int combined) : major(combined / 1000), minor((combined % 100) / 10) {
		}

	} // namespace common

	namespace driver {

		driver_version version() {
			int v;
			attempt(cuDriverGetVersion(&v), "Failed to get runtime version");
			return driver_version(v);
		}

	} // namespace driver

} // namespace