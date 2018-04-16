#pragma once

#include <cuda.h>
#include "error.hpp"

namespace cuda {

	namespace driver {

		void initialize() {
			// TODO: What about possible future start-up flags?
			attempt(cuInit(0), "Failed to initialize CUDA driver API");
		}

	} // namespace driver

} // namespace cuda