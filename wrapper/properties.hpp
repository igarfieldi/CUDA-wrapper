#pragma once

#include "common.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ostream>
#include <string>

namespace cuda {

	namespace dev {

		struct compute_version_tag {};

		using compute_version = cuda::common::version<compute_version_tag>;

	} // namespace dev

	// This is for VC++ which can only explicitly specialize templates at namespace scope
	namespace common {

		template < >
		inline version<cuda::dev::compute_version_tag>::version(int combined) : major(combined / 10), minor(combined % 10) {
		}

	} // namespace common

	namespace dev {

		struct properties : public cudaDeviceProp {
			compute_version compute_capability() const {
				return compute_version(this->major, this->minor);
			}

			bool is_compute_device() const {
				return computeMode != cudaComputeModeProhibited;
			}
		};

	} // namespace dev

} // namespace cuda