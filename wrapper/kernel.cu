#pragma once

#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include "error.hpp"
#include "stream.hpp"

namespace cuda {

	namespace launch {

		struct binary_version_tag {};
		struct ptx_version_tag {};

		using binary_version = cuda::common::version<binary_version_tag>;
		using ptx_version = cuda::common::version<ptx_version_tag>;

	} // namespace launch

	// This is for VC++ which can only explicitly specialize templates at namespace scope
	namespace common {

		template < >
		inline version<cuda::launch::binary_version_tag>::version(int combined) : major(combined / 10), minor(combined % 10) {
		}

		template < >
		inline version<cuda::launch::ptx_version_tag>::version(int combined) : major(combined / 10), minor(combined % 10) {
		}

	} // namespace common

	namespace launch {

		using attribute_type = int;

		struct attributes : public cudaFuncAttributes {
			binary_version binary_version() const {
				return launch::binary_version(this->binaryVersion);
			}

			ptx_version ptx_version() const {
				return launch::ptx_version(this->ptxVersion);
			}
		};

		template < class R, class... Params >
		class kernel {
		private:
			R(*m_func)(Params...);

		public:
			kernel(R(*func)(Params...)) : m_func(func) {
				if (func == nullptr) {
					throw cuda::error::cuda_error(cudaErrorInvalidDeviceFunction, "Failed to take CUDA kernel pointer");
				}
			}

			attributes get_attributes() const {
				attributes attr;
				CUDA_TRY(cudaFuncGetAttributes(&attr, m_func), "Failed to get function attributes");
				return attr;
			}

			void set_attributes(const cudaFuncAttribute &attr, attribute_type val) const {
				CUDA_TRY(cudaFuncSetAttribute(m_func, attr, val), "Failed to set function attributes");
			}

			void set_cache_config(cudaFuncCache config) const {
				CUDA_TRY(cudaFuncSetCacheConfig(m_func, config), "Failed to set function cache configuration");
			}

			void launch(dim3 grid, dim3 block, Params... parameters) const {
				// Launch the kernel
				m_func<<<grid, block>>>(parameters...);
				// Check for errors
				CUDA_TRY(cudaGetLastError(), "Kernel launch failed");
			}

			std::function<void(const stream &)> stream_launch(dim3 grid, dim3 block, Params... parameters) const {
				// Defer kernel launch
				return [=](const stream &st) {
					m_func<<<grid, block, 0, st.id()>> > (parameters...);
					// Check for errors
					CUDA_TRY(cudaGetLastError(), "Kernel launch failed");
				};
			}
		};

		/* To alleviate the issue of lacking constructor template deduction */
		template < class R, class... Params >
		kernel<R, Params...> make_kernel(R(*func)(Params...)) {
			return kernel<R, Params...>(func);
		}

	} // namespace launch

} // namespace cuda