#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "error.hpp"

namespace cuda {

	namespace launch {

		using attribute_type = int;

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

			cudaFuncAttributes get_attributes() const {
				cudaFuncAttributes attr;
				CUDA_TRY(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(m_func)), "Failed to get function attributes");
				return attr;
			}

			void set_attributes(const cudaFuncAttribute &attr, attribute_type val) const {
				CUDA_TRY(cudaFuncSetAttribute(reinterpret_cast<const void *>(m_func), attr, val), "Failed to set function attributes");
			}

			void launch(dim3 grid, dim3 block, Params... parameters) {
				// Launch the kernel
				m_func<<<grid, block>>>(parameters...);
				// Check for errors
				CUDA_TRY(cudaGetLastError(), "Kernel launch failed");
			}
		};

		/* To alleviate the issue of lacking constructor template deduction */
		template < class R, class... Params >
		kernel<R, Params...> make_kernel(R(*func)(Params...)) {
			return kernel<R, Params...>(func);
		}

	} // namespace launch

} // namespace cuda