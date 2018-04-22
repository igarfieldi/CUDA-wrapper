#pragma once

#include "runtime/error.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <surface_types.h>
#include <cuda_surface_types.h>

namespace cuda {

	class surface {
	public:
		using descriptor_type = cudaSurfaceObject_t;

	private:
		descriptor_type m_obj;

	public:
		surface(cudaArray_t arr) : m_obj() {
			cudaResourceDesc desc;
			desc.resType = cudaResourceTypeArray;
			desc.res.array.array = arr;
			CUDA_TRY(cudaCreateSurfaceObject(&m_obj, &desc), "Failed to create surface object");
		}

		~surface() {
			// TODO: error handling
			cudaDestroySurfaceObject(m_obj);
		}

		descriptor_type descriptor() const noexcept {
			return m_obj;
		}
	};

} // namespace cuda