#pragma once

#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device.hpp"
#include "error.hpp"

namespace cuda {


	class stream {
	public:
		using flag_type = unsigned int;
		using priority_type = int;
	
	private:
		cudaStream_t m_id;
		device<false> m_device;

		flag_type get_flags() const {
			flag_type flags;
			CUDA_TRY(cudaStreamGetFlags(m_id, &flags), "Failed to get stream flags");
			return flags;
		}

	public:
		stream() : m_id(), m_device() {
			/* Don't need to make device current */
			CUDA_TRY(cudaStreamCreate(&m_id), "Failed to create stream");
		}

		explicit stream(bool non_blocking) : m_id(), m_device() {
			CUDA_TRY(cudaStreamCreateWithFlags(&m_id, non_blocking ? cudaStreamNonBlocking : cudaStreamDefault),
						"Failed to create stream with flags");
		}

		stream(bool non_blocking, priority_type priority) : m_id(), m_device() {
			CUDA_TRY(cudaStreamCreateWithPriority(&m_id, non_blocking ? cudaStreamNonBlocking : cudaStreamDefault, priority),
						"Failed to create stream with flags and priority");
		}

		explicit stream(device<false> dev) : m_id(), m_device(dev) {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaStreamCreate(&m_id), "Failed to create stream");
		}

		stream(device<false> dev, flag_type flags) : m_id(), m_device(dev) {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaStreamCreateWithFlags(&m_id, flags), "Failed to create stream with flags");
		}

		stream(device<false> dev, flag_type flags, priority_type priority) : m_id(), m_device(dev) {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaStreamCreateWithPriority(&m_id, flags, priority), "Failed to create stream with flags and priority");
		}

		~stream() {
			auto scope = m_device.make_current_in_scope();
			/* TODO: we ignore failure for now because we can't throw an exception */
			(void) cudaStreamDestroy(m_id);
		}

		cudaStream_t id() const {
			return m_id;
		}

		void enqueue(std::function<void(const stream &)> op) const {
			auto scope = m_device.make_current_in_scope();
			op(*this);
		}

		void synchronize() const {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaStreamSynchronize(m_id), "Failed to synchronize stream");
		}

		bool is_empty() const {
			cudaError_t status = cudaStreamQuery(m_id);
			switch (status) {
			case cudaSuccess:
				return true;
			case cudaErrorNotReady:
				return false;
			default:
				throw cuda::error::cuda_error(status, "Failed to query stream status");
			}
		}

		bool syncs_with_default() const {
			return !(get_flags() & cudaStreamNonBlocking);
		}

		priority_type priority() const {
			priority_type p;
			CUDA_TRY(cudaStreamGetPriority(m_id, &p), "Failed to get stream priority");
			return p;
		}
	};

} // namespace cuda