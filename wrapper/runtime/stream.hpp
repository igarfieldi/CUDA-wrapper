#pragma once

#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device.hpp"
#include "error.hpp"

namespace cuda {

	class stream {
	public:
		using id_type = cudaStream_t;
		using flag_type = unsigned int;
		using priority_type = int;
	
	private:
		static constexpr id_type default_id = nullptr;

		flag_type get_flags() const {
			flag_type flags;
			CUDA_TRY(cudaStreamGetFlags(m_id, &flags), "Failed to get stream flags");
			return flags;
		}

		cudaStream_t m_id;
		const device &m_device;

		// Creates the default stream - doesn't need to be actually created
		explicit stream() : m_id(stream::default_id), m_device(device::current()) {}

	public:
		explicit stream(const device &dev) : m_id(), m_device(dev) {
			auto scope = dev.make_current_in_scope();
			CUDA_TRY(cudaStreamCreate(&m_id), "Failed to create stream");
		}

		stream(const device &dev, bool non_blocking) : m_id(), m_device(dev) {
			auto scope = dev.make_current_in_scope();
			CUDA_TRY(cudaStreamCreateWithFlags(&m_id, non_blocking ? cudaStreamNonBlocking : cudaStreamDefault),
						"Failed to create stream with flags");
		}

		stream(const device &dev, bool non_blocking, priority_type priority) : m_id(), m_device(dev) {
			auto scope = dev.make_current_in_scope();
			CUDA_TRY(cudaStreamCreateWithPriority(&m_id, non_blocking ? cudaStreamNonBlocking : cudaStreamDefault,
						priority), "Failed to create stream with flags and priority");
		}

		stream(const stream &) = delete;
		stream(stream &&) = default;
		stream &operator=(const stream &) = delete;
		stream &operator=(stream &&) = default;

		~stream() {
			if (m_id != stream::default_id) {
				auto scope = m_device.make_current_in_scope();
				/* TODO: we ignore failure for now because we can't throw an exception */
				(void)cudaStreamDestroy(m_id);
			}
		}

		static const stream &default_stream() {
			static thread_local stream def;
			return def;
		}

		id_type id() const {
			return m_id;
		}

		const device &device() const {
			/* Default stream doesn't have a bound device */
			if (m_id == stream::default_id) {
				return device::current();
			}
			return m_device;
		}

		void enqueue(std::function<void(const stream &)> op) const {
			auto scope = this->device().make_current_in_scope();
			op(*this);
		}

		void synchronize() const {
			auto scope = this->device().make_current_in_scope();
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