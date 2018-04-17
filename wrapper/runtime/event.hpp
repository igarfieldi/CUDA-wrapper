#pragma once

#include "device.hpp"
#include "error.hpp"
#include "stream.hpp"
#include <cuda.h>

namespace cuda {

	// TODO: give it a device reference?
	class event {
	public:
		using event_type = cudaEvent_t;
		using flag_type = unsigned int;

	private:
		event_type m_event;
		flag_type m_flags;
		const device &m_device;

	public:
		// TODO: are empty events really so nice?
		explicit event(const device &dev) : m_event(), m_flags(cudaEventDefault), m_device(dev) {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaEventCreate(&m_event), "Failed to create event");
		}

		explicit event(const stream &stream) : m_event(), m_flags(cudaEventDefault), m_device(stream.device()) {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaEventCreate(&m_event), "Failed to create event");
			this->record(stream);
		}

		event(const stream &stream, bool block_sync, bool timing = true) : m_event(), m_flags(cudaEventDefault), m_device(stream.device()) {
			// TODO: allow for interprocess events
			m_flags |= (timing ? 0 : cudaEventDisableTiming)
				| (block_sync ? cudaEventBlockingSync : 0);
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaEventCreateWithFlags(&m_event, m_flags), "Failed to create event");
			this->record(stream);
		}

		~event() {
			// TODO: how to deal with error?
			auto scope = m_device.make_current_in_scope();
			cudaEventDestroy(m_event);
		}

		bool is_timing() const {
			return !(m_flags & cudaEventDisableTiming);
		}

		bool blocks_on_sync() const {
			return m_flags & cudaEventBlockingSync;
		}

		bool is_complete() const {
			auto scope = m_device.make_current_in_scope();
			auto res = cudaEventQuery(m_event);
			switch (res) {
			case cudaSuccess:
				return true;
			case cudaErrorNotReady:
				return false;
			default:
				throw error::cuda_error(res, "Failed to query event status");
			}
		}

		void record(const stream &s) const {
			CUDA_TRY(cudaEventRecord(m_event, s.id()), "Failed to record stream event");
		}

		void sychronize() const {
			auto scope = m_device.make_current_in_scope();
			CUDA_TRY(cudaEventSynchronize(m_event), "Failed to synchronize with event");
		}

		float elapsed_time(const event &evt) const {
			float ms;
			CUDA_TRY(cudaEventElapsedTime(&ms, evt.m_event, m_event), "Failed to obtain elapsed time");
			return ms;
		}
	};

} // namespace cuda