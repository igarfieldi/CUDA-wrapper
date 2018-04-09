#pragma once
#include <exception>
#include <initializer_list>
#include <mutex>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "error.hpp"
#include "properties.hpp"

namespace cuda {

	class device {
	public:
		using id_type = int;
		using flag_type = unsigned int;

	private:
		static std::vector<device> m_devices;
		static constexpr id_type default_id = 0;
		static const device *m_curr;
		static std::once_flag m_initialized;

		id_type m_id;

		device(id_type id) : m_id(id) {}

		static id_type get_curr_id() {
			id_type id;
			CUDA_TRY(cudaGetDevice(&id), "Failed to get current device ID");
			return id;
		}

		static void activate(id_type id) {
			CUDA_TRY(cudaSetDevice(id), "Failed to activate device");
		}

		static void init_devices() {
			// Add the available devices to our list
			id_type id_count = device::count();
			if (id_count < 1) {
				throw error::cuda_error(cudaErrorInitializationError, "No CUDA device available");
			}
			for (id_type id = 0; id < id_count; ++id) {
				//device::m_devices.emplace_back(id);
				device::m_devices.push_back(device(id));
			}
			device::m_devices.at(0).make_current();
		}

		struct scoped_device_switch {
		private:
			id_type m_prev;
			bool m_necessary;

		public:
			scoped_device_switch(const device &next) : m_prev(m_curr->id()), m_necessary(m_prev != next.id()) {
				if (m_necessary)
					activate(next.id());
			}

			~scoped_device_switch() {
				if (m_necessary)
					activate(m_prev);
			}
		};

	public:
		static id_type count() {
			id_type count;
			CUDA_TRY(cudaGetDeviceCount(&count), "Failed to get device count");
			return count;
		}

		static const device &current() {
			std::call_once(m_initialized, init_devices);

			return *m_curr;
		}

		static const device &get(id_type id) {
			std::call_once(m_initialized, init_devices);

			return m_devices.at(id);
		}

		id_type id() const {
			return m_id;
		}

		void make_current() const {
			m_curr = this;
			activate(m_curr->id());
		}

		scoped_device_switch make_current_in_scope() const {
			return scoped_device_switch(*this);
		}

		dev::properties get_properties() const {
			dev::properties props;
			CUDA_TRY(cudaGetDeviceProperties(&props, id()), "Failed to get device properties");
			return props;
		}

		void reset() const {
			make_current_in_scope();
			CUDA_TRY(cudaDeviceReset(), "Failed to reset device");
		}

		void synchronize() const {
			make_current_in_scope();
			CUDA_TRY(cudaDeviceSynchronize(), "Failed to synchronize device");
		}

		size_t get_limit(cudaLimit limit) const {
			size_t val;
			make_current_in_scope();
			CUDA_TRY(cudaDeviceGetLimit(&val, limit), "Failed to get device limit");
			return val;
		}

		void set_limit(cudaLimit limit, size_t val) const {
			scoped_device_switch s(*this);
			CUDA_TRY(cudaDeviceSetLimit(limit, val), "Failed to set device limit");
		}

		flag_type get_flags() const {
			unsigned int flags;
			make_current_in_scope();
			CUDA_TRY(cudaGetDeviceFlags(&flags), "Failed to get device flags");
			return flags;
		}

		void set_flags(flag_type flags) const {
			make_current_in_scope();
			CUDA_TRY(cudaSetDeviceFlags(flags), "Failed to set device flags");
		}

		bool operator==(const device &r) const {
			return id() == r.id();
		}

		bool operator!=(const device &r) const {
			return !((*this) == r);
		}
	};

	std::vector<device> device::m_devices;
	const device *device::m_curr = nullptr;
	std::once_flag device::m_initialized;

} // namespace cuda