#pragma once

#include <exception>
#include <initializer_list>
#include <mutex>
#include <string>
#include <tuple>
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
		static constexpr id_type default_id = 0;
		id_type m_id;

		device(id_type id) : m_id(id) {}

		~device() {
			// TODO: deal with error
			id_type old;
			cudaGetDevice(&old);
			if (old != m_id) {
				cudaSetDevice(m_id);
			}
			cudaDeviceReset();
			if (old != m_id) {
				cudaSetDevice(old);
			}
		}

		// "Fixed" vector class to store our devices
		template < class T >
		class fixed_vector {
		private:
			T *m_data;
			size_t m_size;
			size_t m_capacity;

		public:
			explicit fixed_vector(size_t size) : m_data(reinterpret_cast<T*>(::operator new(size * sizeof(T)))),
				m_capacity(size), m_size() {}

			fixed_vector(const fixed_vector &) = delete;
			fixed_vector(fixed_vector &&) = default;
			fixed_vector &operator=(const fixed_vector &) = delete;
			fixed_vector &operator=(fixed_vector &&) = default;

			~fixed_vector() {
				// Destroy the elements in reverse order
				for (size_t i = 0; i < m_size; ++i) {
					m_data[m_size - i - 1].~T();
				}
				::operator delete(m_data);
			}

			template < class... Args >
			T &emplace_back(Args&&... args) {
				if (m_size == m_capacity) {
					throw std::out_of_range("Fixed vector is full");
				}
				new (&m_data[m_size]) T(std::forward<Args>(args)...);
				return m_data[m_size++];
			}

			T &at(size_t index) {
				if (index >= m_size) {
					throw std::out_of_range("Invalid index");
				}
				return m_data[index];
			}

			const T &at(size_t index) const {
				if (index >= m_size) {
					throw std::out_of_range("Invalid index");
				}
				return m_data[index];
			}

			T &operator[](size_t index) noexcept {
				return m_data[index];
			}

			const T &operator[](size_t index) const noexcept {
				return m_data[index];
			}
		};

		static fixed_vector<device> &m_devices() {
			static fixed_vector<device> dev(device::count());
			return dev;
		}

		static const device *&m_curr() {
			static thread_local const device *curr;
			return curr;
		}

		static std::once_flag &m_inited() {
			static std::once_flag flag;
			return flag;
		}

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
			// We can't use vector or array because we don't have a copy constructor and no default constructor either
			// Thus we'll emplace 
			id_type id_count = device::count();
			if (id_count < 1) {
				throw error::cuda_error(cudaErrorInitializationError, "No CUDA device available");
			}
			for (id_type id = 0; id < id_count; ++id) {
				device::m_devices().emplace_back(id);
			}
			device::m_devices().at(0).make_current();
		}

		struct scoped_device_switch {
		private:
			id_type m_prev;
			bool m_necessary;

		public:
			scoped_device_switch(const device &next) : m_prev(m_curr()->id()), m_necessary(m_prev != next.id()) {
				if (m_necessary)
					activate(next.id());
			}

			~scoped_device_switch() {
				if (m_necessary)
					activate(m_prev);
			}
		};

	public:
		device(const device &) = delete;
		device(device &&) = delete;
		device &operator=(const device &) = delete;
		device &operator=(device &&) = delete;

		static id_type count() {
			id_type count;
			CUDA_TRY(cudaGetDeviceCount(&count), "Failed to get device count");
			return count;
		}

		static const device &current() {
			std::call_once(m_inited(), init_devices);

			return *m_curr();
		}

		static const device &get(id_type id) {
			std::call_once(m_inited(), init_devices);

			return m_devices().at(id);
		}

		id_type id() const {
			return m_id;
		}

		void make_current() const {
			m_curr() = this;
			activate(m_curr()->id());
		}

		scoped_device_switch make_current_in_scope() const {
			return scoped_device_switch(*this);
		}

		dev::properties get_properties() const {
			dev::properties props;
			CUDA_TRY(cudaGetDeviceProperties(&props, id()), "Failed to get device properties");
			return props;
		}

		std::pair<size_t, size_t> memory() const {
			size_t free, total;
			make_current_in_scope();
			CUDA_TRY(cudaMemGetInfo(&free, &total), "Failed to get device memory");
			return { free, total };
		}

		size_t free_mem() const {
			return this->memory().first;
		}

		size_t total_mem() const {
			return this->memory().second;
		}

		size_t used_mem() const {
			return this->total_mem() - this->free_mem();
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

} // namespace cuda