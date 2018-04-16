#pragma once

#include <mutex>
#include <cuda.h>
#include "error.hpp"

namespace cuda {

	namespace driver {

		class device {
		public:
			using id_type = int;
			using handle_type = CUdevice;
			using attribute_type = CUdevice_attribute;

		private:
			static constexpr size_t MAX_NAME_LEN = 256;

			// "Fixed" vector class to store our devices
			template < class T >
			class fixed_vector {
			private:
				T * m_data;
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
				static const device *curr;
				return curr;
			}

			static std::once_flag &m_inited() {
				static std::once_flag flag;
				return flag;
			}

			static void init_devices() {
				// Add the available devices to our list
				// We can't use vector or array because we don't have a copy constructor and no default constructor either
				// Thus we'll emplace 
				id_type id_count = device::count();
				if (id_count < 1) {
					throw driver_error(CUDA_ERROR_NO_DEVICE, "No CUDA device available");
				}
				for (id_type id = 0; id < id_count; ++id) {
					device::m_devices().emplace_back(id);
				}
			}

			id_type m_id;
			handle_type m_hdl;

			device(id_type id) : m_id(id), m_hdl() {
				attempt(cuDeviceGet(&m_hdl, m_id), "Failed to get device handle");
			}

		public:
			static const device &current() {
				std::call_once(m_inited(), init_devices);

				return *m_curr();
			}

			static const device &get(id_type id) {
				std::call_once(m_inited(), init_devices);

				return m_devices().at(id);
			}

			static id_type count() {
				id_type count;
				attempt(cuDeviceGetCount(&count), "Failed to get device count");
				return count;
			}

			id_type id() const {
				return m_id;
			}

			handle_type hdl() const {
				return m_hdl;
			}

			size_t total_memory() const {
				size_t bytes;
				attempt(cuDeviceTotalMem(&bytes, m_hdl), "Failed to get device total memory");
				return bytes;
			}

			std::string name() const {
				static char buffer[MAX_NAME_LEN];
				attempt(cuDeviceGetName(buffer, sizeof(buffer), m_hdl), "Failed to get device name");
				return buffer;
			}

			int get_attribute(attribute_type attr) const {
				int val;
				attempt(cuDeviceGetAttribute(&val, attr, m_hdl), "Failed to get device attribute");
				return val;
			}
		};

	} // namespace driver

} // namespace cuda