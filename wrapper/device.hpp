#pragma once
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "error.hpp"

namespace cuda {

	namespace dev {

		using id_type = int;
		using device_type = CUdevice;
		using flag_type = unsigned int;

		template < bool current >
		class device_base {
		public:
			static constexpr bool is_current = current;

		protected:
			id_type m_id;
			device_type m_hdl;

			static id_type get_curr_id() {
				id_type id;
				CUDA_TRY(cudaGetDevice(&id), "Failed to get current device ID");
				return id;
			}

			static void activate(id_type id) {
				CUDA_TRY(cudaSetDevice(id), "Failed to activate device");
			}

			static id_type get_handle(id_type id) {
				device_type hdl;
				CU_TRY(cuDeviceGet(&hdl, id), "Failed to get current device handle");
				return hdl;
			}

			/* This will fail if 'id' doesn't exist */
			explicit device_base(id_type id) : m_id(id), m_hdl(get_handle(id)) {}

			/* Class to temporarily switch devices if needed */
			template < bool current >
			struct scoped_device_switch;

			/* If we're handling the current device don't do anything */
			template <>
			struct scoped_device_switch<true> {
			public:
				scoped_device_switch(device_base<current> next) { (void)next; };
				~scoped_device_switch() = default;
			};

			/* If we're not handling the current device, save the old ID and switch it back */
			template <>
			struct scoped_device_switch<false> {
			private:
				id_type m_prev;
				bool m_necessary;

			public:
				scoped_device_switch(device_base<current> next) : m_prev(get_curr_id()), m_necessary(m_prev != next.id()) {
					if (m_necessary)
						activate(next.id());
				}

				~scoped_device_switch() {
					if (m_necessary)
						activate(m_prev);
				}
			};

		public:
			id_type id() const {
				return m_id;
			}

			device_type handle() const {
				return m_hdl;
			}

			cudaDeviceProp get_properties() const {
				cudaDeviceProp props;
				CUDA_TRY(cudaGetDeviceProperties(&props, id()), "Failed to get device properties");
				return props;
			}

			const char *get_name() const {
				static char buffer[MAX_NAME_LEN];
				auto res = cuDeviceGetName(buffer, sizeof(buffer), m_hdl);
				if (res != CUDA_SUCCESS) {
					/* If we can't get the name directly, try the properties */
					return get_properties().name;
				}
				return buffer;
			}

			void reset() const {
				scoped_device_switch<current> s(*this);
				CUDA_TRY(cudaDeviceReset(), "Failed to reset device");
			}

			void synchronize() const {
				scoped_device_switch<current> s(*this);
				CUDA_TRY(cudaDeviceSynchronize(), "Failed to synchronize device");
			}

			size_t get_limit(cudaLimit limit) const {
				scoped_device_switch<current> s(*this);
				size_t val;
				CUDA_TRY(cudaDeviceGetLimit(&val, limit), "Failed to get device limit");
				return val;
			}

			void set_limit(cudaLimit limit, size_t val) const {
				scoped_device_switch<current> s(*this);
				CUDA_TRY(cudaDeviceSetLimit(limit, val), "Failed to set device limit");
			}

			flag_type get_flags() const {
				scoped_device_switch<current> s(*this);
				unsigned int flags;
				CUDA_TRY(cudaGetDeviceFlags(&flags), "Failed to get device flags");
				return flags;
			}

			void set_flags(flag_type flags) const {
				scoped_device_switch<current> s(*this);
				CUDA_TRY(cudaSetDeviceFlags(flags), "Failed to set device flags");
			}

			template < bool C2 >
			bool operator==(const device_base<C2> &r) const {
				return id() == r.id();
			}

			template < bool C2 >
			bool operator!=(const device_base<C2> &r) const {
				return !((*this) == r);
			}
		};

		template < bool current = true >
		class device;

		/* This specialization is non-owning, and thus whenever an instance is needed one may create it */
		template < >
		class device<false> : public device_base<false> {
		public:
			explicit device(device::id_type id) : device_base(id) {}
		};

		/* This device specialization is owning: no duplicate instance may exist */
		template < >
		class device<true> : public device_base<true> {
		private:
			explicit device() : device_base(get_curr_id()) {}

		public:
			static device<true> &get() {
				static device<true> instance;
				return instance;
			}

			device<true> &make_current(const device<false> &next) {
				if (device_base<true>::operator!=(next)) {
					activate(next.id());
					m_id = next.id();
				}
				return *this;
			}
		};

		id_type count() {
			id_type count;
			CUDA_TRY(cudaGetDeviceCount(&count), "Failed to get device count");
			return count;
		}

	} // namespace dev

} // namespace cuda