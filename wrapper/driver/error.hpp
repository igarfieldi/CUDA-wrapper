#pragma once

#include <exception>
#include <string>
#include <cuda.h>

namespace cuda {

	namespace driver {

		class base_error : public std::runtime_error {
		public:
			base_error(const std::string &msg) : std::runtime_error(msg) {}
		};

		class driver_error : public base_error {
		protected:
			static const char *get_error_name(CUresult type) {
				const char *ptr = nullptr;
				(void) cuGetErrorName(type, &ptr);
				return ptr;
			}

			static const char *get_error_string(CUresult type) {
				const char *ptr = nullptr;
				(void) cuGetErrorString(type, &ptr);
				return ptr;
			}

		public:
			driver_error(CUresult type, const std::string &msg) : base_error(msg + ": " + get_error_string(type) + "(" + get_error_name(type) + ")") {}
		};

		void attempt(CUresult res, const char *msg) {
			if (res != CUDA_SUCCESS) {
				throw driver_error(res, msg);
			}
		}

	} // namespace driver

} // namespace cuda