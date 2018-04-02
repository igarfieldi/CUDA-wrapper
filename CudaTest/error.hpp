#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda {

	namespace error {

		using error_type = cudaError_t;
		using result_type = CUresult;

		class cuda_error : public std::runtime_error {
		public:
			const error_type type;

			cuda_error(error_type type, const std::string &msg) : std::runtime_error(msg + ": " + cudaGetErrorString(type)), type(type) {
			}

			cuda_error(error_type type) : cuda_error(type, "CUDA runtime error") {}
		};

		class cu_error : public std::runtime_error {
		private:
			static const char *to_string(result_type type) {
				const char *res;
				cuGetErrorString(type, &res);
				return res;
			};
		public:
			const result_type type;

			cu_error(result_type type, const std::string &msg) : std::runtime_error(msg + ": " + to_string(type)), type(type) {}

			cu_error(result_type type) : cu_error(type, "CUDA driver error") {}
		};

#define CUDA_TRY(x, msg)									\
		do {												\
			auto err = x;									\
			if(err != cudaSuccess)							\
				throw cuda::error::cuda_error(err, msg);	\
		} while(0)

#define CU_TRY(x, msg)									\
		do {											\
			auto err = x;								\
			if(err != CUDA_SUCCESS)						\
				throw cuda::error::cu_error(err, msg);	\
		} while(0)

	} // namespace error

} // namespace cuda
