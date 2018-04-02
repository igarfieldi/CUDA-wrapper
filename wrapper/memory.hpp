#pragma once

#include <memory>
#include <exception>
#include <cuda.h>
#include "error.hpp"

namespace cuda {

	namespace mem {

		namespace host {

			template < class T, bool owning = true >
			class buffer;

			template < class T >
			class buffer<T, true> {
			private:
				std::unique_ptr<T[]> m_buf;
				size_t m_size;

			public:
				buffer(size_t elems) noexcept : m_buf(new T[elems]), m_size(elems) {
				}

				~buffer() = default;

				const T *data() const noexcept {
					return m_buf.get();
				}
			};

			template < class T >
			class buffer<T, false> {
			private:
				T *m_buf;
				size_t m_size;

			public:
				buffer(T *array, size_t elems) noexcept : m_buf(array), m_size(elems) {
				}
				~buffer() = default;

				const T *data() const noexcept {
					return m_buf;
				}
			};

			template < class T >
			class gpu_buffer {
			protected:
				T *m_buf;
				size_t m_size;

			public:
				gpu_buffer(size_t elems) : m_buf(nullptr), m_size(elems) {
					CUDA_TRY(cudaMalloc(&m_buf, m_size * sizeof(T)), "Failed to allocate GPU buffer");
				}

				~gpu_buffer() noexcept {
					if (m_buf != nullptr)
						cudaFree(m_buf);
				}

				template < class V >
				gpu_buffer(const gpu_buffer<V> &) = delete;
				template < class V >
				gpu_buffer &operator=(const gpu_buffer<V> &) = delete;
			};

			template < class T >
			class input_buffer : public gpu_buffer<T> {
			public:
				input_buffer(size_t elems) : gpu_buffer(elems) {}

				void copy_from(const T *vals) {
					CUDA_TRY(cudaMemcpy(m_buf, vals, m_size * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
				}

				void copy_from(const T *vals, size_t len, size_t off) {
					if (off + len > m_size) {
						throw std::out_of_range("Too large buffer to copy into GPU buffer");
					}
					CUDA_TRY(cudaMemcpy(&m_buf[off], vals, len * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
				}

				const T *data() const {
					return m_buf;
				}
			};

			template < class T >
			class output_buffer : public gpu_buffer<T> {
			public:
				output_buffer(size_t elems) : gpu_buffer(elems) {}

				void copy_into(T *vals) {
					CUDA_TRY(cudaMemcpy(vals, m_buf, m_size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
				}

				void copy_into(T *vals, size_t len, size_t off) {
					if (off + len > m_size) {
						throw std::out_of_range("Too large buffer to copy from GPU buffer");
					}
					CUDA_TRY(cudaMemcpy(vals, &m_buf[off], len, m_size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
				}

				const T *data() const noexcept {
					return m_buf;
				}

				T *data() noexcept {
					return m_buf;
				}
			};

		} // namespace host

	} // namespace mem

} // namespace cuda