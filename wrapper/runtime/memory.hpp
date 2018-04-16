#pragma once

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <iostream>
#include <cuda.h>
#include "error.hpp"
#include "stream.hpp"

namespace cuda {

	namespace host {

		template < class T >
		class pinned_buffer {
		private:
			using flag_type = unsigned int;

			T *m_buf;
			size_t m_size;
			flag_type m_flags;

		public:
			explicit pinned_buffer(size_t elems) : m_buf(nullptr), m_size(elems), m_flags(cudaHostAllocDefault) {
				CUDA_TRY(cudaHostAlloc(&m_buf, elems * sizeof(T), m_flags = cudaHostAllocDefault), "Failed to allocate pinned buffer");
			}

			pinned_buffer(size_t elems, bool portable, bool mapped, bool write_combined) : m_buf(nullptr),
					m_size(elems), m_flags(cudaHostAllocDefault) {
				m_flags |= (portable ? cudaHostAllocPortable : 0)
						| (mapped ? cudaHostAllocMapped : 0)
						| (write_combined ? cudaHostAllocWriteCombined : 0);
				CUDA_TRY(cudaHostAlloc(&m_buf, elems * sizeof(T), m_flags), "Failed to allocate pinned buffer");
			}

			~pinned_buffer() {
				if (m_buf != nullptr) {
					// TODO: no error checking because we can't throw exceptions
					cudaFreeHost(m_buf);
				}
			}

			T &operator[](size_t index) noexcept {
				return m_buf[index];
			}

			const T &operator[](size_t index) const noexcept {
				return m_buf[index];
			}

			size_t size() const noexcept {
				return m_size;
			}

			T *data() noexcept {
				return m_buf;
			}

			const T *data() const noexcept {
				return m_buf;
			}

			bool is_portable() const noexcept {
				return m_flags & cudaHostAllocPortable;
			}

			bool is_mapped() const noexcept {
				return m_flags & cudaHostAllocMapped;
			}

			bool is_write_combined() const noexcept {
				return m_flags & cudaHostAllocWriteCombined;
			}
		};

		template < class T >
		class gpu_buffer {
		protected:
			T *m_buf;
			size_t m_size;

		public:
			explicit gpu_buffer(size_t elems) : m_buf(nullptr), m_size(elems) {
				CUDA_TRY(cudaMalloc(&m_buf, m_size * sizeof(*m_buf)), "Failed to allocate GPU buffer");
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
			explicit input_buffer(size_t elems) : gpu_buffer(elems) {}

			void copy_h2d(const T *vals) {
				CUDA_TRY(cudaMemcpy(m_buf, vals, m_size * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
			}

			void copy_h2d(const pinned_buffer<T> &vals) {
				CUDA_TRY(cudaMemcpy(m_buf, vals.data(), std::min(m_size, vals.size()) * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
			}

			void copy_h2d(const T *vals, size_t len, size_t off) {
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy into GPU buffer");
				}
				CUDA_TRY(cudaMemcpy(&m_buf[off], vals, len * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
			}

			void copy_h2d(const pinned_buffer<T> &vals, size_t len, size_t off) {
				len = std::min(len, vals.size());
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy into GPU buffer");
				}
				CUDA_TRY(cudaMemcpy(&m_buf[off], vals.data(), len * sizeof(T), cudaMemcpyHostToDevice), "Failed to copy data to GPU buffer");
			}

			std::function<void(const stream &)> copy_h2d_async(const T *vals) {
				return [&](const stream &st) {
					cudaMemcpyAsync(m_buf, vals, m_size * sizeof(T), cudaMemcpyHostToDevice, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			std::function<void(const stream &)> copy_h2d_async(const pinned_buffer<T> &vals) {
				return [&](const stream &st) {
					cudaMemcpyAsync(m_buf, vals.data(), std::min(m_size, vals.size()) * sizeof(T), cudaMemcpyHostToDevice, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			std::function<void(const stream &)> copy_h2d_async(const pinned_buffer<T> &vals, size_t len, size_t off) {
				len = std::min(len, vals.size());
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy into GPU buffer");
				}
				return [&](const stream &st) {
					cudaMemcpyAsync(&m_buf[off], vals.data(), len * sizeof(T), cudaMemcpyHostToDevice, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			const T *data() const noexcept {
				return m_buf;
			}

			T *data() noexcept {
				return m_buf;
			}
		};

		template < class T >
		class output_buffer : public gpu_buffer<T> {
		public:
			explicit output_buffer(size_t elems) : gpu_buffer(elems) {}

			void copy_d2h(T *vals) const {
				CUDA_TRY(cudaMemcpy(vals, m_buf, m_size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
			}

			void copy_d2h(pinned_buffer<T> &vals) const {
				CUDA_TRY(cudaMemcpy(vals.data(), m_buf, std::min(m_size, vals.size()) * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
			}

			void copy_d2h(T *vals, size_t len, size_t off) const {
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy from GPU buffer");
				}
				CUDA_TRY(cudaMemcpy(vals, &m_buf[off], len, m_size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
			}

			void copy_d2h(pinned_buffer<T> &vals, size_t len, size_t off) const {
				len = std::min(m_size, vals.size());
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy from GPU buffer");
				}
				CUDA_TRY(cudaMemcpy(vals.data(), &m_buf[off], len, m_size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy data from GPU buffer");
			}

			std::function<void(const stream &)> copy_d2h_async(T *vals) const {
				return [&](const stream &st) {
					cudaMemcpyAsync(vals, m_buf, m_size * sizeof(T), cudaMemcpyDeviceToHost, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			std::function<void(const stream &)> copy_d2h_async(pinned_buffer<T> &vals) const {
				return [&](const stream &st) {
					cudaMemcpyAsync(vals.data(), m_buf, std::min(m_size, vals.size()) * sizeof(T), cudaMemcpyDeviceToHost, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			std::function<void(const stream &)> copy_d2h_async(T *vals, size_t len, size_t off) const {
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy from GPU buffer");
				}
				return [&](const stream &st) {
					cudaMemcpyAsync(vals, &m_buf[off], len * sizeof(T), cudaMemcpyDeviceToHost, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			std::function<void(const stream &)> copy_d2h_async(pinned_buffer<T> &vals, size_t len, size_t off) const {
				len = std::min(m_size, vals.size());
				if (off + len > m_size) {
					throw std::out_of_range("Too large buffer to copy from GPU buffer");
				}
				return [&](const stream &st) {
					cudaMemcpyAsync(vals.data(), &m_buf[off], len * sizeof(T), cudaMemcpyDeviceToHost, st.id());
					CUDA_TRY(cudaGetLastError(), "Async memcpy failed");
				};
			}

			const T *data() const noexcept {
				return m_buf;
			}

			T *data() noexcept {
				return m_buf;
			}
		};


	} // namespace host

} // namespace cuda