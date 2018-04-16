#pragma once

#include <iterator>
#include <cstddef>

namespace cuda {

	namespace util {

		template < class T >
		class fixed_vector {
		public:
			using data_type = T;
			using size_type = size_t;
			using iterator = T *;
			using const_iterator = const T *;
			using reverse_iterator = std::reverse_iterator<iterator>;
			using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		private:
			data_type *m_data;
			size_type m_capacity;
			size_type m_size;

		public:
			explicit fixed_vector(size_type size) : m_data(reinterpret_cast<T*>(::operator new(size * sizeof(T)))),
						m_capacity(size), m_size() {}
			fixed_vector(const fixed_vector &) = delete;
			fixed_vector(fixed_vector &&) = default;
			fixed_vector &operator=(const fixed_vector &) = delete;
			fixed_vector &operator=(fixed_vector &&) = default;

			~fixed_vector() {
				for (auto it = rbegin(); it != rend(); ++it) {
					it->~T();
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

			T &at(size_type index) {
				if (index >= m_size) {
					throw std::out_of_range("Invalid index");
				}
				return m_data[index];
			}

			const T &at(size_type index) const {
				if (index >= m_size) {
					throw std::out_of_range("Invalid index");
				}
				return m_data[index];
			}

			T &operator[](size_type index) noexcept {
				return m_data[index];
			}

			const T &operator[](size_type index) const noexcept {
				return m_data[index];
			}

			iterator begin() {
				return m_data;
			}

			iterator end() {
				return &m_data[m_size];
			}

			const_iterator cbegin() const {
				return &m_data[0];
			}

			const_iterator cend() const {
				return &m_data[m_size];
			}

			reverse_iterator rbegin() {
				return reverse_iterator(begin());
			}

			reverse_iterator rend() {
				return reverse_iterator(end());
			}

			reverse_iterator crbegin() {
				return reverse_iterator(begin());
			}

			reverse_iterator crend() {
				return reverse_iterator(end());
			}

			size_type size() const noexcept {
				return m_size;
			}

			size_type capacity() const noexcept {
				return m_capacity;
			}
		};

	} // namespace cuda

} // namespace cuda