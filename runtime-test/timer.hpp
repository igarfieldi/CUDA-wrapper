#pragma once
#include <chrono>

namespace cpputils {

	template < class C = std::chrono::high_resolution_clock >
	class Timer {
	public:
		using ClockType = C;

	private:
		typename ClockType::time_point curr;

	public:
		Timer() : curr(ClockType::now()) {
		}

		void reset() {
			curr = ClockType::now();
		}

		template < class T = std::chrono::milliseconds >
		typename T::rep duration() const {
			return std::chrono::duration_cast<T>(ClockType::now() - curr).count();
		}

		template < class T = std::chrono::milliseconds >
		T durationRaw() const {
			return std::chrono::duration_cast<T>(ClockType::now() - curr);
		}
	};

} // namespace cpputils