#pragma once

#include <ostream>
#include <string>

namespace cuda {

	namespace common {

		template < class T >
		struct version {
			int major;
			int minor;

			version(int major, int minor) : major(major), minor(minor) {}

			explicit version(int combined);

			int combined() const {
				return major * 10 + minor;
			}

			bool is_valid() const {
				return (major > 0) && (major < 9999) && (minor > 0) && (minor < 9999);
			}

			bool operator==(const version &c) const { return major == c.major && minor == c.minor; };
			bool operator!=(const version &c) const { return major != c.major || minor != c.minor; };
			bool operator>(const version &c) const { return major > c.major || (major == c.major && minor > c.minor); };
			bool operator>=(const version &c) const { return major > c.major || (major == c.major && minor >= c.minor); };
			bool operator<(const version &c) const { return major < c.major || (major == c.major && minor < c.minor); };
			bool operator<=(const version &c) const { return major < c.major || (major == c.major && minor <= c.minor); };
		};

		template < class T >
		std::ostream &operator<<(std::ostream &stream, const version<T> &c) {
			return (stream << std::to_string(c.major) << '.' << std::to_string(c.minor));
		}

	} // namespace common

} // namespace cuda