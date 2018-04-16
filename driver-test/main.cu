#include "driver/init.hpp"
#include "driver/device.hpp"
#include "driver/version.hpp"
#include <iostream>

int main() {
	using namespace cuda::driver;
	initialize();

	std::cout << "Driver version: " << version() << std::endl;

	std::cout << "Devices: " << device::count() << std::endl;
	for (int i = 0; i < device::count(); ++i) {
		const auto &dev = device::get(i);
		std::cout << "\t" << dev.name() << " (" << dev.total_memory() / 1048576 << "MB)" << std::endl;
	}
}