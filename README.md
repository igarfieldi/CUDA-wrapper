# CUDA C++ wrapper

This project has the goal of making C++-developer's lives easier when dealing with CUDA. Currently WIP, it should abstract CUDA into classes and objects with lifetimes.
Instead of error codes it uses exceptions. It currently uses C++11/14 since CUDA 9.1 does not get along with latest Visual Studio.
As of yet it lacks tests other than surface-level sample usage.

## Getting Started

Clone the repository and run CMake:
```
git clone https://github.com/igarfieldi/CUDA-wrapper.git
mkdir CUDA-wrapper/build
cd CUDA-wrapper/build
cmake ..
```

### Prerequisites

You will need CMake 3.9 or higher as well as the CUDA SDK (tested with >= 8.0).

## Contributing

The project is currently developed solo, feature requests or suggestions for improvement are welcome.

## Authors

* **Florian Bethe** - [IGarFieldI](https://github.com/igarfieldi)

## License

This project does not currently have a license - I'll think about something
