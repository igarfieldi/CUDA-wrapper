#include "runtime/error.hpp"
#include "runtime/event.hpp"
#include "runtime/kernel.cu"
#include "runtime/memory.hpp"
#include "runtime/version.hpp"
#include "runtime/device.hpp"
#include "timer.hpp"
#include "surface.hpp"
#include <iostream>
#include <functional>
#include <mutex>
#include <thread>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "oglwrap.h"
#include "shapes/rectangle_shape.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <cuComplex.h>
#include <cuda_surface_types.h>
#include <thrust/complex.h>

template < class complex >
__device__ int pixel_dwell(const unsigned int w, const unsigned int h, const complex cmin,
		const complex cmax, const unsigned int x, const unsigned int y, const unsigned int MAX_DWELL) {
	using fp_type = typename complex::value_type;
	const complex dc = cmax - cmin;
	const fp_type fx = (fp_type)x / (fp_type)w, fy = (fp_type)y / (fp_type)h;
	const complex c = cmin + complex(fx * dc.real(), fy * dc.imag());
	unsigned int dwell = 0;
	complex z = c;
	while (dwell < MAX_DWELL && abs(z) < 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}

__device__ void color_dwell_surface(cudaSurfaceObject_t surf, const unsigned int x, const unsigned int y,
	const int dwell, const int MAX_DWELL) {
	const unsigned int cutoff_dwell = MAX_DWELL / 24;
	if (dwell == MAX_DWELL) {
		surf2Dwrite(make_uchar4(0u, 0u, 0u, 255u), surf, x * sizeof(uchar4), y);
	} else if (dwell <= cutoff_dwell) {
		unsigned int red = 0u;
		unsigned int green = 0u;
		unsigned int blue = 128u + dwell * 127u / cutoff_dwell;
		surf2Dwrite(make_uchar4(red, green, blue, 255u), surf, x * sizeof(uchar4), y);
	}
	else {
		unsigned int red = 255u * (dwell - cutoff_dwell) / (MAX_DWELL - cutoff_dwell);
		unsigned int green = 255u * (dwell - cutoff_dwell) / (MAX_DWELL - cutoff_dwell);
		unsigned int blue = 255u;
		surf2Dwrite(make_uchar4(red, green, blue, 255u), surf, x * sizeof(uchar4), y);
	}
}

__device__ void color_dwell(uint32_t *pixels, const unsigned int x, const unsigned int y, const unsigned int w,
	const int dwell, const int MAX_DWELL) {
	const unsigned int cutoff_dwell = MAX_DWELL / 24;
	if (dwell == MAX_DWELL) {
		pixels[x + y*w] = (255u << 24u);
	}
	else if (dwell <= cutoff_dwell) {
		unsigned int red = 0u;
		unsigned int green = 0u;
		unsigned int blue = 128u + dwell * 127u / cutoff_dwell;
		pixels[x + y*w] = (255u << 24u) | (blue << 16u) | (green << 8u) | red;
	}
	else {
		unsigned int red = 255u * (dwell - cutoff_dwell) / (MAX_DWELL - cutoff_dwell);
		unsigned int green = 255u * (dwell - cutoff_dwell) / (MAX_DWELL - cutoff_dwell);
		unsigned int blue = 255u;
		pixels[x + y*w] = (255u << 24u) | (blue << 16u) | (green << 8u) | red;
	}
}

template < class complex >
__global__ void mandelbrot_k_surface(cudaSurfaceObject_t surf, const unsigned int w, const unsigned int h,
	const complex cmin, const complex cmax, const unsigned int max_dwell) {
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h) {
		color_dwell_surface(surf, x, y, pixel_dwell(w, h, cmin, cmax, x, y, max_dwell), max_dwell);
	}
}

template < class complex >
__global__ void mandelbrot_k(uint32_t *pixels, const unsigned int w, const unsigned int h,
	const complex cmin, const complex cmax, const unsigned int max_dwell) {
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < w && y < h) {
		color_dwell(pixels, x, y, w, pixel_dwell(w, h, cmin, cmax, x, y, max_dwell), max_dwell);
	}
}


GLFWwindow *create_window(const char *title, int w, int h) {
	static std::once_flag glew_flag, glfw_flag;
	std::call_once(glfw_flag, []() {
		if (!glfwInit()) {
			throw std::runtime_error("Failed to initialize GLFW");
		}
		std::atexit(&glfwTerminate);
	});

	glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

	auto window = glfwCreateWindow(w, h, title, nullptr, NULL);

	if (!window) {
		throw std::runtime_error("Failed to create window");
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	glfwMakeContextCurrent(window);

	std::call_once(glew_flag, []() {
		auto status = glewInit();
		if (status != GLEW_OK) {
			throw std::runtime_error("Failed to initialize GLEW");
		}
	});

	return window;
}

#define USE_SURFACE

int main() {
	try {
		using fp_type = double;
		using complex = thrust::complex<fp_type>;

		constexpr unsigned int width = 640u;
		constexpr unsigned int height = 640u;
		const dim3 blocksize(32, 32);
		const dim3 gridsize(width / blocksize.x, height / blocksize.y);

		constexpr unsigned int iterations = 5000u;
		constexpr fp_type theta = 0.85f;
		complex cmin(-2.5f, -1.5f);
		complex cmax(1.0f, 1.5f);
		const complex zoom(0.25f*cosf(theta) - 1.0f, 0.25f*sinf(theta));
		constexpr fp_type zoom_per_frame = 0.00025f;
		constexpr bool time_based = true;

		auto window = create_window("Test", width, height);

		gl::RectangleShape rectangle_shape({ gl::RectangleShape::kPosition, gl::RectangleShape::kTexCoord });
		gl::Texture2D tex;
		gl::Program prog;
		gl::ShaderSource vs_source;
		vs_source.set_source(R"""(
		  #version 330 core
		  in vec2 pos;
		  in vec2 inTexCoord;
		  out vec2 texCoord;
		  void main() {
			texCoord = inTexCoord;
			// Shrink the full screen rectangle to a smaller size
			gl_Position = vec4(pos.x, pos.y, 0, 1);
		  })""");
		vs_source.set_source_file("example_shader.vert");
		gl::Shader vs(gl::kVertexShader, vs_source);

		gl::ShaderSource fs_source;
		fs_source.set_source(R"""(
		  #version 330 core
		  in vec2 texCoord;
		  uniform sampler2D tex;
		  out vec4 fragColor;
		  void main() {
			fragColor = texture(tex, vec2(texCoord.x, 1-texCoord.y));
		  })""");
		fs_source.set_source_file("example_shader.frag");
		gl::Shader fs(gl::kFragmentShader, fs_source);

		// Create a shader program
		prog.attachShader(vs);
		prog.attachShader(fs);
		prog.link();
		gl::Use(prog);

		// Bind the attribute positions to the locations that the RectangleShape uses
		(prog | "pos").bindLocation(gl::RectangleShape::kPosition);
		(prog | "inTexCoord").bindLocation(gl::RectangleShape::kTexCoord);

		// Set the texture uniform
		gl::UniformSampler(prog, "tex") = 0;
		gl::Bind(tex);
		tex.upload(gl::kSrgb8Alpha8, width, height, gl::kRgba, gl::kUnsignedByte, nullptr);
		tex.minFilter(gl::kLinear);
		tex.magFilter(gl::kLinear);

#ifdef USE_SURFACE
		cudaGraphicsResource *tex_res;
		CUDA_TRY(cudaGraphicsGLRegisterImage(&tex_res, tex.expose(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore), "Failed to register image");
#else // USE_SURFACE
		auto host_pixels = std::vector<uint32_t>(width * height);
		auto pixels = cuda::host::output_buffer<uint32_t>(width * height);
#endif // USE_SURFACE

		// Enable alpha blending
		gl::Enable(gl::kBlend);
		gl::BlendFunc(gl::kSrcAlpha, gl::kOneMinusSrcAlpha);

		// Set the clear color to white
		gl::ClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		cpputils::Timer<> zoom_timer;

		while (!glfwWindowShouldClose(window)) {
			cpputils::Timer<> fps;
			gl::Clear().Color();

#ifdef USE_SURFACE
			CUDA_TRY(cudaGraphicsMapResources(1, &tex_res, cuda::stream::default_stream().id()), "Failed to map graphics resources");
			cudaArray_t tex_array;
			CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(&tex_array, tex_res, 0, 0), "Failed to get mapped array");
			auto surf = cuda::surface(tex_array);
			mandelbrot_k_surface<<<gridsize, blocksize>>>(surf.descriptor(), width, height, cmin, cmax, iterations);
			CUDA_TRY(cudaGraphicsUnmapResources(1, &tex_res, cuda::stream::default_stream().id()), "Failed to unmap graphics resources");
#else // USE_SURFACE
			mandelbrot_k << <gridsize, blocksize >> >(pixels.data(), width, height, cmin, cmax, iterations);
			pixels.copy_d2h(host_pixels.data());
			tex.upload(gl::kSrgb8Alpha8, width, height, gl::kRgba, gl::kUnsignedByte, host_pixels.data());
			tex.minFilter(gl::kLinear);
			tex.magFilter(gl::kLinear);
#endif // USE_SURFACE
			cuda::stream::default_stream().synchronize();

			fp_type zoom_factor = zoom_per_frame;
			if (time_based) {
				zoom_factor = zoom_per_frame * zoom_timer.duration<std::chrono::duration<fp_type, std::milli>>();
				zoom_timer.reset();
			}
			cmin = cmin + (zoom - cmin) * zoom_factor;
			cmax = cmax - (cmax - zoom) * zoom_factor;

			rectangle_shape.render();
			glfwSwapBuffers(window);
			glfwPollEvents();
			glfwSetWindowTitle(window, std::string(std::to_string(1000ll / std::max<>(1ll, fps.duration<>())) + " FPS").c_str());
		}


		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	} catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}