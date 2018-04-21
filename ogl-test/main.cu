#include "runtime/device.hpp"
#include "runtime/error.hpp"
#include "runtime/event.hpp"
#include "runtime/kernel.cu"
#include "runtime/memory.hpp"
#include "runtime/version.hpp"
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
#include "device_launch_parameters.h"
#include "surface_functions.h"

__global__ void fill_texture(cudaSurfaceObject_t surface, unsigned int width, unsigned int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		unsigned int greycode = 255 * (x + y) / (width + height);
		surf2Dwrite(make_char4(greycode, greycode, greycode, 255), surface, x * sizeof(char4), y);
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

int main() {
	constexpr unsigned int width = 640;
	constexpr unsigned int height = 640;
	dim3 gridsize(32, 32);
	dim3 blocksize(width / gridsize.x, height / gridsize.y);

	auto window = create_window("Test", width, height);

	gl::RectangleShape rectangle_shape({gl::RectangleShape::kPosition, gl::RectangleShape::kTexCoord});
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
	

	cudaGraphicsResource *tex_res;
	CUDA_TRY(cudaGraphicsGLRegisterImage(&tex_res, tex.expose(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore), "Failed to register image");

	// Enable alpha blending
	gl::Enable(gl::kBlend);
	gl::BlendFunc(gl::kSrcAlpha, gl::kOneMinusSrcAlpha);

	// Set the clear color to white
	gl::ClearColor(.0f, .0f, .0f, 1.0f);

	while (!glfwWindowShouldClose(window)) {
		gl::Clear().Color();
		cudaGraphicsMapResources(1, &tex_res, cuda::stream::default_stream().id());
		cudaArray_t writeArray;
		CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(&writeArray, tex_res, 0, 0), "Failed to get mapped array");
		cudaResourceDesc wdsc;
		wdsc.resType = cudaResourceTypeArray;
		wdsc.res.array.array = writeArray;
		cudaSurfaceObject_t surface;
		CUDA_TRY(cudaCreateSurfaceObject(&surface, &wdsc), "Failed to create surface object");
		fill_texture << <gridsize, blocksize >> > (surface, width, height);
		cudaDestroySurfaceObject(surface);
		cudaGraphicsUnmapResources(1, &tex_res, cuda::stream::default_stream().id());
		cuda::stream::default_stream().synchronize();
		rectangle_shape.render();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return EXIT_SUCCESS;
}