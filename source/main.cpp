

#include "boids.h"

int main(int argc, char* argv[]) {


	// enable memory leak checking in debug mode
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// need static instance because callbacks don't understand member functions
	CUDA_GL& gl = CUDA_GL::getInstance();

	// init OpenGL
	std::cout << "Initializing OpenGL..." << std::endl;
	if (!gl.initGL()) return EXIT_FAILURE;

	// init CUDA and CUDA-GL interop
	std::cout << "Initializing CUDA-GL interop..." << std::endl;
	if (gl.initCuda() != cudaSuccess) return EXIT_FAILURE;

	// while ESC not pressed
	while (!glfwWindowShouldClose(gl.window)) {
		gl.display();

		std::cout << std::endl << "Shutting down..." << std::endl;

		gl.cleanup();
		
	}
	return EXIT_SUCCESS;
}