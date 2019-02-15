#include "CUDA_GL.h"


// This module contains the entirety of the OpenGL, CUDA,
// and CUDA-GL interop for maximum performance. This is
// a showcase of high-performance computing and thus some
// readability sacrifices were made. Comments throughout
// explain rationale and basic optimization trajectory.

// CUDA kernel (first compiled by NVCC, then MSVC)


std::random_device rd;					// mt19937 has much longer period than that of rand, e.g. it will take its random sequence much longer to repeat itself.
										//to generate random initial position for boids
std::mt19937 gen(rd());
std::uniform_real_distribution<float> position_random_dist(0.0f, fP_MAX); //  return an integer between 0.0 to 10000 and THEN cast it  to a float


// need static instance because callbacks don't understand member functions
CUDA_GL_Interoperability& gl = CUDA_GL_Interoperability::getInstance();

void CUDA_GL_Interoperability::cleanup() {
	cleanupCuda();
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLFWwindow* CUDA_GL_Interoperability::initGL() {
	std::cout << "Initializing GLFW..." << std::flush;
	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL isn't available?"
			<< std::endl;
		return false;
	}


	std::cout << " GLFW initialization is now complete." << std::endl;


// full screen code http://discourse.glfw.org/t/creating-windowed-fullscreen/825/2
#ifdef enable_FullScreen
	GLFWmonitor* const primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* const mode = glfwGetVideoMode(primary);
	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	window = glfwCreateWindow(mode->width, mode->height, " FLOCK you ", primary, nullptr);
	std::cout << "FULL screen  framebuffer created." << std::endl;

#else
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(win_width, win_height, "Flock ON", nullptr, nullptr);
	std::cout << "Window created." << std::endl;
#endif

	if (!window) {
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(false);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	glfwSwapInterval(GLFW_FALSE);
	glfwGetFramebufferSize(window, &width, &height);
	std::cout << "Framebuffer dimensions " << width << "x" << height << '.' << std::endl;

	// register callbacks with OpenGL
	glfwSetKeyCallback(window, keyboardCallback);
	glfwSetMouseButtonCallback(window, mouseClickCallback);
	glfwSetWindowPosCallback(window, windowMoveCallback);
	std::cout << "Callbacks registered." << std::endl;

	std::cout << "Initializing GLEW..." << std::flush;
	glewInit();
	std::cout << " done." << std::endl;

	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing. Aborting" << std::endl;
		return nullptr;
	}

	// set viewport and viewing modes
	glViewport(0, 0, width, height);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	std::cout << "OpenGL viewport configured." << std::endl;

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// set proj matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

	// create texture from pixel buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// bind texture
	glBindTexture(GL_TEXTURE_2D, textureID);
	std::cout << "OpenGL texture configured." << std::endl;

	GLenum err;
	if ((err = glGetError()) != GL_NO_ERROR) {
		do {
			std::cerr << "ERROR: OpenGL reports " << err << ". Aborting." << std::endl;
		} while ((err = glGetError()) != GL_NO_ERROR);
		return nullptr;
	}

	blanker_blocks = (width*height - 1) / THREADS_PER_BLOCK + 1;

	return window;
}

void CUDA_GL_Interoperability::display() {
	if (running) {
		// map cuda stream to access pbo
		cudaGraphicsMapResources(1, &CUDA_graphics_res, cs);
		if (do_blank) {
			blanker << <blanker_blocks, THREADS_PER_BLOCK >> > (d_pbo, width*height);
		}


		// simulation kernel:
		kernel << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (
			d_pbo,			// pixel buffer to write into
			d_in,			// boids are stored ping-pong style so that CUDA threads
							// can update boid info without syncing to other threads
			d_out,			// output (swaps with input each frame)
			mouse_x,
			mouse_y,
			width,
			height,

			// send 0 if no mouse buttons down, OR
			// send the repulsion multipler (attract vs. repel) times the weak factor if 1 button down, OR
			// send the repulsion multiplier times the strong factor if two or more buttons down
			(mouse_buttons_down) ? (attr_repulsion_weight_multiplier * ((mouse_buttons_down > 1) ? both_mouse_buttons_pressed_behaviour : mouse_button_press_behaviour)) : 0.0f,
			static_cast<int>(frame_time_sum / frame_times.size())); // moving average of time elapsed per frame (for physics propagation)

		// wait for all threads to complete
		cudaDeviceSynchronize();

		// free the graphics resources for rendering
		cudaGraphicsUnmapResources(1, &CUDA_graphics_res, cs);

		// swap pointers of in and out arrays each frame so data is written back and forth (all on GPU)
		Boid* temp = d_in;
		d_in = d_out;
		d_out = temp;
	}

	// nothing to copy -  it's already on device!
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	// lay simple quad on which we scribble the pixels - or rather, on
	// which they're already scribbled
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	// flip to front buffer
	glfwSwapBuffers(window);

	// check for key/mouse input
	glfwPollEvents();

	// if in normal simulation, frame_times is full, so remove its oldest element
	// and update frame_time_sum accordingly (there's no need to sum it every frame;
	// just subtract from it when you remove a frame and add to it when you add a frame!
	// as long as the initial value is correct (0 in this case), this will maintain a correct
	// sum for use in averaging.
	//
	// if in first few frames of simulation, frame_times is still lengthening so skip this step.
	if (frame_times.size() >= frame_rate_print_freq) {
		frame_time_sum -= frame_times.front();
		frame_times.erase(frame_times.begin());
	}

	// record current time
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();

	// push last frametime to frame_times
	frame_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(now - frame_clock).count());

	// update running sum of all frames in frame_times
	frame_time_sum += frame_times.back();

	// if supposed to print FPS output...
#ifdef frame_rate_print_freq
	// ...and if enough time has elapsed that it's time to print...
	long long us_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - FPS_clock).count();
	if (us_elapsed > frame_rate_print_freq) {
		// ...print...
		std::cout << "FPS: " << fps_frames * us_elapsed / frame_rate_print_freq << std::endl;
		// ...and update the clock to now.
		FPS_clock = now;
		fps_frames = 0;
	}

	++fps_frames;

#endif

	// save current time to be checked after render to determine
	// framedraw time
	frame_clock = std::chrono::high_resolution_clock::now();
}

void CUDA_GL_Interoperability::cleanupCuda() {
	cudaFree(d_in);
	cudaFree(d_out);
	std::cout << "CUDA memory freed." << std::endl;

	cudaGraphicsUnregisterResource(CUDA_graphics_res);
	std::cout << "CUDA graphics resource unregistered." << std::endl;

	cudaStreamDestroy(cs);
	std::cout << "CUDA interop stream destroyed." << std::endl;

	cudaDeviceReset();
	std::cout << "CUDA device reset." << std::endl;

	if (textureID) {
		glDeleteTextures(1, &textureID);
		textureID = 0;
	}
	std::cout << "Texture destroyed." << std::endl;

	if (pbo) {
		glBindBuffer(GL_ARRAY_BUFFER, pbo);
		glDeleteBuffers(1, &pbo);
		pbo = 0;
	}
	std::cout << "Pixel buffer destroyed." << std::endl;
}

cudaError_t CUDA_GL_Interoperability::spawnBoids() {
	cudaError_t cudaStatus;


	// if not in neighborhood matrix mode, we can simply
	// spawn the boids randomly anywhere on the screen
	for (auto&& boid : h_in) {
		boid.vx = boid.vy = 0.0f;
		boid.x = position_random_dist(gen);
		boid.y = position_random_dist(gen);
	}

	// transfer initial boids to GPU (the only time boids are transferred! They
	// reside fully in GPU global memory from now on)
	cudaStatus = cudaMemcpy(d_in, h_in, NUM_BOIDS * sizeof(Boid), cudaMemcpyHostToDevice);
	CHECK_FOR_CUDA_ERRORS;

	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}

cudaError_t CUDA_GL_Interoperability::initCuda() {
	// each pixel is a uchar4 (RGBA)
	size_t size = sizeof(uchar4) * width * height;

	// generate buffer 1 and map pbo to it
	glGenBuffers(1, &pbo);
	std::cout << "Pixel buffer created." << std::endl;

	// attach to it
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// allocate
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW); //GL_DYNAMIC_COPY

	// enable textures
	glEnable(GL_TEXTURE_2D);

	// generate texture ID for buffer 1
	glGenTextures(1, &textureID);

	// attach to it
	glBindTexture(GL_TEXTURE_2D, textureID);

	// allocate. The last parameter is nullptr since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

	// don't need mipmapping
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	std::cout << "Texture created." << std::endl;

	cudaError_t cudaStatus;

	// choose which GPU to run on
	cudaStatus = cudaSetDevice(CUDA_device);
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "CUDA device " << CUDA_device << " selected." << std::endl;

	// register pbo as a CUDA graphics resource
	cudaGraphicsGLRegisterBuffer(&CUDA_graphics_res, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	// create interop stream
	cudaStreamCreate(&cs);
	std::cout << "CUDA interop stream created." << std::endl;

	// map the resource into that stream
	cudaGraphicsMapResources(1, &CUDA_graphics_res, cs);

	// get device ptr for that resource so CUDA can write into it
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pbo), &size, CUDA_graphics_res);
	std::cout << "CUDA graphics resource registered." << std::endl;

	// allocate GPU ping-pong Boid arrays
	cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_in), NUM_BOIDS * sizeof(Boid));
	CHECK_FOR_CUDA_ERRORS;
	cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_out), NUM_BOIDS * sizeof(Boid));
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "CUDA memory allocated." << std::endl;

	cudaStatus = spawnBoids();
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "Boids spawned." << NUM_BOIDS << std::endl;

	// initialize both clocks
	FPS_clock = frame_clock = std::chrono::high_resolution_clock::now();

	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}