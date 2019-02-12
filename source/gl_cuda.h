
// This module contains the entirety of the OpenGL, CUDA,
// and CUDA-GL interop for maximum performance. 
#include <chrono>
#include <cstdint>
#include <iostream>
#include <list>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "params.h"


//http://www.kfish.org/boids/pseudocode.html
//Every particle/boid in this simulation can be represented by vectors 
//it involves simple vector operations on the positions of the boids. 
//Each of the boids rules works independently, so, for each boid, you calculate how much it will get moved by each of the three rules, giving you three velocity vectors. 
//Then you add those three vectors to the boid's current velocity to work out its new velocity. 
//Interpreting the velocity as how far the boid moves per time step we simply add it to the current position.

#define CHECK_FOR_CUDA_ERRORS\
 do { \
	if (cudaStatus != cudaSuccess) { \
		std::cout << "ERROR: CUDA reports " << cudaGetErrorString(cudaStatus) << ". Aborting." << std::endl; \
			goto Error; \
						} \
						} while (false)

struct Boid {
	float x, y, vx, vy;
};

class CUDA_GL {
public:
	GLFWwindow* window;
	bool not_paused;
	std::chrono::high_resolution_clock::time_point frame_clock, FPS_clock;
	bool do_blank;
	int mouse_x, mouse_y, mouse_buttons_down;

private:
	// framebuffer NOT screen
	int width, height;

	// CUDA-GL interop
	cudaGraphicsResource_t cgr;
	GLuint pbo;
	GLuint textureID;
	cudaStream_t cs;
	uchar4* d_pbo;
	//////////////////

	// d_ prefix indicates device (CUDA global) ptr
	Boid* d_in;
	Boid* d_out;

	// h_ prefix indicates host ptr
	Boid h_in[NUM_BOIDS];
	Boid h_out[NUM_BOIDS];

	// screen blanker is also a CUDA kernel
	int blanker_blocks;

	// keep a history of framedraw times
	// for moving average
	std::list<long long> frame_times;

	int frames, fps_frames;
	long long frame_time_sum;

public:
	// Singleton idiom - only one
	// instance of this class is permitted
	// since we need static methods as callbacks
	// but still want class functionality
	static CUDA_GL& getInstance() {
		static CUDA_GL instance;
		return instance;
	};

	GLFWwindow* initGL();
	cudaError_t initCuda();
	void cleanup();
	void display();

private:
	CUDA_GL() : mouse_buttons_down(0),
		do_blank(true),
		frames(0),
		not_paused(true),
		width(WINDOW_WIDTH),
		height(WINDOW_HEIGHT),
		frame_time_sum(0),
		frame_times(1, 0),	// start with { 0 } in the frame_times history
		pbo(0),
		textureID(0),
		d_pbo(nullptr),
		d_in(nullptr),
		d_out(nullptr) {}

	CUDA_GL(const CUDA_GL&) = delete;
	CUDA_GL& operator=(const CUDA_GL&) = delete;

	void cleanupCuda();
	cudaError_t spawnBoids();
};


// GL event callbacks
void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseClickCallback(GLFWwindow* window, int button, int action, int mods);
void mouseMoveCallback(GLFWwindow* window, double x, double y);
void windowMoveCallback(GLFWwindow* window, int x, int y);

// --- CUDA kernels ---




// main compute kernel
__global__ void kernel(uchar4* __restrict__ const d_pbo, const Boid* __restrict__ const in, Boid* __restrict__ const out, const int mouse_x, const int mouse_y, const int width, const int height, const int us_since_last_frame);


// on each frame when in neighborhood matrix mode
__global__ void siftEvenCols(Boid* __restrict__ d_arr);
__global__ void siftOddCols(Boid* __restrict__ d_arr);
__global__ void siftEvenRows(Boid* __restrict__ d_arr);
__global__ void siftOddRows(Boid* __restrict__ d_arr);
