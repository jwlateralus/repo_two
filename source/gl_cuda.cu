
// module for cuda and openGL interoperability 

#include "gl_cuda.h"

// Random number generator for Boid initial positions and velocities
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> position_random_dist(0.0f, BOID_STARTING_BOX);
std::uniform_real_distribution<float> velocity_random_dist(-fMAX_START_VEL, fMAX_START_VEL);

// need static instance because callbacks don't understand member functions
CUDA_GL& gl = CUDA_GL::getInstance();

void CUDA_GL::cleanup() {
	cleanupCuda();
	glfwDestroyWindow(window);
	glfwTerminate();
}

// self-explanatory openGL context

GLFWwindow* CUDA_GL::initGL() {
	std::cout << "Initializing GLFW..." << std::flush;
	if (!glfwInit()) {
		std::cout << "ERROR: Failed to initialize GLFW. Aborting." << std::endl;
		return nullptr;
	}
	std::cout << " done." << std::endl;
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	window = glfwCreateWindow(width, height, "Boids", nullptr, nullptr);
	std::cout << "Window created." << std::endl;


	if (!window) {
		std::cout << "ERROR: Failed to create window. Aborting." << std::endl;
		return nullptr;
	};

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	glfwGetFramebufferSize(window, &width, &height);
	std::cout << "Framebuffer reports dimensions " << width << "x" << height << '.' << std::endl;

	// register callbacks with OpenGL
	glfwSetKeyCallback(window, keyboardCallback);
	glfwSetMouseButtonCallback(window, mouseClickCallback);
	glfwSetWindowPosCallback(window, windowMoveCallback);
	std::cout << "Callbacks registered." << std::endl;

	std::cout << "Initializing GLEW..." << std::flush;
	glewInit();
	std::cout << " done." << std::endl;

	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		std::cout << "ERROR: Support for necessary OpenGL extensions missing. Aborting" << std::endl;
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
			std::cout << "ERROR: OpenGL reports " << err << ". Aborting." << std::endl;
		} while ((err = glGetError()) != GL_NO_ERROR);
		return nullptr;
	}

	blanker_blocks = (width*height - 1) / THREADS_PER_BLOCK + 1;

	return window;
}

__global__ void siftEvenCols(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	int j = boid % SQRT_NUM_BOIDS;
	if (j % 2 == 0) return;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[i*SQRT_NUM_BOIDS + j - 1];

	// swap elements if required such that smaller element
	// (sorted by x-position) ends up on left
	if (one->x < two->x) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftOddCols(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	int j = boid % SQRT_NUM_BOIDS;
	if ((j % 2) || j == 0) return;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[i*SQRT_NUM_BOIDS + j - 1];

	// swap elements if required such that smaller element
	// (sorted by x-position) ends up on left
	if (one->x < two->x) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftEvenRows(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	if (i % 2 == 0) return;
	int j = boid % SQRT_NUM_BOIDS;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[(i - 1)*SQRT_NUM_BOIDS + j];

	// swap elements if required such that smaller element
	// (sorted by y-position) ends up on top
	if (one->y < two->y) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftOddRows(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	if ((i % 2) || i == 0) return;
	int j = boid % SQRT_NUM_BOIDS;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[(i - 1)*SQRT_NUM_BOIDS + j];
	if (one->y < two->y) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}
#endif

void CUDA_GL::display() {
		// sift NUM_SIFTS times to update matrix for any boids
		// which changed positions relative to other boids during previous frame
		for (int i = 0; i < NUM_SIFTS_PER_FRAME; ++i) {
			siftEvenCols << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftOddCols << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftEvenRows << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftOddRows << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
		}


		// simulation kernel:
		kernel << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(
			d_pbo,		
			d_in,		
							
			d_out,			
			mouse_x,
			mouse_y,
			width,
			height,
		cudaDeviceSynchronize();
	// free the graphics resources for rendering
		cudaGraphicsUnmapResources(1, &cgr, cs);
		// swap pointers of in and out arrays each frame so data is written back and forth (all on GPU)
		Boid* temp = d_in;
		d_in = d_out;
		d_out = temp;
	}
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
	if (frame_times.size() >= NUM_FRAMES_TO_AVERAGE) {
		frame_time_sum -= frame_times.front();
		frame_times.erase(frame_times.begin());
	}
	// record current time
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
	// push last frametime to frame_times
	frame_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(now - frame_clock).count());
	// update running sum of all frames in frame_times
	frame_time_sum += frame_times.back();


void CUDA_GL::cleanupCuda() {
	cudaFree(d_in);
	cudaFree(d_out);
	std::cout << "CUDA memory freed." << std::endl;

	cudaGraphicsUnregisterResource(cgr);
	std::cout << "CUDA graphics resource unregistered." << std::endl;

	cudaStreamDestroy(cs);
	std::cout << "CUDA interop finish." << std::endl;

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

cudaError_t CUDA_GL::spawnBoids() {
	cudaError_t cudaStatus;

	// generate host-side random initial positions
	int idx;
	for (int i = 0; i < SQRT_NUM_BOIDS; ++i) {
		for (int j = 0; j < SQRT_NUM_BOIDS; ++j) {
			idx = i*SQRT_NUM_BOIDS + j;
			h_in[idx].vx = velocity_random_dist(gen);
			h_in[idx].vy = velocity_random_dist(gen);
			h_in[idx].x = j*BOID_STARTING_BOX + position_random_dist(gen);
			h_in[idx].y = i*BOID_STARTING_BOX + position_random_dist(gen);
		}
	}
	cudaStatus = cudaMemcpy(d_in, h_in, NUM_BOIDS * sizeof(Boid), cudaMemcpyHostToDevice);
	CHECK_FOR_CUDA_ERRORS;

	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}

cudaError_t CUDA_GL::initCuda() {
	// each pixel is a uchar4 (RGBA)
	size_t size = sizeof(uchar4) * width * height;
	// generate buffer 1 and map pbo to it
	glGenBuffers(1, &pbo);
	std::cout << "Pixel buffer created." << std::endl;
	// attach to it
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW); 
	//GL_DYNAMIC_COPY
	// enable textures
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	std::cout << "Texture created." << std::endl;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(CUDA_DEVICE);
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "CUDA device " << CUDA_DEVICE << " selected." << std::endl;
	cudaGraphicsGLRegisterBuffer(&cgr, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaStreamCreate(&cs);
	std::cout << "CUDA interop stream created." << std::endl;
	cudaGraphicsMapResources(1, &cgr, cs);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pbo), &size, cgr);
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
	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}

void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) 
{
	
}

void mouseClickCallback(GLFWwindow * window, int button, int action, int mods)
{
}

void mouseMoveCallback(GLFWwindow * window, double x, double y)
{
}

void windowMoveCallback(GLFWwindow* window, int x, int y) {
	gl.frame_clock = std::chrono::high_resolution_clock::now();
}


__global__ void kernel(
	uchar4* __restrict__ const d_pbo,
	const Boid* __restrict__ const in,
	Boid* __restrict__ const out,
	const int mouse_x,
	const int mouse_y,
	const int width,
	const int height,
	const int mean_frame_time)
{

	int boid = blockDim.x * blockIdx.x + threadIdx.x;

	if (boid >= NUM_BOIDS) return;

	float diffx, diffy, factor;

	float time_factor = TICK_FACTOR*mean_frame_time;

	// bring boid's position and velocity into local memory
	float x = in[boid].x, y = in[boid].y;
	float Vx = in[boid].vx, Vy = in[boid].vy;

	float magV;
	diffx = (mouse_x * P_MAX / width) - x;
	diffy = (mouse_y * P_MAX / height) - y;

	Vx += time_factor * diffx * factor;
	Vy += time_factor * diffy * factor;

	int i, idx, ix, iy;
	// apply neighbor-related rules
	int neighbors = 0;
	//float neighbors = 0.0f;
	float CMsumX = 0.0f, CMsumY = 0.0f, REPsumX = 0.0f, REPsumY = 0.0f, ALsumX = 0.0f, ALsumY = 0.0f;
	// neighbors (includes own boid - optimization and prevents div-by-zero when no neighbors)
	ix = boid / SQRT_NUM_BOIDS;
	iy = boid % SQRT_NUM_BOIDS;
	int j;
	// find boundaries of Moore radius box while not letting box go outside matrix
	int iterm = (ix < SQRT_NUM_BOIDS - MOORE_NEIGHBORHOOD_RADIUS) ? ix + MOORE_NEIGHBORHOOD_RADIUS : SQRT_NUM_BOIDS;
	int jterm = (iy < SQRT_NUM_BOIDS - MOORE_NEIGHBORHOOD_RADIUS) ? iy + MOORE_NEIGHBORHOOD_RADIUS : SQRT_NUM_BOIDS;

	// loop over Moore radius box
	for (i = (ix > MOORE_NEIGHBORHOOD_RADIUS) ? ix - MOORE_NEIGHBORHOOD_RADIUS : 0; i < iterm ; ++i) {
		for (j = (iy > MOORE_NEIGHBORHOOD_RADIUS) ? iy - MOORE_NEIGHBORHOOD_RADIUS : 0; j < jterm; ++j) {

			idx = i * SQRT_NUM_BOIDS + j;

			diffx = in[idx].x - x;
			diffy = in[idx].y - y;
			magV = fmaf(diffx, diffx, fmaf(diffy, diffy, PREVENT_ZERO_RETURN));

			// if this is a neighbor...
			if (magV < NEIGHBOR_DISTANCE_SQUARED) {
				// __fdividef is fast approximate float division
				factor = __fdividef(1.0f, magV);

				// update center of mass rule by distance and direction to neigbor and the three basic rules
				CMsumX += diffx;
				REPsumX -= diffx * factor;
				ALsumX += in[idx].vx;

				CMsumY += diffy;
				REPsumY -= diffy * factor;
				ALsumY += in[idx].vy;
				++neighbors;
			
				// keep track of total neighbor count for averaging these rule sums
			
			}
		}
		}
	

	for (idx = 0; idx < NUM_BOIDS; ++idx) {

		diffx = in[idx].x - x;
		diffy = in[idx].y - y;


		magV = fmaf(diffx, diffx, fmaf(diffy, diffy, PREVENT_ZERO_RETURN));

		if (magV < NEIGHBOR_DISTANCE_SQUARED) {
			factor = __fdividef(1.0f, magV);
			CMsumX += diffx;
			REPsumX -= diffx * factor;
			ALsumX += in[idx].vx;

			CMsumY += diffy;
			REPsumY -= diffy * factor;
			ALsumY += in[idx].vy;

			++neighbors;
		}
	}


	Vx += fmaf(time_factor, __fdividef(CMsumX * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumX + EDGE_REPULSION_STRENGTH_FACTOR*(x < NEIGHBOR_DISTANCE)*(NEIGHBOR_DISTANCE - x) - EDGE_REPULSION_STRENGTH_FACTOR*(x > fP_MAX - NEIGHBOR_DISTANCE)*(x - (fP_MAX - NEIGHBOR_DISTANCE)), __fdividef((ALsumX - in[boid].vx)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));
	Vy += fmaf(time_factor, __fdividef(CMsumY * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumY + EDGE_REPULSION_STRENGTH_FACTOR*(y < NEIGHBOR_DISTANCE)*(NEIGHBOR_DISTANCE - y) - EDGE_REPULSION_STRENGTH_FACTOR*(y > fP_MAX - NEIGHBOR_DISTANCE)*(y - (fP_MAX - NEIGHBOR_DISTANCE)), __fdividef((ALsumY - in[boid].vy)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));


	// limit velocity
	magV = Vx*Vx + Vy*Vy;
	factor = (magV > V_LIM_2) ? V_LIM * rsqrtf(magV) : 1.0f;
	Vx *= factor;
	Vy *= factor;


	
	if (x < 0.0f) Vx = fabs(Vx);
	if (x >= fP_MAX) Vx = -fabs(Vx);
	if (y < 0.0f) Vy = fabs(Vy);
	if (y >= fP_MAX) Vy = -fabs(Vy);

	// ...and THEN update position so we move back inside
	x += Vx * time_factor;
	y += Vy * time_factor;

	out[boid].x = x;
	out[boid].y = y;


	// store velocities back to global
	out[boid].vx = Vx;
	out[boid].vy = Vy;

	// convert position to screen frame
	ix = width * __fdividef(x, fP_MAX);
	iy = height * __fdividef(y, fP_MAX);

	Vx *= width;
	Vy *= height;

	// ...which is done here.
	magV = rsqrtf(fmaf(Vx, Vx, fmaf(Vy, Vy, PREVENT_ZERO_RETURN)));

	// + pi to make range 0-2pi
	float angle = atan2f(Vx *= magV, Vy *= magV) + fPI;
	float section = angle * THREE_OVER_PI;
	int i = static_cast<int>(section);
	float frac = section - i;

	uint8_t HSV_to_RGB[6] = { MAX_RGB, static_cast<uint8_t>(fMAX_RGB - fMAX_RGB*frac), 0, 0, static_cast<uint8_t>(fMAX_RGB*frac), MAX_RGB };

	uint8_t R = HSV_to_RGB[i];
	uint8_t G = HSV_to_RGB[(i + 4) % NUM_HSV_SECTORS];
	uint8_t B = HSV_to_RGB[(i + 2) % NUM_HSV_SECTORS];
	/////////////////////////


	// -- line draw --- //

	// use normalized velocity vector to
	// compute start and end points
	// of line in screen frame
	int x1 = ix - Vx*LINE_LENGTH;
	int x2 = ix + Vx*LINE_LENGTH;
	int y1 = iy - Vy*LINE_LENGTH;
	int y2 = iy + Vy*LINE_LENGTH;

	int dx, dy, sx, sy, D;

	// get signs of dx and dy
	sx = (x2 > x1) - (x2 < x1);
	sy = (y2 > y1) - (y2 < y1);

	// ensure positive
	dx = sx * (x2 - x1);
	dy = sy * (y2 - y1);
	if (dy > dx) {
		D = 2 * dx - dy;
		for (i = 0; i < dy; ++i) {
			// pixel write //

			// just cap at edges if outside screen
			boid = ((y1 >= 0) ? ((y1 < height) ? y1 : height - 1) : 0)*width + ((x1 >= 0) ? ((x1 < width) ? x1 : width - 1) : 0);
			d_pbo[boid].x = R;
			d_pbo[boid].y = G;
			d_pbo[boid].z = B;
			//////////////

			while (D >= 0) {
				D -= 2 * dy;
				x1 += sx;
			}

			D += 2 * dx;
			y1 += sy;
		}
	}
	else {
		D = 2 * dy - dx;
		for (i = 0; i < dx; ++i) {
			// pixel write

			boid = ((y1 >= 0) ? ((y1 < height) ? y1 : height - 1) : 0)*width + ((x1 >= 0) ? ((x1 < width) ? x1 : width - 1) : 0);
			d_pbo[boid].x = R;
			d_pbo[boid].y = G;
			d_pbo[boid].z = B;
			//////////////

			while (D >= 0) {
				D -= 2 * dx;
				y1 += sy;
			}

			D += 2 * dy;
			x1 += sx;
		}
	}

}