

// test-gitHub
// This module contains user-configurable and other parameters.

//################ User-configurable Parameters ################
//
#define NEIGHBORHOOD_MATRIX

// ---- Integer defines ----

// Square root of the number of boids simulated
#define		SQRT_NUM_BOIDS							(200.0)
#define		WINDOW_WIDTH							(3 * 1920 / 4)
#define		WINDOW_HEIGHT							(3 * 1080 / 4)

#define		CUDA_DEVICE								(0)


// 1 would be sufficient for a carefully tuned simulation.
#define		NUM_SIFTS_PER_FRAME						(1)

// number of frames into the past to track with a moving average
// for determination of physics delta-t's
#define		NUM_FRAMES_TO_AVERAGE					(6)


// how often to print current FPS to console, or comment out altogether to disable
#define		US_TO_PRINT_FPS							(1000000)

//in neighborhood matrix mode, radius of nearest neighbors
// to check and perform rules on (if actually within neighbor distance)
// 6 should be sufficinet  for this
#define		MOORE_NEIGHBORHOOD_RADIUS				(6)

// CUDA threads per CUDA execution block, in multiples of 32
#define		THREADS_PER_BLOCK						(128)

// ---- Float defines ----

// length of lines representing boids' position and velocity
#define		LINE_LENGTH								(5.0f)

// initial velocitys
#define		fMAX_START_VEL							(40.0f)

// speed limit of boids
#define		V_LIM									(200.0f)

// speed of physics
#define		TICK_FACTOR								(0.000005f)

// strength of alignment rule
#define		ALIGNMENT_STRENGTH_FACTOR				(0.015f)

// strength of repulsion rule 
#define		REPULSION_STRENGTH_FACTOR				(55.0f)
#define		EDGE_REPULSION_STRENGTH_FACTOR			(0.0002f)
#define		CENTER_OF_MASS_STRENGTH_FACTOR			(0.6f)
//##############################################################


#define	NUM_BOIDS					4000
#define US_PER_SECOND				(1000000)
#define THREE_OVER_PI				(0.95492965855137f);
#define fPI							(3.14159265359f);

#define NUM_BLOCKS	256

// square coordinate system max value (for screen independence)
#define P_MAX						(10000)
#define fP_MAX						(10000.0f)
#define fHALF_P_MAX					(5000.0f)
#define BOID_STARTING_BOX			(P_MAX / SQRT_NUM_BOIDS) 
// max distance to be considered a neighbor
#define NEIGHBOR_DISTANCE			(P_MAX / 13)
#define NEIGHBOR_DISTANCE_SQUARED	(NEIGHBOR_DISTANCE * NEIGHBOR_DISTANCE)
#define NUM_HSV_SECTORS				(6)
#define MAX_RGB						(255)
#define fMAX_RGB					(255.0f)
#define	V_LIM_2						(V_LIM * V_LIM)
#define PREVENT_ZERO_RETURN			(0.000000001f)

