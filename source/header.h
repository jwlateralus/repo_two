//contains user configurable parameters
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "CUDA_GL.h"
#include "interop.cuh"
#include "UnifiedMathCUDA.cuh"

//#define		enable_FullScreen			//	 Enable/ disable full screen   -- to do
//#define		screen_Wrapping				//		 Enable / disable screen wrapping 

#define		root_num_boids						(300)  // root of the number of boids
#define		NUM_BOIDS							(6000)//NUMBER OF BOIDS | root_num_boids x root_num_boids 
#define		win_width							(3 * 1920 / 4) // width of the window
#define		win_height							(3 * 1080 / 4) // height of the window
#define		CUDA_device								(0)  // Cuda device init


											// number_frames_to_average	 number of frames into the past to track with 
#define		number_frames_to_average					(10)
											//a moving average for determination of physics delta-t's

#define		frame_rate_print_freq (1000000)
											//required to get fps info on console, how often to print

											// CUDA threads per CUDA execution block. 
#define		THREADS_PER_BLOCK (128)
											//Empirically set to 128, multiples of 32 

#define		boid_length	(1.2f)


#define		velocity_limit (200.0f)  // speed limit of boids

											// speed of the boids and boids within the flock
#define		flock_phys_speed_  (0.000006f)
											// higher number means higherspped movement when flying around 


/*
		Rule 1:Boids try to match velocity with near boids.
		Rule 2: Boids try to keep a small distance away from other objects (including other boids)
		Rule 3: Boids try to fly towards the centre of mass of neighbouring boid

		intensive play and experimentation  with the weight values required to get an accurate visual simulation
*/

#define		boids_Alignment_Rule_value				(0.015f)	// magnitude of the alignment rule 


#define		boids_Seperation_Rule_value				(55.0f)   // magnitude of the repel rule, prevents boids from running into each other


#define		boids_Cohesion_Rule_value				(0.6f)		// magnitude of the centre of mass or the cohession rule 


#define neighbourhood_distance			(P_MAX / 13)               //// max distance to be considered a neighbor
#define neighbourhood_distance_sqrd		(neighbourhood_distance * neighbourhood_distance)



#define		screen_edge_avoidance			(0.0002f)			// magnitude of the  edge repulsion rule, bounce off the screen when particles hit the edge of the screen





#define		mouse_button_press_behaviour			(100.0f)  // strength of one mouse button down mouse attaction/repulsion, flip the sign to switch between attraction and repulsion
#define		both_mouse_buttons_pressed_behaviour	(6000.0f) // strength of one mouse button down mouse attaction/repulsion, flip the sign to switch between attraction and repulsion






#define simulation_update_per_seconds	(10000000f)
#define THREE_OVER_PI					(1.464591888f)
#define PI								(3.1415926535f)

#define NUM_BLOCKS	521 //				((NUM_BOIDS - 1) / THREADS_PER_BLOCK + 1)

// in game screen coordinate, maximum value for better visuals
#define P_MAX						(10000)
#define fP_MAX						(10000.0f)
#define fHALF_P_MAX					(5000.0f)
#define NUM_HSV_SECTORS				(6)
#define MAX_RGB						(255)
#define fMAX_RGB					(255.0f)
#define PREVENT_ZERO_RETURN			(0.000000001f)

//***********************************************************************************************************************************************************************************************