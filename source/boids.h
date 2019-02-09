

#ifndef BOIDS_H
#define BOIDS_H

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "header.h"
#include "gl_cuda.h"


#ifdef _DEBUG

#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif // DBG_NEW

#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _CRTDBG_MAP_ALLOC

#endif // _DEBUG

#endif