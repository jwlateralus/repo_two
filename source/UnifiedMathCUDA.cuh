

// This implementation follows the code from
// https://github.com/erwincoumans/experiments/blob/master/opencl/primitives/AdlPrimitives/Math/MathCL.h

#ifndef UNIFIED_MATH_CUDA_H
#define UNIFIED_MATH_CUDA_H

#include <vector_functions.h>

/*****************************************
				Vector
/*****************************************/

__device__
float fastDiv(float numerator, float denominator);

__device__
float getSqrtf(float f2);

__device__
float getReverseSqrt(float f2);

__device__
float3 getCrossProduct(float3 a, float3 b);

__device__
float4 getCrossProduct(float4 a, float4 b);

__device__
float getDotProduct(float3 a, float3 b);

__device__
float getDotProduct(float4 a, float4 b);

__device__ float3 getNormalizedVec(const float3 v);

__device__ float4 getNormalizedVec(const float4 v);

__device__
float dot3F4(float4 a, float4 b);

__device__
float getLength(float3 a);

__device__
float getLength(float4 a);

/*****************************************
				Matrix3x3
/*****************************************/
typedef struct
{
	float4 m_row[3];
}Matrix3x3_d;

__device__
void setZero(Matrix3x3_d& m);

__device__
Matrix3x3_d getZeroMatrix3x3();

__device__
void setIdentity(Matrix3x3_d& m);

__device__
Matrix3x3_d getIdentityMatrix3x3();

__device__
Matrix3x3_d getTranspose(const Matrix3x3_d m);

__device__
Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b);

__device__
float4 MatrixVectorMul(Matrix3x3_d a, float4 b);


#endif  // UNIFIED_MATH_CUDA_H