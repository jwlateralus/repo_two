#include "UnifiedMathCUDA.cuh"


/*****************************************
				Vector
/*****************************************/

__device__
float fastDiv(float numerator, float denominator)
{
	return __fdividef(numerator, denominator);
	//return numerator/denominator;        
}

__device__
float getSqrtf(float f2)
{
	return sqrtf(f2);
}

__device__
float getReverseSqrt(float f2)
{
	return rsqrtf(f2);
}

__device__
float3 getCrossProduct(float3 a, float3 b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__
float4 getCrossProduct(float4 a, float4 b)
{
	float3 v1 = make_float3(a.x, a.y, a.z);
	float3 v2 = make_float3(b.x, b.y, b.z);
	float3 v3 = make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);

	return make_float4(v3.x, v3.y, v3.z, 0.0f);
}

__device__
float getDotProduct(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
float getDotProduct(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ float3 getNormalizedVec(const float3 v)
{
	float invLen = 1.0f / sqrtf(getDotProduct(v, v));
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ float4 getNormalizedVec(const float4 v)
{
	float invLen = 1.0f / sqrtf(getDotProduct(v, v));
	return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.x, a.y, a.z, 0.f);
	float4 b1 = make_float4(b.x, b.y, b.z, 0.f);
	return getDotProduct(a1, b1);
}

__device__
float getLength(float3 a)
{
	return sqrtf(getDotProduct(a, a));
}

__device__
float getLength(float4 a)
{
	return sqrtf(getDotProduct(a, a));
}

/*****************************************
				Matrix3x3
/*****************************************/

__device__
void setZero(Matrix3x3_d& m)
{
	m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__
Matrix3x3_d getZeroMatrix3x3()
{
	Matrix3x3_d m;
	m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	return m;
}

__device__
void setIdentity(Matrix3x3_d& m)
{
	m.m_row[0] = make_float4(1, 0, 0, 0);
	m.m_row[1] = make_float4(0, 1, 0, 0);
	m.m_row[2] = make_float4(0, 0, 1, 0);
}

__device__
Matrix3x3_d getIdentityMatrix3x3()
{
	Matrix3x3_d m;
	m.m_row[0] = make_float4(1, 0, 0, 0);
	m.m_row[1] = make_float4(0, 1, 0, 0);
	m.m_row[2] = make_float4(0, 0, 1, 0);
	return m;
}

__device__
Matrix3x3_d getTranspose(const Matrix3x3_d m)
{
	Matrix3x3_d out;
	out.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
	out.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
	out.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
	return out;
}

__device__
Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b)
{
	Matrix3x3_d transB = getTranspose(b);
	Matrix3x3_d ans;
	//        why this doesn't run when 0ing in the for{}
	a.m_row[0].w = 0.f;
	a.m_row[1].w = 0.f;
	a.m_row[2].w = 0.f;
	for (int i = 0; i < 3; i++)
	{
		//        a.m_row[i].w = 0.f;
		ans.m_row[i].x = dot3F4(a.m_row[i], transB.m_row[0]);
		ans.m_row[i].y = dot3F4(a.m_row[i], transB.m_row[1]);
		ans.m_row[i].z = dot3F4(a.m_row[i], transB.m_row[2]);
		ans.m_row[i].w = 0.f;
	}
	return ans;
}

__device__
float4 MatrixVectorMul(Matrix3x3_d a, float4 b)
{
	float4 ans;
	ans.x = dot3F4(a.m_row[0], b);
	ans.y = dot3F4(a.m_row[1], b);
	ans.z = dot3F4(a.m_row[2], b);
	ans.w = 0.f;
	return ans;
}