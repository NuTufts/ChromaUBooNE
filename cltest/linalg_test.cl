//-*-c-*-

//#include "linalg.h"

//extern "C"
//{

__kernel void float3add(__global float3 *a, __global float3 *b, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx] + b[idx];
}

__kernel void float3addequal(__global float3 *a, __global float3 *b)
{
	int idx = get_global_id(0);
	a[idx] += b[idx];
}

__kernel void float3sub(__global float3 *a, __global float3 *b, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx] - b[idx];
}

__kernel void float3subequal(__global float3 *a, __global float3 *b)
{
	int idx = get_global_id(0);
	a[idx] -= b[idx];
}

__kernel void float3addfloat(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx] + c;
}

__kernel void float3addfloatequal(__global float3 *a, float c)
{
	int idx = get_global_id(0);
	a[idx] += c;
}

__kernel void floataddfloat3(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = c + a[idx];
}

__kernel void float3subfloat(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx] - c;
}

__kernel void float3subfloatequal(__global float3 *a, float c)
{
	int idx = get_global_id(0);
	a[idx] -= c;
}

__kernel void floatsubfloat3(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = c - a[idx];
}

__kernel void float3mulfloat(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx]*c;
}

__kernel void float3mulfloatequal(__global float3 *a, float c)
{
	int idx = get_global_id(0);
	a[idx] *= c;
}

__kernel void floatmulfloat3(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = c*a[idx];
}

__kernel void float3divfloat(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = a[idx]/c;
}

__kernel void float3divfloatequal(__global float3 *a, float c)
{
	int idx = get_global_id(0);
	a[idx] /= c;
}

__kernel void floatdivfloat3(__global float3 *a, float c, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = c/a[idx];
}

__kernel void minusfloat3(__global float3 *a, __global float3 *dest)
{
	int idx = get_global_id(0);
	dest[idx] = -a[idx];
}

//} // extern "c"
