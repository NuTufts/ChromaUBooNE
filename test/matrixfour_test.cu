//-*-c-*-

#include "matrixfour.h"

__device__ MatrixFour array2matrix(float *a)
{
	return make_matrixfour(a[0], a[1], a[2], a[3], 
                           a[4], a[5], a[6], a[7], 
                           a[8], a[9], a[10], a[11],
                           a[12], a[13], a[14], a[15]);
}

__device__ void matrix2array(const MatrixFour &m, float *a)
{
	a[0] = m.a00;
	a[1] = m.a01;
	a[2] = m.a02;
	a[3] = m.a03;

	a[4] = m.a10;
	a[5] = m.a11;
	a[6] = m.a12;
	a[7] = m.a13;

	a[8] = m.a20;
	a[9] = m.a21;
	a[10] = m.a22;
	a[11] = m.a23;

	a[12] = m.a30;
	a[13] = m.a31;
	a[14] = m.a32;
	a[15] = m.a33;


}

extern "C"
{

__global__ void det(float *a, float *dest)
{
	MatrixFour m = array2matrix(a);
	dest[0] = det(m);
}

__global__ void inv(float *a, float *dest)
{
	MatrixFour m = array2matrix(a);
	matrix2array(inv(m), dest);
}

__global__ void minusmatrix(float *a, float *dest)
{
	matrix2array(-array2matrix(a), dest);
}

__global__ void matrixadd(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)+array2matrix(b), dest);
}

__global__ void matrixsub(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)-array2matrix(b), dest);
}

__global__ void matrixmul(float *a, float *b, float *dest)
{
	matrix2array(array2matrix(a)*array2matrix(b), dest);
}

__global__ void multiply(float *a, float4 *x, float4 *dest)
{
	dest[0] = array2matrix(a)*x[0];
}

__global__ void matrixaddfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)+c, dest);
}

__global__ void matrixsubfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)-c, dest);
}

__global__ void matrixmulfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)*c, dest);
}

__global__ void matrixdivfloat(float *a, float c, float *dest)
{
	matrix2array(array2matrix(a)/c, dest);
}

__global__ void floataddmatrix(float *a, float c, float *dest)
{
	matrix2array(c+array2matrix(a), dest);
}

__global__ void floatsubmatrix(float *a, float c, float *dest)
{
	matrix2array(c-array2matrix(a), dest);
}

__global__ void floatmulmatrix(float *a, float c, float *dest)
{
	matrix2array(c*array2matrix(a), dest);
}

__global__ void floatdivmatrix(float *a, float c, float *dest)
{
	matrix2array(c/array2matrix(a), dest);
}

__global__ void matrixaddequals(float *a, float *b)
{
	MatrixFour m = array2matrix(a);
	m += array2matrix(b);
	matrix2array(m,a);
}

__global__ void matrixsubequals(float *a, float *b)
{
	MatrixFour m = array2matrix(a);
	m -= array2matrix(b);
	matrix2array(m,a);
}

__global__ void matrixaddequalsfloat(float *a, float c)
{
	MatrixFour m = array2matrix(a);
	m += c;
	matrix2array(m,a);
}

__global__ void matrixsubequalsfloat(float *a, float c)
{
	MatrixFour m = array2matrix(a);
	m -= c;
	matrix2array(m,a);
}

__global__ void matrixmulequalsfloat(float *a, float c)
{
	MatrixFour m = array2matrix(a);
	m *= c;
	matrix2array(m,a);
}

__global__ void matrixdivequalsfloat(float *a, float c)
{
	MatrixFour m = array2matrix(a);
	m /= c;
	matrix2array(m,a);
}

__global__ void outer(float4 a, float4 b, float* dest)
{
	matrix2array(outer(a,b), dest);
}

} // extern "c"
