#ifndef __MATRIX_H__
#define __MATRIX_H__

typedef struct Matrix
{
  float a00;
  float a01;
  float a02;
  float a10;
  float a11;
  float a12;
  float a20;
  float a21;
  float a22;
} Matrix;

// Declarations
Matrix make_matrix(float a00, float a01, float a02,
		   float a10, float a11, float a12,
		   float a20, float a21, float a22);
Matrix make_matrix_fromvecs(const float3 *u1, const float3 *u2, const float3 *u3);
Matrix neg(const Matrix *m);
float3 multiplyvec(const Matrix *m, const float3 *a);
Matrix add_elems(const Matrix *m, const Matrix *n);
void add_to_elems(Matrix* m, const Matrix* n);
Matrix subtract_elems(const Matrix *m, const Matrix *n);
void subtract_from_elems(Matrix *m, const Matrix* n);
Matrix multiply(const Matrix *m, const Matrix *n);
Matrix add_scalar(const Matrix *m, const float c);
void add_to_scalar(Matrix *m, const float c);
Matrix subtract_scalar (const Matrix *m, const float c);
void subtract_from_scalar(Matrix *m, const float c);
Matrix subtract_with_scalar(const float c, const Matrix *m);
Matrix multiply_scalar(const Matrix *m, const float c);
void scale(Matrix *m, const float c);
Matrix divide_scalar(const Matrix *m, const float c);
void divide(Matrix *m, const float c);
float det(const Matrix *m);
Matrix inv(const Matrix *m);
Matrix outer(const float3 *a, const float3 *b);

// Defintions

Matrix make_matrix(float a00, float a01, float a02,
		   float a10, float a11, float a12,
		   float a20, float a21, float a22)
{
  Matrix m = {a00, a01, a02, a10, a11, a12, a20, a21, a22};
  return m;
}

Matrix make_matrix_fromvecs(const float3 *u1, const float3 *u2, const float3 *u3)
{
  Matrix m = {u1->x, u2->x, u3->x, u1->y, u2->y, u3->y, u1->z, u2->z, u3->z};
  return m;
}

Matrix neg(const Matrix *m)
{
  return make_matrix(-m->a00, -m->a01, -m->a02,
		     -m->a10, -m->a11, -m->a12,
		     -m->a20, -m->a21, -m->a22);
}

float3 multiplyvec(const Matrix *m, const float3 *a)
{
  return (float3) (m->a00*a->x + m->a01*a->y + m->a02*a->z,
		   m->a10*a->x + m->a11*a->y + m->a12*a->z,
		   m->a20*a->x + m->a21*a->y + m->a22*a->z);
}

Matrix add_elems(const Matrix *m, const Matrix *n)
{
  return make_matrix(m->a00+n->a00, m->a01+n->a01, m->a02+n->a02,
		     m->a10+n->a10, m->a11+n->a11, m->a12+n->a12,
		     m->a20+n->a20, m->a21+n->a21, m->a22+n->a22);
}

void add_to_elems(Matrix* m, const Matrix* n)
{
    m->a00 += n->a00;
    m->a01 += n->a01;
    m->a02 += n->a02;
    m->a10 += n->a10;
    m->a11 += n->a11;
    m->a12 += n->a12;
    m->a20 += n->a20;
    m->a21 += n->a21;
    m->a22 += n->a22;
}

Matrix subtract_elems(const Matrix *m, const Matrix *n)
{
  return make_matrix(m->a00-n->a00, m->a01-n->a01, m->a02-n->a02,
		     m->a10-n->a10, m->a11-n->a11, m->a12-n->a12,
		     m->a20-n->a20, m->a21-n->a21, m->a22-n->a22);
}

void subtract_from_elems(Matrix *m, const Matrix* n)
{
    m->a00 -= n->a00;
    m->a01 -= n->a01;
    m->a02 -= n->a02;
    m->a10 -= n->a10;
    m->a11 -= n->a11;
    m->a12 -= n->a12;
    m->a20 -= n->a20;
    m->a21 -= n->a21;
    m->a22 -= n->a22;
}

Matrix multiply(const Matrix *m, const Matrix *n)
{
  return make_matrix(m->a00*n->a00 + m->a01*n->a10 + m->a02*n->a20,
		     m->a00*n->a01 + m->a01*n->a11 + m->a02*n->a21,
		     m->a00*n->a02 + m->a01*n->a12 + m->a02*n->a22,
		     m->a10*n->a00 + m->a11*n->a10 + m->a12*n->a20,
		     m->a10*n->a01 + m->a11*n->a11 + m->a12*n->a21,
		     m->a10*n->a02 + m->a11*n->a12 + m->a12*n->a22,
		     m->a20*n->a00 + m->a21*n->a10 + m->a22*n->a20,
		     m->a20*n->a01 + m->a21*n->a11 + m->a22*n->a21,
		     m->a20*n->a02 + m->a21*n->a12 + m->a22*n->a22);
}

Matrix add_scalar(const Matrix *m, const float c)
{
  return make_matrix(m->a00+c, m->a01+c, m->a02+c,
		     m->a10+c, m->a11+c, m->a12+c,
		     m->a20+c, m->a21+c, m->a22+c);
}

void add_to_scalar(Matrix *m, const float c)
{
    m->a00 += c;
    m->a01 += c;
    m->a02 += c;
    m->a10 += c;
    m->a11 += c;
    m->a12 += c;
    m->a20 += c;
    m->a21 += c;
    m->a22 += c;
}

/* Matrix operator+ (*const float *c, const Matrix *m) */
/* { */
/*     return make_matrix(c+m->a00, c+m->a01, c+m->a02, */
/* 		       c+m->a10, c+m->a11, c+m->a12, */
/* 		       c+m->a20, c+m->a21, c+m->a22); */
/* } */

Matrix subtract_scalar(const Matrix *m, const float c)
{
  return make_matrix(m->a00-c, m->a01-c, m->a02-c,
		     m->a10-c, m->a11-c, m->a12-c,
		     m->a20-c, m->a21-c, m->a22-c);
}

void subtract_from_scalar(Matrix *m, const float c)
{
    m->a00 -= c;
    m->a01 -= c;
    m->a02 -= c;
    m->a10 -= c;
    m->a11 -= c;
    m->a12 -= c;
    m->a20 -= c;
    m->a21 -= c;
    m->a22 -= c;
}

Matrix subtract_with_scalar(const float c, const Matrix *m)
{
    return make_matrix(c-m->a00, c-m->a01, c-m->a02,
		       c-m->a10, c-m->a11, c-m->a12,
		       c-m->a20, c-m->a21, c-m->a22);
}

Matrix multiply_scalar(const Matrix *m, const float c)
{
  return make_matrix(m->a00*(c), m->a01*(c), m->a02*(c),
		     m->a10*(c), m->a11*(c), m->a12*(c),
		     m->a20*(c), m->a21*(c), m->a22*(c));
}

void scale(Matrix *m, const float c)
{
  m->a00 *= c;
  m->a01 *= c;
  m->a02 *= c;
  m->a10 *= c;
  m->a11 *= c;
  m->a12 *= c;
  m->a20 *= c;
  m->a21 *= c;
  m->a22 *= c;
}

Matrix divide_scalar(const Matrix *m, const float c)
{
    return make_matrix(m->a00/(c), m->a01/(c), m->a02/(c),
		       m->a10/(c), m->a11/(c), m->a12/(c),
		       m->a20/(c), m->a21/(c), m->a22/(c));
}

void divide(Matrix *m, const float c)
{
    m->a00 /= (c);
    m->a01 /= (c);
    m->a02 /= (c);
    m->a10 /= (c);
    m->a11 /= (c);
    m->a12 /= (c);
    m->a20 /= (c);
    m->a21 /= (c);
    m->a22 /= (c);
}


float det(const Matrix *m)
{
    return m->a00*(m->a11*m->a22 - m->a12*m->a21) -
	   m->a10*(m->a01*m->a22 - m->a02*m->a21) +
	   m->a20*(m->a01*m->a12 - m->a02*m->a11);
}

Matrix inv(const Matrix *m)
{
  Matrix a = make_matrix(m->a11*m->a22 - m->a12*m->a21,
			 m->a02*m->a21 - m->a01*m->a22,
			 m->a01*m->a12 - m->a02*m->a11,
			 m->a12*m->a20 - m->a10*m->a22,
			 m->a00*m->a22 - m->a02*m->a20,
			 m->a02*m->a10 - m->a00*m->a12,
			 m->a10*m->a21 - m->a11*m->a20,
			 m->a01*m->a20 - m->a00*m->a21,
			 m->a00*m->a11 - m->a01*m->a10);
  float d = det(m);
  divide(&a, d);
  return a;
}

Matrix outer(const float3 *a, const float3 *b)
{
  return make_matrix(a->x*b->x, a->x*b->y, a->x*b->z,
		     a->y*b->x, a->y*b->y, a->y*b->z,
		     a->z*b->x, a->z*b->y, a->z*b->z);
}

#endif
