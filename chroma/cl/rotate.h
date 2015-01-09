#ifndef __ROTATE_H__
#define __ROTATE_H__

#include "linalg.h"
#include "matrix.h"

Matrix make_rotation_matrix(float phi, const float3 *n);
float3 rotate_with_vec(const float3 *a, float phi, const float3 *n);
float3 rotate_with_matrix(const float3 *a, float phi, const float3 *n);
Matrix identity_matrix( const float c);

Matrix identity_matrix( const float c) {
  return make_matrix( c, 0.0f, 0.0f, 
		      0.0f, c, 0.0f,
		      0.0f, 0.0f, c );
}

Matrix make_rotation_matrix(float phi, const float3 *n)
{
  float cos_phi = cos(phi);
  float sin_phi = sin(phi);
  Matrix I = identity_matrix( cos_phi );
  Matrix m1 = outer(n,n);
  scale( &m1, 1-cos_phi );
  Matrix m2 = make_matrix(0,n->z,-n->y,-n->z,0,n->x,n->y,-n->x,0);
  scale( &m2, sin_phi );
  add_to_elems( &I, &m1 );
  add_to_elems( &I, &m2 );
  return I;
  //return IDENTITY_MATRIX*cos_phi + (1-cos_phi)*outer(n,n) +
  //sin_phi*make_matrix(0,n->z,-n->y,-n->z,0,n->x,n->y,-n->x,0);
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
float3 rotate_with_vec(const float3 *a, float phi, const float3 *n)
{
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);
    float3 v = (*a)*cos_phi;
    v += (*n)*dot(*a,*n)*(1.0f-cos_phi);
    v += cross(*a,*n)*sin_phi;

    return v;
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
float3 rotate_with_matrix(const float3 *a, float phi, const float3 *n)
{
  Matrix rot = make_rotation_matrix(phi,n);
  return multiplyvec( &rot, a );
}

#endif
