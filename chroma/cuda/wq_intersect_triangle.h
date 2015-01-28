//-*-c++-*-

#ifndef __INTERSECT_TRI_H__
#define __INTERSECT_TRI_H__

#include "geometry_types.h"
#include "linalg.h"
#include "matrix.h"

/* Tests the intersection between a ray and a triangle. */
/*   If the ray intersects the triangle, set `distance` to the distance from */
/*   `origin` to the intersection and return true, else return false. */
/*   `direction` must be normalized to one.  */

__device__ bool intersect_triangle(const float3 &origin, const float3 &direction, const Triangle &triangle, float &distance);

#define EPSILON 0.0f

__device__
int intersect_triangle(const float3 &origin, const float3 &direction, const Triangle &triangle, float &distance)
{
  float3 m1 = triangle.v1-triangle.v0;
  float3 m2 = triangle.v2-triangle.v0;
  float3 m3 = -direction;

  Matrix m = make_matrix(m1, m2, m3);
	
  float determinant = det(m);

  if (determinant == 0.0f)
    return 0;

  float3 b = origin-triangle.v0;
  
  float u1 = ((m.a11*m.a22 - m.a12*m.a21)*b.x +
	      (m.a02*m.a21 - m.a01*m.a22)*b.y +
	      (m.a01*m.a12 - m.a02*m.a11)*b.z)/determinant;

  if (u1 < -EPSILON || u1 > 1.0f)
    return 0;
  
  float u2 = ((m.a12*m.a20 - m.a10*m.a22)*b.x +
	      (m.a00*m.a22 - m.a02*m.a20)*b.y +
	      (m.a02*m.a10 - m.a00*m.a12)*b.z)/determinant;

  if (u2 < -EPSILON || u2 > 1.0f)
    return 0;
  
  float u3 = ((m.a10*m.a21 - m.a11*m.a20)*b.x +
	      (m.a01*m.a20 - m.a00*m.a21)*b.y +
	      (m.a00*m.a11 - m.a01*m.a10)*b.z)/determinant;
  
  if (u3 <= 0.0f || (1.0f-u1-u2) < -EPSILON)
    return 0;
  
  distance = u3;
  
  return 1;
}

#endif
