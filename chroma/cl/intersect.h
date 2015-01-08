//-*-c-*-

#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include "linalg.h"
#include "matrix.h"
#include "geometry.h"
//#include "math_constants.h"

#define EPSILON 0.0f

bool intersect_triangle(const float3 *origin, const float3 *direction, const Triangle *triangle, float* distance);
bool intersect_box(const float3 *neg_origin_inv_dir, const float3 *inv_dir,
		   const float3 *lower_bound, const float3 *upper_bound,
		   float* distance_to_box);
bool intersect_box_axis_aligned( const int* aligned_axis, const float3 *origin, const float3 *direction, 
				 const float3 *lower_bound, const float3 *upper_bound,
				 float* distance_to_box);


/* Tests the intersection between a ray and a triangle.
   If the ray intersects the triangle, set `distance` to the distance from
   `origin` to the intersection and return true, else return false.
   `direction` must be normalized to one. */
bool intersect_triangle(const float3 *origin, const float3 *direction,
			const Triangle *triangle, float* distance)
{
	float3 m1 = triangle->v1-triangle->v0;
	float3 m2 = triangle->v2-triangle->v0;
	float3 m3 = -(*direction);

	Matrix m = make_matrix_fromvecs(&m1, &m2, &m3);
	
	float determinant = det(&m);

	if (determinant == 0.0f)
		return false;

	float3 b = (*origin)-triangle->v0;

	float u1 = ((m.a11*m.a22 - m.a12*m.a21)*b.x +
		    (m.a02*m.a21 - m.a01*m.a22)*b.y +
		    (m.a01*m.a12 - m.a02*m.a11)*b.z)/determinant;

	if (u1 < -EPSILON || u1 > 1.0f)
		return false;

	float u2 = ((m.a12*m.a20 - m.a10*m.a22)*b.x +
		    (m.a00*m.a22 - m.a02*m.a20)*b.y +
		    (m.a02*m.a10 - m.a00*m.a12)*b.z)/determinant;

	if (u2 < -EPSILON || u2 > 1.0f)
		return false;

	float u3 = ((m.a10*m.a21 - m.a11*m.a20)*b.x +
		    (m.a01*m.a20 - m.a00*m.a21)*b.y +
		    (m.a00*m.a11 - m.a01*m.a10)*b.z)/determinant;

	if (u3 <= 0.0f || (1.0f-u1-u2) < -EPSILON)
		return false;

	(*distance) = u3;

	return true;
}

/* Tests the intersection between a ray and an axis-aligned box defined by
   an upper and lower bound. If the ray intersects the box, set
   `distance_to_box` to the distance from `origin` to the intersection and
   return true, else return false. `direction` must be normalized to one.

    Source: Optimizing ray tracing for CUDA by Hannu Saransaari
    https://wiki.aalto.fi/download/attachments/40023967/gpgpu.pdf
*/
// INFINITY is already defined elsewhere
#define CHROMA_INFINITY __int_as_float(0x7f800000)

bool intersect_box(const float3 *neg_origin_inv_dir, const float3 *inv_dir,
		   const float3 *lower_bound, const float3 *upper_bound,
		   float* distance_to_box)
{
    //
    // http://tavianator.com/2011/05/fast-branchless-raybounding-box-intersections/
    // http://tavianator.com/cgit/dimension.git/tree/libdimension/bvh.c#n191
    //
	//float tmin = 0.0f, tmax = CHROMA_INFINITY;
  //float tmin = -CUDART_INF_F, tmax = CUDART_INF_F ;
  float tmin = -INFINITY;
  float tmax = INFINITY;
    
  float t0, t1;

    // X
    //if (isfinite(inv_dir.x)) {
    t0 = lower_bound->x * inv_dir->x + neg_origin_inv_dir->x;
    t1 = upper_bound->x * inv_dir->x + neg_origin_inv_dir->x;
	
    tmin = max(tmin, min(t0, t1));
    tmax = min(tmax, max(t0, t1));
    //}

    // Y
    //if (isfinite(inv_dir.y)) {
    t0 = lower_bound->y * inv_dir->y + neg_origin_inv_dir->y;
    t1 = upper_bound->y * inv_dir->y + neg_origin_inv_dir->y;
	
    tmin = max(tmin, min(t0, t1));
    tmax = min(tmax, max(t0, t1));
    //}

    // Z
    //if (isfinite(inv_dir.z)) {
    t0 = lower_bound->z * inv_dir->z + neg_origin_inv_dir->z;
    t1 = upper_bound->z * inv_dir->z + neg_origin_inv_dir->z;
    
    tmin = max(tmin, min(t0, t1));
    tmax = min(tmax, max(t0, t1));
    //}
    
    if (max(0.0f,tmin) > tmax)
      return false;
    
    (*distance_to_box) = tmin;
    
    return true;
}

bool intersect_box_axis_aligned( const int* aligned_axis, const float3 *origin, const float3 *direction, 
				 const float3 *lower_bound, const float3 *upper_bound,
				 float* distance_to_box)
{
  // handling edge case of rays travelling along axes
  if( *aligned_axis == 3 ){ // along z 
    (*distance_to_box) = origin->z - lower_bound->z ;
    return ( origin->x > lower_bound->x 
	     && origin->x < upper_bound->x 
	     && origin->y > lower_bound->y 
	     && origin->y < upper_bound->y );
  }
  else if ( *aligned_axis == 2 ){  // along y
    (*distance_to_box) = origin->y - lower_bound->y ;
    return ( origin->x > lower_bound->x 
	     && origin->x < upper_bound->x 
	     && origin->z > lower_bound->z 
	     && origin->z < upper_bound->z );
  }
  else if ( *aligned_axis == 1){   // along x
    (*distance_to_box) = origin->x - lower_bound->x ;
    return ( origin->y > lower_bound->y 
	     && origin->y < upper_bound->y 
	     && origin->z > lower_bound->z 
	     && origin->z < upper_bound->z );
  }
  else {                 // not applicable
    return false ;
  }
}






#endif
