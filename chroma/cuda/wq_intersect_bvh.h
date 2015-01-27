#include "geometry_types.h"
#include "intersect.h"

__device__ int get_aligned_axis( const float3 &direction );
__device__ bool intersect_node_special(const float3 &neg_origin_inv_dir, const float3 &inv_dir, 
				       const float3 &origin, const float3 &direction, const int aligned_axis, 
				       const Node &node, const float min_distance);
__device__ int intersect_internal_node(const float3 &origin, const float3& direction, Node &node);


// ------------------------------------------------------------------------------------------------------------------

__device__ int get_aligned_axis( const float3 &direction ) 
{
  int axis = 0 ;  
  if( direction.x == 0.0f && direction.y == 0.0f && direction.z != 0.0f ) // along z 
    { 
      axis = 3 ;
    }
  else if ( direction.x == 0.0f && direction.y != 0.0f && direction.z == 0.0f ) // along y
    {  
      axis = 2 ;
    }
  else if ( direction.x != 0.0f && direction.y == 0.0f && direction.z == 0.0f ) // along x
    {   
      axis = 1 ;
    }
  return axis ; 
}

__device__ bool intersect_node_special(const float3 &neg_origin_inv_dir, const float3 &inv_dir, 
				       const float3 &origin, const float3 &direction, const int aligned_axis,
				       const Node &node, const float min_distance)
{
  float distance_to_box;
  bool intersects ;

  if(aligned_axis > 0 ) //  axis aligned photon special case 
    {
      intersects = intersect_box_axis_aligned( aligned_axis, origin, direction, node.lower, node.upper, distance_to_box);
    }
  else   // 0 : not axis aligned
    {
      intersects = intersect_box(neg_origin_inv_dir, inv_dir, node.lower, node.upper, distance_to_box);
    } 
  
  if (intersects) 
    {
      if (min_distance < 0.0f)
	return true;

      if (distance_to_box > min_distance)
	return false;

      return true;
    }
  else 
    {
      return false;
    }
}

__device__ 
int intersect_internal_node(const float3 &origin, const float3 &direction, Node &node) {
  
  float3 neg_origin_inv_dir = -origin / direction;
  float3 inv_dir = 1.0f / direction;
  float min_distance = -1.0f;

  // special case intersection test: lot's of ifs...unroll this or find better test?z
  // handle this as early as possible to minimise processing 
  int aligned_axis = get_aligned_axis( direction );   // 0:usual case not aligned, special case direction along axis 1:x 2:y 3:z
  if ( intersect_node_special(neg_origin_inv_dir, inv_dir, origin, direction, aligned_axis, node, min_distance) )
    return 1; // intersects
  else
    return 0; // misses
}
