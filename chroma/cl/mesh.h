#ifndef __MESH_H__
#define __MESH_H__

#include "intersect.h"
#include "geometry.h"

//#include "stdio.h"

#define STACK_SIZE 1000

bool intersect_node(const float3 *neg_origin_inv_dir, const float3 *inv_dir,
		    __local Geometry *g, const Node *node, const float min_distance); // min_distance=-1.0f
bool intersect_node_special(const float3 *neg_origin_inv_dir, const float3 *inv_dir, 
			    const float3 *origin, const float3 *direction, const int* aligned_axis,
			    __local Geometry *g, const Node *node, const float min_distance);
int get_aligned_axis( const float3 *direction );
int intersect_mesh(const float3 *origin, const float3* direction, __local Geometry *g,
		   float *min_distance, int last_hit_triangle); // last_hit_triangle=-1


/* Tests the intersection between a ray and a node in the bounding volume
   hierarchy. If the ray intersects the bounding volume and `min_distance`
   is less than zero or the distance from `origin` to the intersection is
   less than `min_distance`, return true, else return false. */
bool intersect_node(const float3 *neg_origin_inv_dir, const float3 *inv_dir,
		    __local Geometry *g, const Node *node, const float min_distance)
{
    float distance_to_box;

    if (intersect_box(neg_origin_inv_dir, inv_dir, &(node->lower), &(node->upper),
		      &distance_to_box)) 
    {
      if (min_distance < 0.0f)
	return true;
      
      if (distance_to_box > min_distance)
	return false;

      return true;
    }
    else  {
      return false;
    }
}


/*
Differs from intersect_node in that the special case of axis aligned photon
directions is handled.  Without using this axis aligned photons never succeed
to complete intersect_mesh, as infinities conspire to get almost every 
intersect_box 
*/
bool intersect_node_special(const float3 *neg_origin_inv_dir, const float3 *inv_dir, 
			    const float3 *origin, const float3 *direction, const int* aligned_axis,
			    __local Geometry *g, const Node *node, const float min_distance)
{
  float distance_to_box;
  bool intersects ;

  if(*aligned_axis > 0 ) //  axis aligned photon special case 
    {
      intersects = intersect_box_axis_aligned( aligned_axis, origin, direction, &(node->lower), &(node->upper), &distance_to_box);
    }
  else   // 0 : not axis aligned
    {
      intersects = intersect_box(neg_origin_inv_dir, inv_dir, &(node->lower), &(node->upper), &distance_to_box);
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

int get_aligned_axis( const float3 *direction ) 
{
  int axis = 0 ;  
  if( direction->x == 0.0f && direction->y == 0.0f && direction->z != 0.0f ) // along z 
    { 
      axis = 3 ;
    }
  else if ( direction->x == 0.0f && direction->y != 0.0f && direction->z == 0.0f ) // along y
    {  
      axis = 2 ;
    }
  else if ( direction->x != 0.0f && direction->y == 0.0f && direction->z == 0.0f ) // along x
    {   
      axis = 1 ;
    }
  return axis ; 
}






/* Finds the intersection between a ray and `geometry`. If the ray does
   intersect the mesh and the index of the intersected triangle is not equal
   to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
   the intersection and return the index of the triangle which the ray
   intersected, else return -1. */
int intersect_mesh(const float3 *origin, const float3* direction, __local Geometry *g,
		   float *min_distance, int last_hit_triangle) // default for last_hit_triangle=-1
{
  int triangle_index = -1;

  // handle this as early as possible to minimise processing 
  int aligned_axis = get_aligned_axis( direction );   // 0:usual case not aligned, special case direction along axis 1:x 2:y 3:z

  float distance;
  *min_distance = -1.0f; // set default

  Node root = get_node(g, 0);

  float3 neg_origin_inv_dir = -(*origin) / (*direction);
  float3 inv_dir = 1.0f / (*direction);

  //if (!intersect_node(neg_origin_inv_dir, inv_dir, g, root, min_distance))
  if (!intersect_node_special(&neg_origin_inv_dir, &inv_dir, origin, direction, &aligned_axis, g, &root, (*min_distance)))
    return -1;

  unsigned int child_ptr_stack[STACK_SIZE];
  unsigned int nchild_ptr_stack[STACK_SIZE];
  child_ptr_stack[0] = root.child;
  nchild_ptr_stack[0] = root.nchild;

  int curr = 0;
  unsigned int count = 0;
  unsigned int tri_count = 0;
  unsigned int hitsame = 0 ;
  int maxcurr = 0 ;

  while (curr >= 0) 
    {
      unsigned int first_child = child_ptr_stack[curr];
      unsigned int nchild = nchild_ptr_stack[curr];
      curr--;

      for (unsigned int i=first_child; i < first_child + nchild; i++) {
	Node node = get_node(g, i);
	count++;

	//if (intersect_node(neg_origin_inv_dir, inv_dir, g, node, min_distance)) {
	if (intersect_node_special(&neg_origin_inv_dir, &inv_dir, origin, direction, &aligned_axis, g, &node, (*min_distance) )) {

	  if (node.nchild == 0) { /* leaf node */

	    // This node wraps a triangle
	    
	    if (node.child != (unsigned int)last_hit_triangle) {
	      // Can't hit same triangle twice in a row
	      tri_count++;
	      Triangle t = get_triangle(g, node.child);			
	      if (intersect_triangle(origin, direction, &t, &distance)) 
		{

		  if (triangle_index == -1 || distance < (*min_distance)) 
		    {
		      triangle_index = node.child;
		      (*min_distance) = distance;
		    }    // if hit triangle is closer than previous hits
		  
		} // if hit triangle
			
	      
	      
	    } else {
	      hitsame++;
	    }    // if not hitting same triangle as last step
	    
	  } else {
	    curr++;
	    child_ptr_stack[curr] = node.child;
	    nchild_ptr_stack[curr] = node.nchild;
	  }  // leaf or internal node?
	} // hit node?
	    
	maxcurr = max( maxcurr, curr );

	if (curr >= STACK_SIZE) {
	  printf("warning: intersect_mesh() aborted; node > tail\n");
	  break;
	}
      }    // loop over children, starting with first_child

    }       // while nodes on stack

   // if (blockIdx.x == 0 && threadIdx.x == 0) {
   //   printf("node gets: %d\n", count);
   //   printf("triangle count: %d\n", tri_count);
   // }

  //printf("node gets: %d triangle count: %d maxcurr: %d hitsame: %d \n", count, tri_count, maxcurr, hitsame );
  
  return triangle_index;
}


__kernel void distance_to_mesh(int nthreads, __global float3 *_origin, __global float3 *_direction, __global float* _distance,
			       //__global Geometry *g)
			       __global float3 *vertices, __global uint3 *triangles, 
			       __global unsigned int *material_codes, __global unsigned int *colors,
			       __global uint4 *primary_nodes, __global uint4 *extra_nodes,
			       float3 world_origin, float world_scale, int nprimary_nodes ) {
  __local Geometry sg;

  if (get_local_id(0) == 0) {
    // fill geometry struct
    //sg = *g;
    sg.vertices = vertices;
    sg.triangles = triangles;
    sg.material_codes = material_codes;
    sg.colors = colors;
    sg.primary_nodes = primary_nodes;
    sg.extra_nodes = extra_nodes;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);

  if (id >= nthreads)
    return;

  float3 origin = _origin[id];
  float3 direction = _direction[id];
  direction /= length(direction);

  float distance = -1;
  int last_hit_triangle = -1;

  int triangle_index = intersect_mesh(&origin, &direction, &sg, &distance, last_hit_triangle);

  if (triangle_index != -1)
    _distance[id] = distance;
}


__kernel void color_solids(int first_triangle, int nthreads, __global int *solid_id_map,
			   __global unsigned int *solid_hit, __global unsigned int *solid_colors, // solid_hit changed from bool to int. Opencl does not allow bool args in kernels
			   __global float3 *vertices, __global uint3 *triangles, 
			   __global unsigned int *material_codes, __global unsigned int *colors,
			   __global uint4 *primary_nodes, __global uint4 *extra_nodes,
			   float3 world_origin, float world_scale, int nprimary_nodes ) {
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);

  if (id >= nthreads)
    return;

  int triangle_id = first_triangle + id;
  int solid_id = solid_id_map[triangle_id];
  if (solid_hit[solid_id]==1)
    colors[triangle_id] = solid_colors[solid_id];
}


#endif
