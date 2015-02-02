#ifndef __MESH_H__
#define __MESH_H__

#include "intersect.h"
#include "geometry.h"

#include "stdio.h"

#define STACK_SIZE 1000

/* Tests the intersection between a ray and a node in the bounding volume
   hierarchy. If the ray intersects the bounding volume and `min_distance`
   is less than zero or the distance from `origin` to the intersection is
   less than `min_distance`, return true, else return false. */
__device__ bool
intersect_node(const float3 &neg_origin_inv_dir, const float3 &inv_dir,
	       Geometry *g, const Node &node, const float min_distance=-1.0f)
{
    float distance_to_box;

    if (intersect_box(neg_origin_inv_dir, inv_dir, node.lower, node.upper,
		      distance_to_box)) 
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


/*
Differs from intersect_node in that the special case of axis aligned photon
directions is handled.  Without using this axis aligned photons never succeed
to complete intersect_mesh, as infinities conspire to get almost every 
intersect_box 
*/
__device__ bool
intersect_node_special(const float3 &neg_origin_inv_dir, const float3 &inv_dir, 
                       const float3 &origin, const float3 &direction, const int& aligned_axis,
	       Geometry *g, const Node &node, const float min_distance=-1.0f)
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






__device__ int 
get_aligned_axis( const float3 &direction ) 
{
    int axis = 0 ;  
    if( direction.x == 0. && direction.y == 0. && direction.z != 0. ) // along z 
    { 
         axis = 3 ;
    }
    else if ( direction.x == 0. && direction.y != 0. && direction.z == 0. ) // along y
    {  
         axis = 2 ;
    }
    else if ( direction.x != 0. && direction.y == 0. && direction.z == 0. ) // along x
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
__device__ int
intersect_mesh(const float3 &origin, const float3& direction, Geometry *g,
	       float &min_distance, int last_hit_triangle = -1)
{
    int triangle_index = -1;

    // handle this as early as possible to minimise processing 
    int aligned_axis = get_aligned_axis( direction );   // 0:usual case not aligned, special case direction along axis 1:x 2:y 3:z

    float distance;
    min_distance = -1.0f;

    Node root = get_node(g, 0);

    float3 neg_origin_inv_dir = -origin / direction;
    float3 inv_dir = 1.0f / direction;

    //if (!intersect_node(neg_origin_inv_dir, inv_dir, g, root, min_distance))
    if (!intersect_node_special(neg_origin_inv_dir, inv_dir, origin, direction, aligned_axis, g, root, min_distance))
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
	        if (intersect_node_special(neg_origin_inv_dir, inv_dir, origin, direction, aligned_axis, g, node, min_distance)) {

		        if (node.nchild == 0) { /* leaf node */

		            // This node wraps a triangle

		            if (node.child != last_hit_triangle) {
			             // Can't hit same triangle twice in a row
			             tri_count++;
			             Triangle t = get_triangle(g, node.child);			
			             if (intersect_triangle(origin, direction, t, distance)) 
                         {

			                 if (triangle_index == -1 || distance < min_distance) 
                             {
				                 triangle_index = node.child;
				                 min_distance = distance;
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
#if __CUDA_ARCH__ >= 200
	    	    printf("warning: intersect_mesh() aborted; node > tail\n");
#endif
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

__device__ int
intersect_mesh_nvidia( Photon& p, Geometry* g ) 
{

  int triangle_index = -1;

  Node root = get_node(g, 0);
  
  unsigned int child_ptr_stack[STACK_SIZE];
  unsigned int nchild_ptr_stack[STACK_SIZE];
  child_ptr_stack[0] = root.child;
  nchild_ptr_stack[0] = root.nchild;
  
  int curr = 0;

  unsigned int count = 0;
  unsigned int tri_count = 0;
  unsigned int hitsame = 0 ;
  int maxcurr = 0 ;
  float dist_to_tri = -1.0f;
  float tmaxbox = -1.0f;
  float tminbox = -1.0f;

  while (curr >= 0)  {

    unsigned int first_child = child_ptr_stack[curr];
    unsigned int nchild = nchild_ptr_stack[curr];
    curr--;
    
    for (unsigned int i=first_child; i < first_child + nchild; i++) {
      Node node = get_node(g, i);
      count++;
      
      bool intersect_node = intersect_box_nvidia( p.position, p.diection, p.invdir, p.ood,
						  node.lower, node.upper, p.hitT, p.tmin, tminbox, tmaxbox );
      if (intersect_node) {
	// could I cut on nodes further than previously hit triangles?
	
	if (node.nchild == 0) { /* leaf node */
	  
	  // This node wraps a triangle
	  if (node.child != last_hit_triangle) {
	    // Can't hit same triangle twice in a row
	    tri_count++;
	    Triangle t = get_triangle(g, node.child);
	    bool intersect_tri = intersect_triangle( p, t, dist_to_tri);
	    if (intersect_tri )
	      {
		if (triangle_index == -1 || dist_to_tri < p.hitT ) 
		  {
		    triangle_index = node.child;
		    p.hitT = dist_to_tri;
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
#if __CUDA_ARCH__ >= 200
	printf("warning: intersect_mesh() aborted; node > tail\n");
#endif
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

extern "C"
{

__global__ void
distance_to_mesh(int nthreads, float3 *_origin, float3 *_direction,
		 Geometry *g, float *_distance)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    g = &sg;

    float3 origin = _origin[id];
    float3 direction = _direction[id];
    direction /= norm(direction);

    float distance;

    int triangle_index = intersect_mesh(origin, direction, g, distance);

    if (triangle_index != -1)
	_distance[id] = distance;
}

__global__ void
color_solids(int first_triangle, int nthreads, int *solid_id_map,
	     bool *solid_hit, unsigned int *solid_colors, Geometry *g)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int triangle_id = first_triangle + id;
    int solid_id = solid_id_map[triangle_id];
    if (solid_hit[solid_id])
	g->colors[triangle_id] = solid_colors[solid_id];
}

} // extern "C"

#endif
