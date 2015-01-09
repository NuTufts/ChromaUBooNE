//-*-c++-*-

#include "mesh.h"

// we don't put together the full geometry struct, as all we are doing is calculating distance to each mesh
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

  
