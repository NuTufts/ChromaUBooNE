//-*-c++-*-

#include "geometry_types.h"
#include "geometry.h"
//#include "linalg.h"
#include "physical_constants.h"
#include "sorting.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float3 vminf(const float3 *a, const float3 *b);
float3 vmaxf(const float3 *a, const float3 *b);
unsigned long spread3_16(unsigned int input);
unsigned long spread2_16(unsigned int input);
unsigned int quantize(float v, float world_origin, float world_scale);
uint3 quantize3(float3 v, float3 world_origin, float world_scale);
uint3 quantize3_cyl(float3 v, float3 world_origin, float world_scale);
uint3 quantize3_sph(float3 v, float3 world_origin, float world_scale);
uint4 node_union(const uint4* a, const uint4* b);
unsigned long surface_half_area(const uint4 *node);
unsigned long ullmin( unsigned long a, unsigned long b );

// There is a lot of morton code stuff.
// OpenCL doesnt support 128 bits.
// So I change all unsigned long long into unsigned long long -- made it 64-bits
// Just going to pray it works.

// OpenCL has native versions of these functions
// // Vector utility functions
float3 vminf(const float3 *a, const float3 *b)
{
  //return make_float3(fmin((*a).x, (*b).x), fmin((*a).y, (*b).y), fmin((*a).z, (*b).z));
  return (float3)( fmin((*a).x, (*b).x), fmin((*a).y, (*b).y), fmin((*a).z, (*b).z) );
}

float3 vmaxf(const float3 *a, const float3 *b)
{
  //return make_float3(fmax((*a).x, (*b).x), fmax((*a).y, (*b).y), fmax((*a).z, (*b).z));
  return (float3)( fmax((*a).x, (*b).x), fmax((*a).y, (*b).y), fmax((*a).z, (*b).z) );
}

// uint3 min(const uint3 &a, const uint3 &b)
// {
//   return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
// }

// uint3 max(const uint3 &a, const uint3 &b)
// {
//   return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
// }

// uint3 operator+ (const uint3 &a, const unsigned int &b)
// {
//   return make_uint3(a.x + b, a.y + b, a.z + b);
// }

unsigned long ullmin( unsigned long a, unsigned long b ) {
  if (a<b) 
    return a;
  else
    return b;
}

// spread out the first 16 bits in x to occupy every 3rd slot in the return value
unsigned long spread3_16(unsigned int input)
{
  // method from http://stackoverflow.com/a/4838734
  unsigned long x = input;
  x = (x | (x << 16)) & 0x00000000FF0000FFul;
  x = (x | (x << 8)) & 0x000000F00F00F00Ful;
  x = (x | (x << 4)) & 0x00000C30C30C30C3ul;
  x = (x | (x << 2)) & 0X0000249249249249ul;
  
  return x;
}

unsigned long spread2_16(unsigned int input)
{
  unsigned long x = input;
  x = (x | (x << 16)) & 0x000000ff00ff00fful;
  x = (x | (x <<  8)) & 0x00000f0f0f0f0f0ful;
  x = (x | (x <<  4)) & 0x0000333333333333ul;
  x = (x | (x <<  2)) & 0x0000555555555555ul;
  return x;
}


unsigned int quantize(float v, float world_origin, float world_scale)
{
  // truncate!
  // returns a scaled 32 bit unsigned integer
  double world_origin0 = world_origin;
  double v0 = v;
  double world_scale0 = world_scale;
  double scaled_pos = ((v0 - world_origin0) / world_scale0);
  return (unsigned int) scaled_pos;
//return (unsigned int) ((v - world_origin) / world_scale);
}

uint3 quantize3(float3 v, float3 world_origin, float world_scale)
{
  return (uint3) (quantize(v.x, world_origin.x, world_scale),
		  quantize(v.y, world_origin.y, world_scale),
		  quantize(v.z, world_origin.z, world_scale));
}

uint3 quantize3_cyl(float3 v, float3 world_origin, float world_scale)
{
  float3 rescaled_v = (v - world_origin) / world_scale / sqrt(3.0f); 
  unsigned int z = rescaled_v.z;
  rescaled_v.z = 0.0f;
  //unsigned int rho = (unsigned int) norm(rescaled_v);
  unsigned int rho = (unsigned int) length(rescaled_v); // using opencl native
  unsigned int phi = (unsigned int) ((atan2(v.y, v.x)/PI/2.0f + 1.0f) * 65535.0f);

  return (uint3) (rho, phi, z);
}

uint3 quantize3_sph(float3 v, float3 world_origin, float world_scale)
{
  float3 rescaled_v = (v - world_origin) / world_scale;

  //unsigned int r = (unsigned int) (norm(rescaled_v) / sqrt(3.0f));
  unsigned int r = (unsigned int) (length(rescaled_v) / sqrt(3.0f)); // switching to opencl native

  unsigned int phi = (unsigned int) ((atan2(rescaled_v.y, rescaled_v.x)/PI/2.0f + 1.0f) * 65535.0f);
  
  //unsigned int theta = (unsigned int) (acosf(rescaled_v.z / norm(rescaled_v)) / PI * 65535.0f);
  unsigned int theta = (unsigned int) (acos(rescaled_v.z / length(rescaled_v)) / PI * 65535.0f); // switching to opencl native
 
  return (uint3) (r, theta, phi);
}

uint4 node_union(const uint4 *a, const uint4 *b)
{
  uint3 lower = (uint3) (min((*a).x & 0xFFFF, (*b).x & 0xFFFF),
                           min((*a).y & 0xFFFF, (*b).y & 0xFFFF),
                           min((*a).z & 0xFFFF, (*b).z & 0xFFFF));
  uint3 upper = (uint3) (max((*a).x >> 16, (*b).x >> 16),
                           max((*a).y >> 16, (*b).y >> 16),
                           max((*a).z >> 16, (*b).z >> 16));

  return (uint4) (upper.x << 16 | lower.x,
		  upper.y << 16 | lower.y,
		  upper.z << 16 | lower.z,
		  0);
}      



unsigned long surface_half_area(const uint4 *node)
{
  unsigned long x = ((*node).x >> 16) - ((*node).x & 0xFFFF);
  unsigned long y = ((*node).y >> 16) - ((*node).y & 0xFFFF);
  unsigned long z = ((*node).z >> 16) - ((*node).z & 0xFFFF);

  unsigned long surf = x*y + y*z + z*x;
  return surf;
}

//extern "C"
//{

__kernel void  node_area(unsigned int first_node,
			 unsigned int nnodes_this_round,
			 __global uint4 *nodes,
			 __global unsigned int *areas)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  //unsigned int thread_id = get_global_id(0);
  if (thread_id >= nnodes_this_round)
    return;
  
  unsigned int node_id = first_node + thread_id;
  uint4 node = nodes[node_id];
  //areas[node_id] = surface_half_area(&nodes[node_id]);
  areas[node_id] = surface_half_area(&node);
}


// Main Kernel that is called
__kernel void
make_leaves(unsigned int first_triangle,
	    unsigned int ntriangles, 
	    __global uint3 *triangles, __global float3 *vertices,
	    float3 world_origin, float world_scale,
	    __global uint4 *leaf_nodes, __global unsigned long *morton_codes)
  
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= ntriangles)
    return;
  
  unsigned int triangle_id = first_triangle + thread_id;
  
  // Find bounding corners and centroid
  uint3 triangle = triangles[triangle_id];
  float3 lower = vertices[triangle.x];
  float3 centroid = lower;
  float3 upper = lower;
  float3 temp_vertex;
  
  temp_vertex = vertices[triangle.y];
  lower = fmin(lower, temp_vertex); // fminf(lower, temp_vertex); // using builtin
  upper = fmax(upper, temp_vertex); // fmaxf(upper, temp_vertex); // using builtin
  centroid  += temp_vertex;
  
  temp_vertex = vertices[triangle.z];
  lower = fmin(lower, temp_vertex); // fminf(lower, temp_vertex);
  upper = fmax(upper, temp_vertex); // fmaxf(upper, temp_vertex);
  centroid  += temp_vertex;
  
  centroid /= 3.0f;
  
  // Quantize bounding corners and centroid
  uint3 q_lower = quantize3(lower, world_origin, world_scale);
  if (q_lower.x > 0) q_lower.x--;
  if (q_lower.y > 0) q_lower.y--;
  if (q_lower.z > 0) q_lower.z--;
  uint3 q_upper = quantize3(upper, world_origin, world_scale) + 1;
  uint3 q_centroid = quantize3(centroid, world_origin, world_scale);
  
  // Compute Morton code from quantized centroid
  unsigned long morton = 
    spread3_16(q_centroid.x) 
    | (spread3_16(q_centroid.y) << 1)
    | (spread3_16(q_centroid.z) << 2);
  
  // Write leaf and morton code
  uint4 leaf_node;
  leaf_node.x = q_lower.x | (q_upper.x << 16);
  leaf_node.y = q_lower.y | (q_upper.y << 16);
  leaf_node.z = q_lower.z | (q_upper.z << 16);
  leaf_node.w = triangle_id;
  
  leaf_nodes[triangle_id] = leaf_node;
  morton_codes[triangle_id] = morton;
  /*printf("id: %d, centroid: %.3f, %.3f, %.3f, qcent: %u, %u, %u  morton code: %lu\n", 
	 triangle_id, 
	 centroid.x, centroid.y, centroid.z, 
	 q_centroid.x, q_centroid.y, q_centroid.z,
	 morton );*/
}

__kernel void
reorder_leaves(unsigned int first_triangle,
	       unsigned int ntriangles,
	       __global uint4 *leaf_nodes_in, 
	       __global uint4 *leaf_nodes_out,
	       __global unsigned int *remap_order)  
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  //unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= ntriangles)
    return;
  
  unsigned int dest_id = first_triangle + thread_id;
  unsigned int source_id = remap_order[dest_id];
  
  leaf_nodes_out[dest_id] = leaf_nodes_in[source_id];
}

__kernel void
build_layer(unsigned int first_node,
	    unsigned int n_parent_nodes, 
	    unsigned int n_children_per_node,
	    __global uint4 *nodes,
	    unsigned int parent_layer_offset,
	    unsigned int child_layer_offset)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= n_parent_nodes)
    return;
  
  unsigned int parent_id = first_node + thread_id;
  unsigned int first_child = child_layer_offset + parent_id * n_children_per_node;
  
  // Load first child
  uint4 parent_node = nodes[first_child];
  uint3 lower = (uint3) (parent_node.x & 0xFFFF, parent_node.y & 0xFFFF, parent_node.z & 0xFFFF);
  uint3 upper = (uint3) (parent_node.x >> 16, parent_node.y >> 16, parent_node.z >> 16);
  
  
  // Scan remaining children
  unsigned int real_children = 1;
  for (unsigned int i=1; i < n_children_per_node; i++) {
    uint4 child_node = nodes[first_child + i];
    
    if (child_node.x == 0)
      break;  // Hit first padding node in list of children
    
    real_children++;
    
    uint3 child_lower = (uint3) (child_node.x & 0xFFFF, child_node.y & 0xFFFF, child_node.z & 0xFFFF);
    uint3 child_upper = (uint3) (child_node.x >> 16, child_node.y >> 16, child_node.z >> 16);
    
    lower = min(lower, child_lower);
    upper = max(upper, child_upper);
  }
  
  parent_node.w = (real_children << CHILD_BITS) | first_child;
  parent_node.x = upper.x << 16 | lower.x;
  parent_node.y = upper.y << 16 | lower.y;
  parent_node.z = upper.z << 16 | lower.z;
  
  nodes[parent_layer_offset + parent_id] = parent_node;
}

__kernel void
make_parents_detailed(unsigned int first_node,
		      unsigned int elements_this_launch, 
		      __global uint4 *child_nodes,
		      __global uint4 *parent_nodes,
		      __global int *first_children,
		      __global int *nchildren)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= elements_this_launch)
    return;
  
  unsigned int parent_id = first_node + thread_id;
  unsigned int first_child = first_children[parent_id];
  unsigned int nchild = nchildren[parent_id];
  
  // Load first child
  uint4 parent_node = child_nodes[first_child];
  uint3 lower = (uint3) (parent_node.x & 0xFFFF, parent_node.y & 0xFFFF, parent_node.z & 0xFFFF);
  uint3 upper = (uint3) (parent_node.x >> 16, parent_node.y >> 16, parent_node.z >> 16);
  
  // Scan remaining children
  for (unsigned int i=1; i < nchild; i++) {
    uint4 child_node = child_nodes[first_child + i];
    
    uint3 child_lower = (uint3) (child_node.x & 0xFFFF, child_node.y & 0xFFFF, child_node.z & 0xFFFF);
    uint3 child_upper = (uint3) (child_node.x >> 16, child_node.y >> 16, child_node.z >> 16);
    
    lower = min(lower, child_lower);
    upper = max(upper, child_upper);
  }
  
  parent_node.w = (nchild << CHILD_BITS) | first_child;
  parent_node.x = upper.x << 16 | lower.x;
  parent_node.y = upper.y << 16 | lower.y;
  parent_node.z = upper.z << 16 | lower.z;
  
  parent_nodes[parent_id] = parent_node;
}


__kernel void
make_parents(unsigned int first_node,
	     unsigned int elements_this_launch, 
	     unsigned int n_children_per_node,
	     __global uint4 *parent_nodes,
	     __global uint4 *child_nodes,
	     unsigned int child_id_offset,
	     unsigned int num_children)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= elements_this_launch)
    return;
  
  unsigned int parent_id = first_node + thread_id;
  unsigned int first_child = parent_id * n_children_per_node;
  
  if (first_child >= num_children)
    return;
  
  // Load first child
  uint4 parent_node = child_nodes[first_child];
  uint3 lower = (uint3) (parent_node.x & 0xFFFF, parent_node.y & 0xFFFF, parent_node.z & 0xFFFF);
  uint3 upper = (uint3) (parent_node.x >> 16, parent_node.y >> 16, parent_node.z >> 16);
  
  // Scan remaining children
  unsigned int real_children = 1;
  for (unsigned int i=1; i < n_children_per_node; i++) {
    if (first_child + i >= num_children)
      break;
    
    uint4 child_node = child_nodes[first_child + i];
    
    if (child_node.x == 0)
      break;  // Hit first padding node in list of children
    
    real_children++;
    
    uint3 child_lower = (uint3) (child_node.x & 0xFFFF, child_node.y & 0xFFFF, child_node.z & 0xFFFF);
    uint3 child_upper = (uint3) (child_node.x >> 16, child_node.y >> 16, child_node.z >> 16);
    
    lower = min(lower, child_lower);
    upper = max(upper, child_upper);
  }
  
  parent_node.w = (real_children << CHILD_BITS)
    | (first_child + child_id_offset);
  parent_node.x = upper.x << 16 | lower.x;
  parent_node.y = upper.y << 16 | lower.y;
  parent_node.z = upper.z << 16 | lower.z;
  
  parent_nodes[parent_id] = parent_node;
}

__kernel void
copy_and_offset(unsigned int first_node,
		unsigned int elements_this_launch, 
		unsigned int child_id_offset,
		unsigned int layer_start,
		__global uint4 *src_nodes,
		__global uint4 *dest_nodes)
  
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= elements_this_launch)
    return;
  
  unsigned int node_id = first_node + thread_id;
  uint4 src_node = src_nodes[node_id];
  
  unsigned int nchild = src_node.w >> CHILD_BITS;
  unsigned int child_id = src_node.w &  ~NCHILD_MASK;
  src_node.w = (nchild << CHILD_BITS) | (child_id + child_id_offset);
  
  dest_nodes[node_id+layer_start] = src_node;
}

__kernel void distance_to_prev(unsigned int first_node, 
			       unsigned int threads_this_round,
			       __global uint4 *node, 
			       __global unsigned int *area)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= threads_this_round)
    return;
  
  unsigned int node_id = first_node + thread_id;
  
  uint4 a = node[node_id - 1];
  uint4 b = node[node_id];
  uint4 u = node_union(&a, &b);
  area[node_id] = surface_half_area(&u);
}

__kernel void pair_area(unsigned int first_node,
			unsigned int threads_this_round,
			__global uint4 *node, __global unsigned long *area)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= threads_this_round)
    return;
  
  unsigned int node_id = first_node + thread_id;
  unsigned int child_id = node_id * 2;
  
  uint4 a = node[child_id];
  uint4 b = node[child_id+1];
  if (b.x == 0)
    b = a;
  
  uint4 u = node_union(&a, &b);
  
  area[node_id] = 2*surface_half_area(&u);
}

__kernel void distance_to(unsigned int first_node, 
			  unsigned int threads_this_round,
			  unsigned int target_index,
			  __global uint4 *node, 
			  __global unsigned int *area)
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_id >= threads_this_round)
    return;
  
  unsigned int node_id = first_node + thread_id;
  
  if (node_id == target_index) {
    area[node_id] = 0xFFFFFFFF;
  } else {
    uint4 a = node[target_index];
    uint4 b = node[node_id];
    uint4 u = node_union(&a, &b);
    
    area[node_id] = surface_half_area(&u);
  }
}

__kernel void min_distance_to( unsigned int first_node, unsigned int threads_this_round,
			       unsigned int target_index,
			       __global uint4 *node,
			       unsigned int block_offset,
			       __global unsigned long *min_area_block,
			       __global unsigned int *min_index_block,
			       __global unsigned int *flag)
{
  __local unsigned long min_area[128];
  __local unsigned long adjacent_area;
  
  target_index = get_group_id(1); //target_index += blockIdx.y;
  
  uint4 a = node[target_index];
  uint4 adj = node[target_index+1];
  
  if ( get_local_id(0)/*threadIdx.x*/ == 0) {
    uint4 u = node_union(&a, &adj/*node[target_index+1]*/);
    adjacent_area = surface_half_area(&u);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
  
  //unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int thread_id = get_local_size(0) * get_group_id(0) + get_local_id(0);
  
  unsigned int node_id = first_node + thread_id;
  
  if (thread_id >= threads_this_round)
    node_id = target_index;
  
  unsigned long area;
  
  if (node_id == target_index) {
    area = 0xFFFFFFFFFFFFFFFF;
  } else {
    uint4 b = node[node_id];
    
    if (b.x == 0) {
      area = 0xFFFFFFFFFFFFFFFF;
    } else {
      uint4 u = node_union(&a, &b);
      area = surface_half_area(&u);
    }
  }
  
  //min_area[threadIdx.x] = area;
  min_area[get_local_id(0)] = area;
  
  barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
  
  // Too lazy to code parallel reduction right now
  if (get_local_id(0) == 0) {
    for (unsigned int i=1; i < get_local_size(0); i++)
      min_area[0] = ullmin(min_area[0], min_area[i]); // unsigned long min
  }
  
  barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
  
  if (min_area[0] == area) {
    
    if (get_group_id(1)==0 /*blockIdx.y == 0*/) {
      if (min_area[0] < adjacent_area) {
	min_index_block[block_offset + get_group_id(0)/*blockIdx.x*/] = node_id;
	min_area_block[block_offset + get_group_id(0)/*blockIdx.x*/] = area;
	flag[0] = 1;
      } else {
	min_area_block[block_offset + get_group_id(0)/*blockIdx.x*/] = 0xFFFFFFFFFFFFFFFF;
	min_index_block[block_offset + get_group_id(0)/*blockIdx.x*/] = target_index + 1;
      }
    } else {
      
      if (min_area[0] < adjacent_area)
	flag[get_group_id(1)/*blockIdx.y*/] = 1;
    }
    
  }
}



__kernel void swap_nodes(unsigned int a_index, unsigned int b_index,
			 __global uint4 *node)
{
  uint4 temp4 = node[a_index];
  node[a_index] = node[b_index];
  node[b_index] = temp4;
}

__kernel void collapse_child(unsigned int start, unsigned int end, __global uint4 *node)
			     
{
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = get_num_groups(0)*get_local_size(0);//gridDim.x * blockDim.x;
  
  for (unsigned int i=start+thread_id; i < end; i += stride) {
    uint4 this_node = node[i];
    unsigned int nchild = this_node.w >> CHILD_BITS;
    unsigned int child_id = this_node.w &  ~NCHILD_MASK;
    if (nchild == 1)
      node[i] = node[child_id];
  }
}

__kernel void area_sort_child(unsigned int start, unsigned int end,
			      __local Geometry *geometry)
{
  //unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  //unsigned int stride = gridDim.x * blockDim.x;
  unsigned int thread_id = get_local_size(0)*get_group_id(0) + get_local_id(0); // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int stride = get_num_groups(0)*get_local_size(0);//gridDim.x * blockDim.x;
  
  float distance[MAX_CHILD];
  uint4 children[MAX_CHILD];
  
  for (unsigned int i=start+thread_id; i < end; i += stride) {
    uint4 this_node = get_packed_node(geometry, i);
    unsigned int nchild = this_node.w >> CHILD_BITS;
    unsigned int child_id = this_node.w &  ~NCHILD_MASK;
    
    if (nchild <= 1)
      continue;
    
    for (unsigned int i=0; i < nchild; i++) {
      children[i] = get_packed_node(geometry, child_id+i);
      Node unpacked = get_node(geometry, child_id+i);
      float3 delta = unpacked.upper - unpacked.lower;
      distance[i] = -(delta.x * delta.y + delta.y * delta.z + delta.z * delta.x);
    }
    
    piksrt2(nchild, distance, children);
    
    for (unsigned int i=0; i < nchild; i++)
      put_packed_node(geometry, child_id + i, &children[i]);
  }
}

//} // extern "C"
