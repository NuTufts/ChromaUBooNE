#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

// just going to pass variables around
// opencl is not very struct friendly

#include "geometry_types.h"
//#include "linalg.h"

// Declarations
float3 to_float3(const uint3 *a);
uint4 get_packed_node(__local Geometry *geometry, const unsigned int i);
void put_packed_node(__local Geometry *geometry, const unsigned int i, const uint4 *node);
Node get_node(__local Geometry *geometry, const unsigned int i);
Triangle get_triangle(__local Geometry *geometry, const unsigned int i);
float interp_material_property(Material *m, const float *x, const float *fp);
float interp_surface_property(Surface *m, const float *x, const float *fp);
void fill_geostruct( __local Geometry* g,
                     __global float3* vertices, __global uint3* triangles,
                     __global unsigned int *material_codes, __global unsigned int *colors,
                     __global uint4 *primary_nodes, __global uint4 *extra_nodes,
		     int nmaterials,
                     __global float *refractive_index, __global float *absorption_length, __global float *scattering_length,
                     __global float *reemission_prob, __global float *reemission_cdf,
		     int nsurfaces,
                     __global float *detect, __global float *absorb, __global float *reemit,
                     __global float *reflect_diffuse, __global float *reflect_specular,
                     __global float *eta, __global float *k, __global float *surf_reemission_cdf,
                     __global unsigned int *model, __global unsigned int *transmissive, __global float *thickness, 
                     float3 world_origin, float world_scale, int nprimary_nodes,
                     int nwavelengths, float step, float wavelength0 );

// Definitions
float3 to_float3(const uint3 *a)
{ 
  return make_float3((*a).x, (*a).y, (*a).z);
}

uint4 get_packed_node(__local Geometry *geometry, const unsigned int i)
{
  if (i < (unsigned int)geometry->nprimary_nodes)
    return geometry->primary_nodes[i];
  else
    return geometry->extra_nodes[i - geometry->nprimary_nodes];
}

void put_packed_node(__local Geometry *geometry, const unsigned int i, const uint4 *node)
{
  if (i < (unsigned int)geometry->nprimary_nodes)
    geometry->primary_nodes[i] = *node;
  else
    geometry->extra_nodes[i - geometry->nprimary_nodes] = *node;
}

Node get_node(__local Geometry *geometry, const unsigned int i)
{
     uint4 node = get_packed_node(geometry, i); 

     Node node_struct;

     uint3 lower_int = make_uint3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
     uint3 upper_int = make_uint3(node.x >> 16, node.y >> 16, node.z >> 16);


     node_struct.lower = geometry->world_origin + to_float3(&lower_int) * geometry->world_scale;
     node_struct.upper = geometry->world_origin + to_float3(&upper_int) * geometry->world_scale;
     node_struct.child = node.w & ~NCHILD_MASK;
     node_struct.nchild = node.w >> CHILD_BITS;

     return node_struct;
 }

 Triangle get_triangle(__local Geometry *geometry, const unsigned int i)
 {
   // struct (and its pointers to global address) lives locally
   // addresses in triangles and vertices live globally and get copied to device values
   uint3 triangle_data = geometry->triangles[i];
   
   Triangle triangle;
   triangle.v0 = geometry->vertices[triangle_data.x];
   triangle.v1 = geometry->vertices[triangle_data.y];
   triangle.v2 = geometry->vertices[triangle_data.z];

   return triangle;
}

// opencl is C99. Templates not allowed...
/* template <class T> float interp_property(T *m, const float *x, const float *fp) */
/* { */
/*     if (*x < m->wavelength0) */
/* 	return fp[0]; */

/*     if (*x > (m->wavelength0 + (m->n-1)*m->step)) */
/* 	return fp[m->n-1]; */

/*     int jl = (*x-m->wavelength0)/m->step; */

/*     return fp[jl] + (*x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step; */
/* } */

float interp_material_property(Material *m, const float *x, const float *fp)
{
    if (*x < m->wavelength0)
	return fp[0];

    if (*x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (*x-m->wavelength0)/m->step;

    return fp[jl] + (*x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

float interp_surface_property(Surface *m, const float *x, const float *fp)
{
    if (*x < m->wavelength0)
	return fp[0];

    if (*x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (*x-m->wavelength0)/m->step;

    return fp[jl] + (*x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

void fill_geostruct( __local Geometry* g,
		     __global float3* vertices, __global uint3* triangles,
		     __global unsigned int *material_codes, __global unsigned int *colors,
		     __global uint4 *primary_nodes, __global uint4 *extra_nodes,
		     // Materials
		     int nmaterials,
		     __global float *refractive_index, __global float *absorption_length, __global float *scattering_length,
		     __global float *reemission_prob, __global float *reemission_cdf,
		     // Surfaces
		     int nsurfaces,
		     __global float *detect, __global float *absorb, __global float *reemit,
		     __global float *reflect_diffuse, __global float *reflect_specular,
		     __global float *eta, __global float *k, __global float *surf_reemission_cdf,
		     __global unsigned int *model, __global unsigned int *transmissive, __global float *thickness,
		     // world info
		     float3 world_origin, float world_scale, int nprimary_nodes,
		     // wavelength array info
		     int nwavelengths, float step, float wavelength0 ) {
  g->vertices = vertices;
  g->triangles = triangles;
  g->material_codes = material_codes;
  g->colors = colors;
  g->primary_nodes = primary_nodes;
  g->extra_nodes = extra_nodes;
  // Material: unrolled arrays
  g->refractive_index = refractive_index;
  g->absorption_length = absorption_length;
  g->scattering_length = scattering_length;
  g->reemission_prob = reemission_prob;
  g->reemission_cdf = reemission_cdf;
  // Sufaces: unrolled arrays
  g->detect = detect;
  g->absorb = absorb;
  g->reemit = reemit;
  g->reflect_diffuse = reflect_diffuse;
  g->reflect_specular = reflect_specular;
  g->eta = eta;
  g->k = k;
  g->surf_reemission_cdf = surf_reemission_cdf;
  g->model = model;
  g->transmissive = transmissive;
  g->thickness = thickness;
  g->world_origin = world_origin;
  g->world_scale = world_scale;
  g->nprimary_nodes = nprimary_nodes;
  g->nwavelengths = nwavelengths;
  g->step = step;
  g->wavelength0 = wavelength0;
}

#endif
