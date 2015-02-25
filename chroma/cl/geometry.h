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
float interp_material_property(Material *m, const float x, __global const float *fp);
float interp_surface_property(Surface *m, const float x, __global const float *fp);
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
		     __global float* nplanes, __global float* wire_diameter, __global float* wire_pitch,
                     float3 world_origin, float world_scale, int nprimary_nodes,
                     unsigned int nwavelengths, float step, float wavelength0 );
void fill_material_struct( unsigned int material_index, Material* m, __local Geometry* g );
void fill_surface_struct( unsigned int surface_index, Surface* s, __local Geometry* g );
void dump_geostruct_info( __local Geometry* g, int threadid );

// Definitions
float3 to_float3(const uint3 *a)
{ 
  //return make_float3((*a).x, (*a).y, (*a).z);
  return (float3) ( (*a).x, (*a).y, (*a).z );
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

     //uint3 lower_int = make_uint3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
     //uint3 upper_int = make_uint3(node.x >> 16, node.y >> 16, node.z >> 16);
     uint3 lower_int = (uint3)(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
     uint3 upper_int = (uint3)(node.x >> 16, node.y >> 16, node.z >> 16);

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

float interp_material_property(Material *m, const float x, __global const float *fp)
{
    if (x < m->wavelength0)
	return fp[0];

    if (x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (x-m->wavelength0)/m->step;

    return fp[jl] + (x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

float interp_surface_property(Surface *m, const float x, __global const float *fp)
{
    if (x < m->wavelength0)
	return fp[0];

    if (x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (x-m->wavelength0)/m->step;

    return fp[jl] + (x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
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
		     __global float* nplanes, __global float* wire_diameter, __global float* wire_pitch,
		     // world info
		     float3 world_origin, float world_scale, int nprimary_nodes,
		     // wavelength array info
		     unsigned int nwavelengths, float step, float wavelength0 ) {
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
  g->nplanes = nplanes;
  g->wire_diameter = wire_diameter;
  g->wire_pitch = wire_pitch;
  g->world_origin = world_origin;
  g->world_scale = world_scale;
  g->nprimary_nodes = nprimary_nodes;
  g->nwavelengths = nwavelengths;
  g->step = step;
  g->wavelength0 = wavelength0;
  g->nmaterials = nmaterials;
  g->nsurfaces = nsurfaces;
}

void fill_material_struct( unsigned int material_index, Material* m, __local Geometry* g ) {
  // the material arrays in Geometry are unrolled
  unsigned int offset = material_index*g->nwavelengths;
  m->refractive_index  = (g->refractive_index + offset);
  m->absorption_length = (g->absorption_length + offset);
  m->scattering_length = (g->scattering_length + offset);
  m->reemission_prob   = (g->reemission_prob + offset);
  m->reemission_cdf    = (g->reemission_cdf + offset);
  m->n = g->nwavelengths;
  m->step = g->step;
  m->wavelength0 = g->wavelength0;
  
}

void fill_surface_struct( unsigned int surface_index, Surface* s, __local Geometry* g ) {
  // the material arrays in Geometry are unrolled
  unsigned int offset = surface_index*g->nwavelengths;
  s->detect           = (g->detect + offset);
  s->absorb           = (g->absorb + offset);
  s->reemit           = (g->reemit + offset);
  s->reflect_diffuse  = (g->reflect_diffuse + offset);
  s->reflect_specular = (g->reflect_specular + offset);
  s->eta              = (g->eta + offset);
  s->k                = (g->k + offset);
  s->reemission_cdf   = (g->surf_reemission_cdf + offset);
  s->model            = (g->model + offset);
  s->transmissive     = (g->transmissive + offset);
  s->thickness        = (g->thickness + offset);
  s->nplanes          = (g->nplanes + offset);
  s->wire_diameter    = (g->wire_diameter + offset);
  s->wire_pitch       = (g->wire_pitch + offset);

  s->n = g->nwavelengths;
  s->step = g->step;
  s->wavelength0 = g->wavelength0;
}

void dump_geostruct_info( __local Geometry* g, int threadid ) {
#if __OPENCL_VERSION__>=120
  printf("========================================================================\n");
  printf("DUMPING GEOMETRY INFO IN STRUCT\n");
  printf("-- World info --\n");
  printf("thread id: %d\n",threadid);
  printf("Number of Nodes: %d\n",g->nprimary_nodes);
  printf("World origin: (%.3f, %.3f, %.3f)\n", g->world_origin.x, g->world_origin.y, g->world_origin.z);
  printf("World scale: %.3f\n",g->world_scale);
  printf("-- Wavelength arrays info --\n");
  printf("Number of wavelengths: %d\n",g->nwavelengths);
  printf("Size of wavelength steps: %.2f nm\n",g->step);
  printf("Starting wavelength: %.2f nm\n",g->wavelength0);
  printf("-- Materials --\n");
  printf("Number of materials: %d\n",g->nmaterials);

  for (int imat=0; imat<g->nmaterials; imat++) {
    Material mat;
    fill_material_struct( imat, &mat, g );

    printf(" Material %d (nwavelengths=%d)\n",imat, mat.n );
    printf("  refractive_index: ");
    for (unsigned int iw=0; iw<mat.n; iw++)
      printf(" %.2f",mat.refractive_index[iw] );
    printf("\n");

    printf("  absorption_length: ");
    for (unsigned int iw=0; iw<mat.n; iw++)
      printf(" %.2f",mat.absorption_length[iw] );
    printf("\n");

    printf("  scattering_length: ");
    for (unsigned int iw=0; iw<mat.n; iw++)
      printf(" %.2f",mat.scattering_length[iw] );
    printf("\n");

    printf("  reemission_prob: ");
    for (unsigned int iw=0; iw<mat.n; iw++)
      printf(" %.2f",mat.reemission_prob[iw] );
    printf("\n");

    printf("  reemission_cdf: ");
    for (unsigned int iw=0; iw<mat.n; iw++)
      printf(" %.2f",mat.reemission_cdf[iw] );
    printf("\n");

  }

  printf("-- Surfaces --\n");
  printf("Number of surfaces: %d\n",g->nsurfaces);

  for (int isurf=0; isurf<g->nsurfaces; isurf++) {
    Surface surf;
    fill_surface_struct( isurf, &surf, g );

    printf(" Surface %d (nwavelengths=%d)\n",isurf, surf.n );

    printf("  detect: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.detect[iw] );
    printf("\n");

    printf("  absorb: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.absorb[iw] );
    printf("\n");

    printf("  reemit: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.reemit[iw] );
    printf("\n");

    printf("  reflect_diffuse: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.reflect_diffuse[iw] );
    printf("\n");

    printf("  reflect_specular: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.reflect_specular[iw] );
    printf("\n");

    printf("  eta: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.eta[iw] );
    printf("\n");

    printf("  k: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.k[iw] );
    printf("\n");

    printf("  reemission_cdf: ");
    for (unsigned int iw=0; iw<surf.n; iw++)
      printf(" %.2f",surf.reemission_cdf[iw] );
    printf("\n");
    
  }

  printf("========================================================================\n");
#endif

};

#endif

