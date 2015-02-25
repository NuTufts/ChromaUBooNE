//-*-c++-*-
#ifndef __GEOMETRY_STRUCTS_CL__
#define __GEOMETRY_STRUCTS_CL__
#include "geometry.h"


// tests of all this madness this broken-C99 madness
__kernel void make_geostruct(//Geometry 
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
			     // World Info
			     float3 world_origin, float world_scale, int nprimary_nodes,
			     // Wavelength array info
			     unsigned int nwavelengths, float wavelengthstep, float wavelength0 ) {
  // so we don't end up passing a million pointers around, we define global structs
  __local Geometry g;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  if ( get_local_id(0) == 0) {
    fill_geostruct( &g, vertices, triangles, material_codes, colors, primary_nodes, extra_nodes,
		    nmaterials, refractive_index, absorption_length, scattering_length, reemission_prob, reemission_cdf,
		    nsurfaces, detect, absorb, reemit, reflect_diffuse, reflect_specular, eta, k, surf_reemission_cdf, model, transmissive, thickness,
		    nplanes, wire_diameter, wire_pitch,
		    world_origin, world_scale, nprimary_nodes,
		    nwavelengths, wavelengthstep, wavelength0 );
    
  }
  barrier( CLK_LOCAL_MEM_FENCE );
  if ( id==0 )
    dump_geostruct_info( &g, id, nprimary_nodes );
};

#endif
