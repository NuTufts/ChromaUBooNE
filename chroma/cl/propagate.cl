//-*-c++-*-

#include "linalg.h"
#include "geometry.h"
#include "photon.h"

__kernel void photon_duplicate(int first_photon, int nthreads,
			       __global float3 *positions, __global float3 *directions,
			       __global float *wavelengths, 
			       __global float3 *polarizations,
			       __global float *times, __global unsigned int *histories,
			       __global int *last_hit_triangles, __global float *weights,
			       int copies, int stride)
{

  int id = get_local_size(0)*get_group_id(0) + get_local_id(0); 

  if (id >= nthreads)
    return;

  int photon_id = first_photon + id;

  Photon p;
  p.position = positions[photon_id];
  p.direction = directions[photon_id];
  p.polarization = polarizations[photon_id];
  p.wavelength = wavelengths[photon_id];
  p.time = times[photon_id];
  p.last_hit_triangle = last_hit_triangles[photon_id];
  p.history = histories[photon_id];
  p.weight = weights[photon_id];

  for (int i=1; i <= copies; i++) {
    int target_photon_id = photon_id + stride * i;

    positions[target_photon_id] = p.position;
    directions[target_photon_id] = p.direction;
    polarizations[target_photon_id] = p.polarization;
    wavelengths[target_photon_id] = p.wavelength;
    times[target_photon_id] = p.time;
    last_hit_triangles[target_photon_id] = p.last_hit_triangle;
    histories[target_photon_id] = p.history;
    weights[target_photon_id] = p.weight;
  }
}


__kernel void count_photons(int first_photon, int nthreads, unsigned int target_flag,
			    __global unsigned int *index_counter,
			    __global unsigned int *histories)
{
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0); 
  __local unsigned int counter;

  if (get_local_id(0) == 0)
    counter = 0;
  barrier( CLK_LOCAL_MEM_FENCE );

  if (id < nthreads) {
    int photon_id = first_photon + id;

    if (histories[photon_id] & target_flag) {
      atomic_add(&counter, 1);
    }
    
  }

  barrier( CLK_LOCAL_MEM_FENCE );

  if (get_local_id(0) == 0)
    atomic_add(index_counter, counter);
}

__kernel void copy_photons(int first_photon, int nthreads, unsigned int target_flag,
			   __global unsigned int *index_counter,
			   __global float3 *positions, __global float3 *directions,
			   __global float *wavelengths, __global float3 *polarizations,
			   __global float *times, __global unsigned int *histories,
			   __global int *last_hit_triangles, __global float *weights,
			   __global float3 *new_positions, __global float3 *new_directions,
			   __global float *new_wavelengths, __global float3 *new_polarizations,
			   __global float *new_times, __global unsigned int *new_histories,
			   __global int *new_last_hit_triangles, __global float *new_weights)
{
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
    
  if (id >= nthreads)
    return;
    
  int photon_id = first_photon + id;

  if (histories[photon_id] & target_flag) {
    int offset = atomic_add(index_counter, 1);

    new_positions[offset] = positions[photon_id];
    new_directions[offset] = directions[photon_id];
    new_polarizations[offset] = polarizations[photon_id];
    new_wavelengths[offset] = wavelengths[photon_id];
    new_times[offset] = times[photon_id];
    new_histories[offset] = histories[photon_id];
    new_last_hit_triangles[offset] = last_hit_triangles[photon_id];
    new_weights[offset] = weights[photon_id];
  }
}
	      
__kernel void propagate( int first_photon, int nthreads, 
			 __global unsigned int *input_queue,
			 __global unsigned int *output_queue, __global clrandState *rng_states,
			 __global float3 *positions, __global float3 *directions,
			 __global float *wavelengths, __global float3 *polarizations,
			 __global float *times, __global unsigned int *histories,
			 __global int *last_hit_triangles, __global float *weights,
			 int max_steps, int iuse_weights, int scatter_first,
			 float world_scale, __global float3* world_origin, int nprimary_nodes,
			 unsigned int n, float wavelength_step, float wavelength0,
			 //Geometry
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
			 __global unsigned int *model, __global unsigned int *transmissive, __global float *thickness )
{
  __local Geometry sg;

  if ( get_local_id(0) == 0) {
    // populate geometry item. populates the pointers I guess.
    // what's the point? have to dereference back to global memory
    fill_geostruct(  &sg, vertices, triangles, material_codes, colors, primary_nodes, extra_nodes,
		     nmaterials, refractive_index, absorption_length, scattering_length, reemission_prob, reemission_cdf,
		     nsurfaces, detect, absorb, reemit, reflect_diffuse, reflect_specular, eta, k, surf_reemission_cdf, model, transmissive, thickness,
		     *world_origin, world_scale, nprimary_nodes,
		     n, wavelength_step, wavelength0 );
  }
  
  barrier( CLK_LOCAL_MEM_FENCE );

  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  if (id >= nthreads)
    return;

  int photon_id = input_queue[first_photon + id];
  if (histories[photon_id] & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT))
    return; // short circuit finished photons

  __global clrandState* rng = rng_states + id;

  Photon p; // deep copy photon instance
  p.position          = positions[photon_id];
  p.direction         = directions[photon_id];
  p.direction         /= length(p.direction);
  p.polarization      = polarizations[photon_id];
  p.polarization      /= length(p.polarization);
  p.wavelength        = wavelengths[photon_id];
  p.time              = times[photon_id];
  p.last_hit_triangle = last_hit_triangles[photon_id];
  p.history           = histories[photon_id];
  p.weight            = weights[photon_id];
  bool use_weights = false;
  if ( iuse_weights==1 )
    use_weights = true;

  State s;
  
  int steps = 0;

  while (steps < max_steps) {
    steps++;
    int command;    
    // check for NaN and fail
    if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) {
      p.history |= NO_HIT | NAN_ABORT;
      break;
    }
    
    fill_state(&s, &p, &sg);
    //    pdump( &p, photon_id, p.history, steps, command, id );

    if (p.last_hit_triangle == -1)
      break;
    
    command = propagate_to_boundary(&p, &s, rng, use_weights, scatter_first);
    scatter_first = 0; // Only use the scatter_first value once
    //pdump( &p, photon_id, p.history, steps, command, id );
 
    if (command == BREAK)
      break;
    
    if (command == CONTINUE)
      continue;
    
    if (s.surface_index != -1) {
      command = propagate_at_surface(&p, &s, rng, &sg, use_weights);
      
      if (command == BREAK)
	break;
      
      if (command == CONTINUE)
	continue;
    }
    
    propagate_at_boundary(&p, &s, rng);

    //pdump( &p, photon_id, p.history, steps, command, id );
  } // while (steps < max_steps)

  // return the values to the host
  //rng_states[id] = rng; // no need, we've been passing the address around. Maybe a bad idea.
  positions[photon_id] = p.position;
  directions[photon_id] = p.direction;
  polarizations[photon_id] = p.polarization;
  wavelengths[photon_id] = p.wavelength;
  times[photon_id] = p.time;
  histories[photon_id] = p.history;
  last_hit_triangles[photon_id] = p.last_hit_triangle;
  weights[photon_id] = p.weight;
  
  // Not done, put photon in output queue
  if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) {
    int out_idx = atomic_add(output_queue, 1); // adds one to first address
    output_queue[out_idx] = photon_id; // gives photon_id at sequential address
  }
  

} // propagate

