//-*-c-*- 

#include "random.h"

__kernel void
gen_photon_from_step( const int first_photon, const int nphotons, __global int* source_step_index,
		      __global float3 *step_start, __global float3 *step_end, __global float* step_fsratio,
		      const float fast_time_constant, const float slow_time_constant, const float input_wavelength,
		      __global clrandState *rng_states,
		      // outputs
		      __global float3* pos, __global float3* dir, __global float3* pol, 
		      __global float* t, __global float* wavelengths, 
		      __global int* last_hit_triangle, __global uint* flags, __global float* weight) {
  int id = first_photon + get_global_id(0);

  if (id >= nphotons)
    return;
  
  // get the easy stuff out of the way
  flags[id] = 0;
  weight[id] = 1.0f;
  last_hit_triangle[id] = -1;
  wavelengths[id] = input_wavelength;

  // ------------------------------------------
  // random position and direction

  // random state
  __global clrandState* rng = rng_states+id;

  // pick the position
  float3 start = step_start[id];
  float3 end   = step_end[id];
  //float3 step = make_float3( end.x-start.x, end.y-start.y, end.z-start.z );
  float3 step = end-start; //make_float3( end.x-start.x, end.y-start.y, end.z-start.z );
  float step_size = clrand_uniform(rng, 0.0f, 1.0f);
/*   float3 pos_temp = make_float3( start.x + step.x*step_size, */
/* 				 start.y + step.y*step_size, */
/* 				 start.z + step.z*step_size ); */
  float3 pos_temp = start + step_size*step;
  
  // isotropic direction
  float cosz = 2.0f*clrand_uniform(rng, 0.0f, 1.0f)-1.0f;
  float sinz = 1-sqrtf(cosz*cosz);
  float phi  = 2.0f*M_PI_F*clrand_uniform(rng, 0.0f, 1.0f);
  float3 dir_temp = (float3)( sinz*cosf(phi), sinz*sinf(phi), cosz );
  dir_temp = normalize( dir_temp );

  // polarization
  float phi_pol = 2.0f*M_PI_F*clrand_uniform(rng, 0.0f, 1.0f);
  // now rotate into photo reference frame
  float3 fixedz = (float3)( 0.0f, 0.0f, 1.0f );
  // rotx, roty, dir_temp are the photon coordinate axes in the fixed frame
  float3 rotx = cross( fixedz, dir_temp ); 
  float3 roty = cross( dir_temp, rotx );
  float3 lineOfnodes = (float3)( rotx.x, rotx.y, 0.0f );
  float LONnorm = dot( lineOfnodes, lineOfnodes );
  //lineOfnodes = make_float3( lineOfnodes.x/sqrtf( LONnorm ), lineOfnodes.y/sqrtf( LONnorm ), lineOfnodes.z/sqrtf( LONnorm ) );
  lineOfnodes = normalize( step_size );
  float alpha = atan2f( dir_temp.x, -dir_temp.y );
  float beta = dir_temp.z;
  float gamma = atan2f( rotx.z, roty.z );
  float c[3] = { cosf(alpha), cosf(beta), cosf(gamma) };
  float s[3] = { sinf(alpha), sinf(beta), sinf(gamma) };
  float rot[3][3] = { { c[0]*c[2]-c[1]*s[0]*s[2], -c[0]*s[2]-c[1]*c[2]*s[0], s[0]*s[1] },
		      { c[2]*s[0]+c[0]*c[1]*s[2], c[0]*c[1]*c[2]-s[0]*s[2], -c[0]*s[1] },
		      {              s[1]*s[2],       c[2]*s[1],                 c[1]  } };
  float pol_fixed[3] = { cosf( phi_pol ), sinf( phi_pol ), 0.0 };
  float pol_rot[3];
  for (int row=0; row<3; row++) {
    pol_rot[row] = 0.0f;
    for (int col=0; col<3; col++) {
      pol_rot[row] += rot[row][col]*pol_fixed[col];
    }
  }
  float3 pol_temp = (float3)( pol_rot[0], pol_rot[1], pol_rot[2] );
  float pol_norm = dot( pol_temp, pol_temp );
  //pol_temp /= sqrtf( pol_norm );
  pol_norm = normalize( pol_norm );

  // output
  dir[id] = dir_temp;
  pol[id] = pol_temp;
  pos[id] = pos_temp;

  // ------------------------------------------
  // time

  float fsrand = clrand_uniform(rng, 0.0f, 1.0f);
  if ( fsrand<step_fsratio[id] ) // fast time component
    t[id] += -1.0*logf( clrand_uniform(rng, 0.0f, 1.0f) )/fast_time_constant;
  else
    t[id] += -1.0*logf( clrand_uniform(rng, 0.0f, 1.0f) )/slow_time_constant;

  // ------------------------------------------
  
}
