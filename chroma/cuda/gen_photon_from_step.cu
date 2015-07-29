//-*-c-*- 

#include <curand_kernel.h>
#include <math_constants.h>
#include <helper_math.h>

extern "C"
{

__global__ void
gen_photon_from_step( int first_photon, int nphotons, int* source_step_index,
		      float3 *step_start, float3 *step_end, float* step_fsratio,
		      float fast_time_constant, float slow_time_constant, float input_wavelength,
		      curandState *rng_states,
		      // outputs
		      float3* pos, float3* dir, float3* pol, float* t, float* wavelengths, int* last_hit_triangle, uint* flags, float* weight) {
  int id = first_photon + blockIdx.x*blockDim.x + threadIdx.x;

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
  curandState rng = rng_states[blockIdx.x*blockDim.x + threadIdx.x]; // use thread's rng

  // pick the position
  int stepid = source_step_index[id];
  float3 start = step_start[stepid];
  float3 end   = step_end[stepid];
  float3 step = make_float3( end.x-start.x, end.y-start.y, end.z-start.z );
  float step_size = curand_uniform(&rng);
  float3 pos_temp = make_float3( start.x + step.x*step_size,
				 start.y + step.y*step_size,
				 start.z + step.z*step_size );
  
  // isotropic direction
  float cosz = 2.0f*curand_uniform(&rng)-1.0f;
  float sinz = sqrtf( 1.0f-cosz*cosz );
  float phi  = 2.0f*CUDART_PI_F*curand_uniform(&rng);
  float3 dir_temp = make_float3( sinz*cosf(phi), sinz*sinf(phi), cosz );
  float dirnorm = sqrtf( dir_temp.x*dir_temp.x + dir_temp.y*dir_temp.y + dir_temp.z*dir_temp.z );
  dir_temp = dir_temp/dirnorm;

  // polarization
  float phi_pol = 2.0f*CUDART_PI_F*curand_uniform(&rng);
  // now rotate into photo reference frame
  float3 fixedz = make_float3( 0.0f, 0.0f, 1.0f );
  // rotx, roty, dir_temp are the photon coordinate axes in the fixed frame
  float3 rotx = cross( fixedz, dir_temp ); 
  float3 roty = cross( dir_temp, rotx );
  float3 lineOfnodes = make_float3( rotx.x, rotx.y, 0.0f );
  float LONnorm = dot( lineOfnodes, lineOfnodes );
  lineOfnodes = make_float3( lineOfnodes.x/sqrtf( LONnorm ), lineOfnodes.y/sqrtf( LONnorm ), lineOfnodes.z/sqrtf( LONnorm ) );
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
  float3 pol_temp = make_float3( pol_rot[0], pol_rot[1], pol_rot[2] );
  float pol_norm = dot( pol_temp, pol_temp );
  pol_temp /= sqrtf( pol_norm );

  // output
  dir[id] = dir_temp;
  pol[id] = pol_temp;
  pos[id] = pos_temp;

  // ------------------------------------------
  // time

  float fsrand = curand_uniform(&rng);
  if ( fsrand<step_fsratio[stepid] ) // fast time component
    t[id] += -1.0*logf( 1.0-curand_uniform(&rng) )*fast_time_constant;
  else
    t[id] += -1.0*logf( 1.0-curand_uniform(&rng) )*slow_time_constant;

  // ------------------------------------------
  // return the new random state
  rng_states[blockIdx.x*blockDim.x + threadIdx.x] = rng;
  
}

}//end of extern C
