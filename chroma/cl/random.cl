//-*-c++-*-
#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <Random123/threefry.h>

// Random123 provides stateless RNGs.
// It produces a random number given a counter and key
// The counter is some type or collection of integers, e.g. 4 unsigned 32 bit integers
// The counter should be incremented each call somehow
// The key is some fixed set of unsigned integers
//  
// For our use, we must keep and store the counter we use.
// The number must be big enough

//#include "physical_constants.h"
#include "interpolate.h"

// See gpu/clrandstate.py for pyopencl definition
typedef struct {
  unsigned int a;
  unsigned int b; 
  unsigned int c; 
  unsigned int d;
} clrandState;

float clrand_uniform(__global clrandState *s, float low, float high);
float sample_cdf(__global clrandState *rng, int ncdf, __global float *cdf_x, __global float *cdf_y);
float sample_cdf_interp(__global clrandState *rng, int ncdf, float x0, float delta, float *cdf_y);


__kernel void makeRandStates( unsigned int first_state_id,
			      __global unsigned int* a,
			      __global unsigned int* b,
			      __global unsigned int* c,
			      __global unsigned int* d,
			      __global clrandState* s ) {
  unsigned int id = first_state_id + get_global_id(0);
  s[id].a = a[id];
  s[id].b = b[id];
  s[id].c = c[id];
  s[id].d = d[id];
};

__kernel void fillArray( unsigned int first_state_id,
			 __global clrandState *s,
			 __global float* out ) {
  unsigned int id = first_state_id + get_global_id(0);
  out[id] = clrand_uniform( &s[id], 0.0f, 1.0f );
};

float clrand_uniform(__global clrandState *s, float low, float high)
{
  s->a++;
  unsigned int tid  = 0;
  threefry4x32_key_t k = {{tid, 0xdecafbad, 0xfacebead, 0x12345678}};
  threefry4x32_ctr_t c = {{s->a, s->b, s->c, s->d}};
  union {
    threefry4x32_ctr_t c;
    int4 i;
  } u;
  c.v[0];
  u.c = threefry4x32(c, k); // should
  unsigned int ix = u.i.x; // converts the generated unsigned int into a float
  unsigned int max = 0xffffffff;
  float factor = 1.0f/(float)max;
  float x = float(ix)*factor; // [0,1)
  return low + x*(high-low);
};



/* __device__ float3 */
/* uniform_sphere(curandState *s) */
/* { */
/*     float theta = uniform(s, 0.0f, 2*PI); */
/*     float u = uniform(s, -1.0f, 1.0f); */
/*     float c = sqrtf(1.0f-u*u); */

/*     return make_float3(c*cosf(theta), c*sinf(theta), u); */
/* } */

 // Draw a random sample given a cumulative distribution function */
 // Assumptions: ncdf >= 2, cdf_y[0] is 0.0, and cdf_y[ncdf-1] is 1.0 */
float sample_cdf(__global clrandState *rng, int ncdf, __global float *cdf_x, __global float *cdf_y) 
{
  float u = clrand_uniform(rng,0.0f,1.0f);
  return interp(u,ncdf,cdf_y,cdf_x); 
};

// Sample from a uniformly-sampled CDF */
float sample_cdf_interp(__global clrandState *rng, int ncdf, float x0, float delta, float *cdf_y) {
  float u = clrand_uniform(rng,0.0f,1.0f);

  int lower = 0;
  int upper = ncdf - 1;

  while(lower < upper-1)
    { 
      int ihalf = (lower + upper) / 2;
      
      if (u < cdf_y[ihalf]) 
	upper = ihalf; 
      else 
	lower = ihalf;
    }
  
  float delta_cdf_y = cdf_y[upper] - cdf_y[lower];

  return x0 + delta*lower + delta*(u-cdf_y[lower])/delta_cdf_y;
}

/* extern "C" */
/* { */

/* __global__ void */
/* init_rng(int nthreads, curandState *s, unsigned long long seed, */
/* 	 unsigned long long offset) */
/* { */
/*     int id = blockIdx.x*blockDim.x + threadIdx.x; */

/*     if (id >= nthreads) */
/* 	return; */

/*     curand_init(seed, id, offset, &s[id]); */
/* } */

/* __global__ void */
/* fill_uniform(int nthreads, curandState *s, float *a, float low, float high) */
/* { */
/*     int id = blockIdx.x*blockDim.x + threadIdx.x; */

/*     if (id >= nthreads) */
/* 	return; */

/*     a[id] = clrand_uniform(&s[id], low, high); */

/* } */

/* __global__ void */
/* fill_uniform_sphere(int nthreads, curandState *s, float3 *a) */
/* { */
/*     int id = blockIdx.x*blockDim.x + threadIdx.x; */

/*     if (id >= nthreads) */
/* 	return; */

/*     a[id] = clrand_uniform_sphere(&s[id]); */
/* } */

/* __global__ void */
/* fill_sample_cdf(int offset, int nthreads, curandState *rng_states, */
/* 		int ncdf, float *cdf_x,	float *cdf_y, float *x) */
/* { */
/*     int id = blockIdx.x*blockDim.x + threadIdx.x; */

/*     if (id >= nthreads) */
/* 	return; */

/*     curandState *s = rng_states+id; */

/*     x[id+offset] = sample_cdf(s,ncdf,cdf_x,cdf_y); */
/* } */

/* } // extern "c" */

#endif
