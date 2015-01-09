//-*-c++-*-
#ifndef __RANDOM_CL__
#define __RANDOM_CL__

#include "random.h"

// Random123 provides stateless RNGs.
// It produces a random number given a counter and key
// The counter is some type or collection of integers, e.g. 4 unsigned 32 bit integers
// The counter should be incremented each call somehow
// The key is some fixed set of unsigned integers
//  
// For our use, we must keep and store the counter we use.
// The number must be big enough

//#include "physical_constants.h"


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


#endif
