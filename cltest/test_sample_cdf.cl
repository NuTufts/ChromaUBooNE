// -*-c++-*-
#include "random.cl"

__kernel void test_sample_cdf(__global clrandState* s,
			      int ncdf, 
			      __global float *cdf_x, __global float *cdf_y, __global float *out)
{
  int id = get_global_id(0);
  out[id] = sample_cdf(&s[id], ncdf, cdf_x, cdf_y);
};
