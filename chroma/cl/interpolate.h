#ifndef __INTERPOLATE_H__
#define __INTERPOLATE_H__

float interp(float x, int n, __global float *xp, __global float *fp);
float interp_uniform(float x, int n, float x0, float dx, float *fp);


float interp(float x, int n, __global float *xp, __global float *fp)
{
    int lower = 0;
    int upper = n-1;
    int ihalf;

    if (x <= xp[lower])
	return fp[lower];

    if (x >= xp[upper])
	return fp[upper];

    while (lower < upper-1)
    {
      ihalf = (lower+upper)/2;
      
      if (x < xp[ihalf])
	upper = ihalf;
      else
	lower = ihalf;
    }

    float df = fp[upper] - fp[lower];
    float dx = xp[upper] - xp[lower];
    
    return fp[lower] + df*(x-xp[lower])/dx;
}

float interp_uniform(float x, int n, float x0, float dx, float *fp)
{
    if (x <= x0)
	return x0;

    float xmax = x0 + dx*(n-1);

    if (x >= xmax)
	return xmax;

    int lower = (x - x0)/dx;
    int upper = lower + 1;

    float df = fp[upper] - fp[lower];

    return fp[lower] + df*(x-(x0+dx*lower))/dx;
}

#endif
