#include <pyopencl-complex.h>
/* complex math functions adding to those available in pyopencl-complex.h */

float clCargf (cfloat_t x);


float clCargf (cfloat_t x)
{
    return atan2(x.y, x.x);
}

