//#include "cuComplex.h"
#include <pyopencl-complex.h>

/* complex math functions adding to those available in cuComplex.h */

cfloat_t clCsinf (cfloat_t x)
{
    float u = coshf(x.y) * sinf(x.x);
    float v = sinhf(x.y) * cosf(x.x);
    return make_cfloat_t(u, v);
}

cfloat_t clCcosf (cfloat_t x)
{
    float u = coshf(x.y) * cosf(x.x);
    float v = -sinhf(x.y) * sinf(x.x);
    return make_cfloat_t(u, v);
}

cfloat_t clCtanf (cfloat_t x)
{
    return clCdivf(clCsinf(x), clCcosf(x));
}

float clCargf (cfloat_t x)
{
    return atan2f(x.y, x.x);
}

cfloat_t clCsqrtf (cfloat_t x)
{
    float r = sqrtf(clCabsf(x));
    float t = clCargf(x) / 2.0f;
    return make_cfloat_t(r * cosf(t), r * sinf(t));
}

