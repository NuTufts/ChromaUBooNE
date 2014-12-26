#ifndef __LINALG_H__
#define __LINALG_H__

// OpenCL has native support for these functions I think
/* float3 operator- (const float3 &a) */
/* { */
/*     return make_float3(-a.x, -a.y, -a.z); */
/* } */

/* float3 operator* (const float3 &a, const float3 &b) */
/* { */
/*     return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); */
/* } */

/* float3 operator/ (const float3 &a, const float3 &b) */
/* { */
/*     return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); */
/* } */

/* void operator*= (float3 &a, const float3 &b) */
/* { */
/*     a.x *= b.x; */
/*     a.y *= b.y; */
/*     a.z *= b.z; */
/* } */

/* void operator/= (float3 &a, const float3 &b) */
/* { */
/*     a.x /= b.x; */
/*     a.y /= b.y; */
/*     a.z /= b.z; */
/* } */

/* float3 operator+ (const float3 &a, const float3 &b) */
/* { */
/*     return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); */
/* } */

/* void operator+= (float3 &a, const float3 &b) */
/* { */
/*     a.x += b.x; */
/*     a.y += b.y; */
/*     a.z += b.z; */
/* } */

/* float3 operator- (const float3 &a, const float3 &b) */
/* { */
/*     return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); */
/* } */

/* void operator-= (float3 &a, const float3 &b) */
/* { */
/*     a.x -= b.x; */
/*     a.y -= b.y; */
/*     a.z -= b.z; */
/* } */

/* float3 operator+ (const float3 &a, const float &c) */
/* { */
/*     return make_float3(a.x+c, a.y+c, a.z+c); */
/* } */

/* void operator+= (float3 &a, const float &c) */
/* { */
/*     a.x += c; */
/*     a.y += c; */
/*     a.z += c; */
/* } */

/* float3 operator+ (const float &c, const float3 &a) */
/* { */
/*     return make_float3(c+a.x, c+a.y, c+a.z); */
/* } */

/* float3 operator- (const float3 &a, const float &c) */
/* { */
/*     return make_float3(a.x-c, a.y-c, a.z-c); */
/* } */

/* void operator-= (float3 &a, const float &c) */
/* { */
/*     a.x -= c; */
/*     a.y -= c; */
/*     a.z -= c; */
/* } */

/* float3 operator- (const float &c, const float3& a) */
/* { */
/*     return make_float3(c-a.x, c-a.y, c-a.z); */
/* } */

/* float3 operator* (const float3 &a, const float &c) */
/* { */
/*     return make_float3(a.x*c, a.y*c, a.z*c); */
/* } */

/* void operator*= (float3 &a, const float &c) */
/* { */
/*     a.x *= c; */
/*     a.y *= c; */
/*     a.z *= c; */
/* } */

/* float3 operator* (const float &c, const float3& a) */
/* { */
/*     return make_float3(c*a.x, c*a.y, c*a.z); */
/* } */

/* float3 operator/ (const float3 &a, const float &c) */
/* { */
/*     return make_float3(a.x/c, a.y/c, a.z/c); */
/* } */

/* void operator/= (float3 &a, const float &c) */
/* { */
/*     a.x /= c; */
/*     a.y /= c; */
/*     a.z /= c; */
/* } */

/* float3 operator/ (const float &c, const float3 &a) */
/* { */
/*     return make_float3(c/a.x, c/a.y, c/a.z); */
/* } */

/* float dot(const float3 &a, const float3 &b) */
/* { */
/*     return a.x*b.x + a.y*b.y + a.z*b.z; */
/* } */

/* float3 cross(const float3 &a, const float3 &b) */
/* { */
/*     return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); */
/* } */

/* float3 abs(const float3&a) */
/* { */
/*     return make_float3(abs(a.x),abs(a.y),abs(a.z)); */
/* } */

/* float norm(const float3 &a) */
/* { */
/*     return sqrtf(dot(a,a)); */
/* } */

/* float3 normalize(const float3 &a) */
/* { */
/*     return a/norm(a); */
/* } */

#endif
