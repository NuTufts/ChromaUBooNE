#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

unsigned int AvoidWarningThatCausesGDBGrief(void);


typedef struct Material
{
  __global float *refractive_index;
  __global float *absorption_length;
  __global float *scattering_length;
  __global float *reemission_prob;
  __global float *reemission_cdf;
  // these are moved out to satify requirement that all members are from same address space
  unsigned int n;
  float step;
  float wavelength0;
} Material;

typedef struct WavelengthAxis {
  unsigned int n;
  float step;
  float wavelength0;
} WavelengthAxis;

enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS };

typedef struct Surface
{
  __global float *detect;
  __global float *absorb;
  __global float *reemit;
  __global float *reflect_diffuse;
  __global float *reflect_specular;
  __global float *eta;
  __global float *k;
  __global float *reemission_cdf;

  __global unsigned int *model;
  __global unsigned int *transmissive;
  __global float *thickness;

  unsigned int n;
  float step;
  float wavelength0;

} Surface;

typedef struct Triangle
{
    float3 v0, v1, v2;
} Triangle;

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };
__constant const unsigned int CHILD_BITS = 28;
__constant const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);

unsigned int AvoidWarningThatCausesGDBGrief()
{
   return NCHILD_MASK;
}


typedef struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int nchild;
} Node;

// when things go poorly, blames this struct
typedef struct Geometry
{
  __global float3 *vertices;
  __global uint3 *triangles;
  __global unsigned int *material_codes;
  __global unsigned int *colors;
  __global uint4 *primary_nodes;
  __global uint4 *extra_nodes;

  //__global Material *materials;
  __global float *refractive_index;
  __global float *absorption_length;
  __global float *scattering_length;
  __global float *reemission_prob;
  __global float *reemission_cdf;  

  //__global Surface *surfaces;
  __global float *detect;
  __global float *absorb;
  __global float *reemit;
  __global float *reflect_diffuse;
  __global float *reflect_specular;
  __global float *eta;
  __global float *k;
  __global float *surf_reemission_cdf;

  __global unsigned int *model;
  __global unsigned int *transmissive;
  __global float *thickness;  

  float3 world_origin;
  float world_scale;
  int nprimary_nodes;

  int nwavelengths;
  float step;
  float wavelength0;

} Geometry;

#endif

