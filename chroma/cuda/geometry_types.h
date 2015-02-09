#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

struct Material
{
    float *refractive_index;
    float *absorption_length;
    float *scattering_length;
    float *reemission_prob;
    float *reemission_cdf;
    unsigned int n;
    float step;
    float wavelength0;
};

enum { SURFACE_DEFAULT=0, SURFACE_COMPLEX, SURFACE_WLS, SURFACE_WIREPLANE }; // models

struct Surface
{
    float *detect;
    float *absorb;
    float *reemit;
    float *reflect_diffuse;
    float *reflect_specular;
    float *eta;
    float *k;
    float *reemission_cdf;

    unsigned int model;
    unsigned int n;
    unsigned int transmissive;
    float step;
    float wavelength0;
    float thickness;
    float nplanes;
    float wire_diameter;
    float wire_pitch;
};

struct Triangle
{
    float3 v0, v1, v2;
};

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };
const unsigned int CHILD_BITS = 28;
const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);

unsigned int AvoidWarningThatCausesGDBGrief()
{
   return NCHILD_MASK ;
}


struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int nchild;
};

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    uint4 *primary_nodes;
    uint4 *extra_nodes;
    Material **materials;
    Surface **surfaces;
    float3 world_origin;
    float world_scale;
    int nprimary_nodes;
};

struct NodeDSAR
{
  unsigned int* node_index;
  unsigned int* parent_index;
  unsigned int* aunt_index;
  unsigned int* sibling_index;
  unsigned int* first_daughter_index;
  unsigned int* ndaughters;
};

#endif

