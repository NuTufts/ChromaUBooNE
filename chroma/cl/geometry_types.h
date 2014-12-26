#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__


unsigned int AvoidWarningThatCausesGDBGrief(void);


typedef struct Material
{
    float *refractive_index;
    float *absorption_length;
    float *scattering_length;
    float *reemission_prob;
    float *reemission_cdf;
    unsigned int n;
    float step;
    float wavelength0;
} Material;

enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS };

typedef struct Surface
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

typedef struct Geometry
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
} Geometry;

#endif

