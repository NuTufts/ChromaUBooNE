#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "geometry_types.h"
#include "linalg.h"

float3 to_float3(const uint3 &a)
{
  return make_float3(a.x, a.y, a.z);
}

uint4 get_packed_node(Geometry *geometry, const unsigned int &i)
{
    if (i < geometry->nprimary_nodes)
	return geometry->primary_nodes[i];
    else
	return geometry->extra_nodes[i - geometry->nprimary_nodes];
}

void put_packed_node(Geometry *geometry, const unsigned int &i, const uint4 &node)
{
    if (i < geometry->nprimary_nodes)
	geometry->primary_nodes[i] = node;
    else
        geometry->extra_nodes[i - geometry->nprimary_nodes] = node;
}

Node get_node(Geometry *geometry, const unsigned int &i)
{
    uint4 node = get_packed_node(geometry, i); 
	
    Node node_struct;

    uint3 lower_int = make_uint3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
    uint3 upper_int = make_uint3(node.x >> 16, node.y >> 16, node.z >> 16);


    node_struct.lower = geometry->world_origin + to_float3(lower_int) * geometry->world_scale;
    node_struct.upper = geometry->world_origin + to_float3(upper_int) * geometry->world_scale;
    node_struct.child = node.w & ~NCHILD_MASK;
    node_struct.nchild = node.w >> CHILD_BITS;
    
    return node_struct;
}

Triangle get_triangle(Geometry *geometry, const unsigned int &i)
{
    uint3 triangle_data = geometry->triangles[i];

    Triangle triangle;
    triangle.v0 = geometry->vertices[triangle_data.x];
    triangle.v1 = geometry->vertices[triangle_data.y];
    triangle.v2 = geometry->vertices[triangle_data.z];

    return triangle;
}

template <class T> float interp_property(T *m, const float &x, const float *fp)
{
    if (x < m->wavelength0)
	return fp[0];

    if (x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (x-m->wavelength0)/m->step;

    return fp[jl] + (x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

#endif
