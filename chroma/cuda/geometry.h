#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "geometry_types.h"
#include "linalg.h"
#include "nodetexref.h"

__device__ float3 
to_float3(const uint3 &a)
{
  return make_float3(a.x, a.y, a.z);
}

__device__ uint4
get_packed_node(Geometry *geometry, const unsigned int &i)
{
  if (i < geometry->nprimary_nodes) {
    // vector texture access
    return tex1Dfetch( nodevec_tex_ref, i );

    // individual texture access
    //uint x = tex1Dfetch( node_tex_ref, 4*i+0 );
    //uint y = tex1Dfetch( node_tex_ref, 4*i+1 );
    //uint z = tex1Dfetch( node_tex_ref, 4*i+2 );
    //uint w = tex1Dfetch( node_tex_ref, 4*i+3 );
    //return make_uint4( x, y, z, w );
    
    // original: global access
    //return geometry->primary_nodes[i];
  }
  else {

    // vector texture access
    return tex1Dfetch( extra_nodevec_tex_ref, i );

    // individual texture access
    /* uint x = tex1Dfetch( extra_node_tex_ref, 4*(i-geometry->nprimary_nodes)+0 ); */
    /* uint y = tex1Dfetch( extra_node_tex_ref, 4*(i-geometry->nprimary_nodes)+1 ); */
    /* uint z = tex1Dfetch( extra_node_tex_ref, 4*(i-geometry->nprimary_nodes)+2 ); */
    /* uint w = tex1Dfetch( extra_node_tex_ref, 4*(i-geometry->nprimary_nodes)+3 ); */
    /* return make_uint4( x, y, z, w ); */

    // original: global access
    //return geometry->extra_nodes[i - geometry->nprimary_nodes];
  }
}
__device__ void
put_packed_node(Geometry *geometry, const unsigned int &i, const uint4 &node)
{
    if (i < geometry->nprimary_nodes)
	geometry->primary_nodes[i] = node;
    else
        geometry->extra_nodes[i - geometry->nprimary_nodes] = node;
}

__device__ Node
get_node(Geometry *geometry, const unsigned int &i)
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

__device__ Triangle
get_triangle(Geometry *geometry, const unsigned int &i)
{
  // access to global memory
    /* uint3 triangle_data = geometry->triangles[i]; */

    /* Triangle triangle; */
    /* triangle.v0 = geometry->vertices[triangle_data.x]; */
    /* triangle.v1 = geometry->vertices[triangle_data.y]; */
    /* triangle.v2 = geometry->vertices[triangle_data.z]; */

  // individual access to texture memory
  /* uint tri_vert[3]; */
  /* for (int t=0; t<3; t++) */
  /*   tri_vert[t] = tex1Dfetch( triangles_tex_ref, 3*i+t ); // index of 3 vertices */

  /* float verts[3][3]; */
  /* for (int t=0; t<3; t++) */
  /*   for (int v=0; v<3; v++) */
  /*     verts[t][v] = tex1Dfetch( vertices_tex_ref, 3*tri_vert[t]+v ); // coordinates */

  /* Triangle triangle; */
  /* triangle.v0 = make_float3( verts[0][0], verts[0][1], verts[0][2] ); */
  /* triangle.v1 = make_float3( verts[1][0], verts[1][1], verts[1][2] ); */
  /* triangle.v2 = make_float3( verts[2][0], verts[2][1], verts[2][2] ); */

  uint4 tri_vert = tex1Dfetch( trianglesvec_tex_ref, i );
  float4 v0 = tex1Dfetch( verticesvec_tex_ref, tri_vert.x );
  float4 v1 = tex1Dfetch( verticesvec_tex_ref, tri_vert.y );
  float4 v2 = tex1Dfetch( verticesvec_tex_ref, tri_vert.z );
  Triangle triangle;
  triangle.v0 = make_float3( v0.x, v0.y, v0.z );
  triangle.v1 = make_float3( v1.x, v1.y, v1.z );
  triangle.v2 = make_float3( v2.x, v2.y, v2.z );

  return triangle;
}

template <class T>
__device__ float
interp_property(T *m, const float &x, const float *fp)
{
    if (x < m->wavelength0)
	return fp[0];

    if (x > (m->wavelength0 + (m->n-1)*m->step))
	return fp[m->n-1];

    int jl = (x-m->wavelength0)/m->step;

    return fp[jl] + (x-(m->wavelength0 + jl*m->step))*(fp[jl+1]-fp[jl])/m->step;
}

#endif
