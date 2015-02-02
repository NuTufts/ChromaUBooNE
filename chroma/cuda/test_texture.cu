//-*-c++-*- 

#include "nodetexref.h"
#include "stdio.h"


extern "C"
{
__global__ 
void test_texture( int nnodes, uint* nodes ) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  if ( id<nnodes ) {

    uint a = tex1Dfetch( node_tex_ref, 4*id+0 );
    uint b = tex1Dfetch( node_tex_ref, 4*id+1 );
    uint c = tex1Dfetch( node_tex_ref, 4*id+2 );
    uint d = tex1Dfetch( node_tex_ref, 4*id+3 );
    //printf("id %d: %u %u %u %u\n", id, a, b, c, d );    
    //uint4 node = make_uint4( a, b, c, d );
    uint4 node = tex1Dfetch( nodevec_tex_ref, id );

    float4 vertices = tex1Dfetch( verticesvec_tex_ref, id );

    //uint4 node = tex1Dfetch( node_tex_ref, id );
    printf("id %d: %u %u %u %u\n", id, node.x, node.y, node.z, node.w );
    printf("id %d: %.2f %.2f %.2f\n", id, vertices.x, vertices.y, vertices.z );

  }
}
}
