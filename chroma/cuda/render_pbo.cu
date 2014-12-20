//-*-c-*-

#include "linalg.h"
#include "intersect.h"
#include "mesh.h"
#include "sorting.h"
#include "geometry.h"
#include "matrixfour.h"

#include "stdio.h"

__device__ float4
get_color(const float3 &direction, const Triangle &t, unsigned int rgba)
{
    float3 v01 = t.v1 - t.v0;
    float3 v12 = t.v2 - t.v1;
    
    float3 surface_normal = normalize(cross(v01,v12));

    float cos_theta = dot(surface_normal,-direction);

    if (cos_theta < 0.0f) cos_theta = -cos_theta;

    unsigned int a0 = 0xff & (rgba >> 24);
    unsigned int r0 = 0xff & (rgba >> 16);
    unsigned int g0 = 0xff & (rgba >> 8);
    unsigned int b0 = 0xff & rgba;

    float alpha = (255 - a0)/255.0f;

    return make_float4(r0*cos_theta, g0*cos_theta, b0*cos_theta, alpha);
}


__device__ MatrixFour array2matrix(float *a) 
{
    return make_matrixfour(a[0], a[1], a[2], a[3], 
                           a[4], a[5], a[6], a[7], 
                           a[8], a[9], a[10], a[11],
                           a[12], a[13], a[14], a[15]);
}




extern "C"
{


__constant__ unsigned int  g_alpha_depth ;
__constant__ int2   g_offset ;
__constant__ int2   g_flags ;
__constant__ int2   g_size ;
__constant__ float4 g_origin ;
__constant__ float  g_pixel2world[16];



__global__ void
render_pbo_debug( 
            unsigned char*pixels, 
                Geometry *g)
{

    __shared__ Geometry sg;

    if (threadIdx.x == 0) sg = *g;

    __syncthreads();


    int x = blockIdx.x*blockDim.x + threadIdx.x + g_offset.x ;
    int y = blockIdx.y*blockDim.y + threadIdx.y + g_offset.y ; 

    if ( x >= g_size.x ) return;
    if ( y >= g_size.y ) return;

    int idx = 4 * ( y*g_size.x + x ) ;          // pixel base index


    g = &sg;

    Node root = get_node(g, 0);

    float3 origin = make_float3( g_origin.x , g_origin.y, g_origin.z );
    MatrixFour pixel2world = array2matrix(g_pixel2world);

    //float4 image_pixel = make_float4( id % g_size.x , id / g_size.x, 0.0f, 1.0f );
    float4 image_pixel = make_float4( x , y, 0.0f, 1.0f );
    float4 world_pixel = pixel2world * image_pixel ;   
    float3 world_pixel_3 = make_float3( world_pixel.x, world_pixel.y, world_pixel.z )  ; 

    float3 direction   = normalize(world_pixel_3 - origin)  ; 


    if (threadIdx.x == 0 && blockIdx.x == 0){

         //printf("[%d][%d] image_pixel  %10.2f %10.2f %10.2f %10.2f \n", blockIdx.x, threadIdx.x, image_pixel.x, image_pixel.y, image_pixel.z, image_pixel.w ); 
         //printf("[%d][%d] world_pixel  %10.2f %10.2f %10.2f %10.2f \n", blockIdx.x, threadIdx.x, world_pixel.x, world_pixel.y, world_pixel.z, world_pixel.w ); 
         //printf("[%d][%d] origin       %10.2f %10.2f %10.2f  \n", blockIdx.x, threadIdx.x, origin.x, origin.y, origin.z ); 
         //printf("[%d][%d] direction    %10.2f %10.2f %10.2f  \n", blockIdx.x, threadIdx.x, direction.x, direction.y, direction.z ); 
    }
   

    //  PBO BGRA format 
    /*
    pixels[idx]   = threadIdx.x * 10   ; // B
    pixels[idx+1] = threadIdx.y * 10  ; // G
    pixels[idx+2] = origin.z     ; // R 
    pixels[idx+3] = 0   ;          // A
    */

    pixels[idx]   = blockIdx.x * 5   ; // B
    pixels[idx+1] = blockIdx.y * 10  ; // G
    pixels[idx+2] = origin.z     ; // R 
    pixels[idx+3] = 0   ;          // A


}



__global__ void
render_pbo( 
            unsigned char*pixels, 
                Geometry *g )
{

    __shared__ Geometry sg;

    if (threadIdx.x == 0) sg = *g;

    __syncthreads();

    // absolute image pixel coordinate (x,y)  and raster order idx

    int x = blockIdx.x*blockDim.x + threadIdx.x + g_offset.x ;
    int y = blockIdx.y*blockDim.y + threadIdx.y + g_offset.y ;
 
    int idx = 4 * ( y*g_size.x + x ) ;   // output pixel base index

    if ( x >= g_size.x ) return;
    if ( y >= g_size.y ) return;


    g = &sg;

    int64_t start = clock64();

    MatrixFour pixel2world = array2matrix(g_pixel2world);
    float4 image_pixel = make_float4( x, y, 0.0f, 1.0f );
    float4 world_pixel = pixel2world * image_pixel ;   
    float3 world_pixel_3 = make_float3( world_pixel.x, world_pixel.y, world_pixel.z )  ; 

    float3 origin = make_float3( g_origin.x , g_origin.y, g_origin.z );
    float3 direction   = normalize(world_pixel_3 - origin)  ; 

    float3 neg_origin_inv_dir = -origin / direction;
    float3 inv_dir = 1.0f / direction;

    float _dx[10] ;
    unsigned int dxlen[10] ; 
    float4 _color[10] ;

    dxlen[0] = 0 ;
    unsigned int n = dxlen[0];
    unsigned int alpha_depth = g_alpha_depth ;

    float distance;


    Node root = get_node(g, 0);

    if (n < 1 && !intersect_node(neg_origin_inv_dir, inv_dir, g, root)) {
        pixels[idx] = 0;      // B
        pixels[idx+1] = 0;    // G
        pixels[idx+2] = 255;  // R    MAKE EARLY RETURN RED
        pixels[idx+3] = 0;    // A
        return;
    }

    unsigned int child_ptr_stack[STACK_SIZE];
    unsigned int nchild_ptr_stack[STACK_SIZE];
    child_ptr_stack[0] = root.child;
    nchild_ptr_stack[0] = root.nchild;

    int curr = 0;

    unsigned int node_count = 0;
    unsigned int intersect_count = 0;
    unsigned int tri_count = 0;

    float *dx = _dx ;
    float4 *color_a = _color ;

    while (curr >= 0) {
        unsigned int first_child = child_ptr_stack[curr];     // pop
        unsigned int nchild = nchild_ptr_stack[curr];
        curr--;
        for (unsigned int i=first_child; i < first_child + nchild; i++) {
      
            Node node = get_node(g, i);
            node_count++;
            if (intersect_node(neg_origin_inv_dir, inv_dir, g, node)) {

                intersect_count++;
                if (node.nchild == 0) {   /* leaf node that wraps a triangle */
                    tri_count++;
                    Triangle t = get_triangle(g, node.child);

                    if (intersect_triangle(origin, direction, t, distance)) 
                    {
                        if (n < 1) {            // first triangle intersect
                            dx[0] = distance;
                            unsigned int rgba = g->colors[node.child];
                            float4 color = get_color(direction, t, rgba);
                            color_a[0] = color;
                        } else {
                            unsigned long j = searchsorted(n, dx, distance);
                            if (j <= alpha_depth-1) {
                                insert(alpha_depth, dx, j, distance);
                                unsigned int rgba = g->colors[node.child];
                                float4 color = get_color(direction, t, rgba);
                                insert(alpha_depth, color_a, j, color);
                            }
                        }
                        if (n < alpha_depth) n++;
                    }                                            // if hit triangle

                } else {
                    curr++;                                 // push
                    child_ptr_stack[curr] = node.child;
                    nchild_ptr_stack[curr] = node.nchild;
                }                               // leaf or internal node?
            }                                   // intersect node?

            //if (curr >= STACK_SIZE) {
            //printf("warning: intersect_mesh() aborted; node > tail\n");
            //break;
            //}

        }   // loop over children, starting with first_child
    }       // while nodes on stack
    

    if (n < 1) {
        pixels[idx] = 0;
        pixels[idx+1] = 0;
        pixels[idx+2] = 0;
        pixels[idx+3] = 0;
        return;
    }

    float scale = 1.0f;
    float fr = 0.0f;
    float fg = 0.0f;
    float fb = 0.0f;

    for (int i=0; i < n; i++) {
        float alpha = color_a[i].w;
        fr += scale*color_a[i].x*alpha;
        fg += scale*color_a[i].y*alpha;
        fb += scale*color_a[i].z*alpha;
        scale *= (1.0f-alpha);
    }

    unsigned int a;
    if (n < alpha_depth)
        a = floorf(255*(1.0f-scale));
    else
        a = 255;

    unsigned int red = floorf(fr/(1.0f-scale));
    unsigned int green = floorf(fg/(1.0f-scale));
    unsigned int blue = floorf(fb/(1.0f-scale));

    int metric = 0 ;  // so it compiles when no metric chosen

    if( g_flags.x > 0){

        // pick metric with options:   --metric time --kernel-flags 19,0
        //
        //metric{time}       int64_t metric = clock64() - start ;
        //metric{node}       int     metric = node_count ; 
        //metric{intersect}  int     metric = intersect_count ; 
        //metric{tri}        int     metric = tri_count ; 
        //
 
        unsigned int shifted_metric = (int) metric >> g_flags.x ;     
        pixels[idx]   = shifted_metric ;
        pixels[idx+1] = shifted_metric ;
        pixels[idx+2] = shifted_metric ;
        pixels[idx+3] = shifted_metric ;
    } else {
        pixels[idx] = blue ;         // PBO format BGRA as that is preferred by OpenGL
        pixels[idx+1] = green ;
        pixels[idx+2] = red ;
        pixels[idx+3] = 0 ;
    }


}

} // extern "C"
