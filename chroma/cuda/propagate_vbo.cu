//-*-c-*-

/*
   https://devtalk.nvidia.com/default/topic/393213/nvcc-horribly-breaking-float4-read/

*/


#include "linalg.h"
#include "geometry.h"
#include "stdio.h"

#include "wav2color.h"


// --debugkernel commandline option uncomments the below debug{1} otherwise debug{0}
//
//debug{0}#define PDUMP( p, photon_id, status, steps, command, slot) 
//debug{0}#define PDUMP_ALL( p, photon_id, status, steps, command, slot) 
//debug{0}#define SDUMP( s, p, g, photon_id, steps ) 

//debug{1}#define VBO_DEBUG 1
//debug{1}#define PDUMP( p, photon_id, status, steps, command, slot)      if( photon_id == %(debugphoton)s ) pdump( p, photon_id, status, steps, command, slot)
//debug{1}#define PDUMP_ALL( p, photon_id, status, steps, command, slot)  pdump( p, photon_id, status, steps, command, slot)
//debug{1}#define SDUMP( s, p, g, photon_id, steps )   if( photon_id  == %(debugphoton)s ) sdump( s, p, g, photon_id, steps )


// --debugphoton 0  commandline options sets the below value
#define VBO_DEBUG_PHOTON_ID %(debugphoton)s
#define VBO_DEBUG_CONDITION photon_id == %(debugphoton)s
//#define VBO_DEBUG_CONDITION photon_id %% 100 == 0


#include "photon.h"

extern "C"
{

enum vbo_quad 
{ 
  vbo_position_time, 
  vbo_direction_wavelength, 
  vbo_polarization_weight, 
  vbo_ccolor, 
  vbo_flags, 
  vbo_last_hit_triangle
};

enum vbo_quad_meta 
{ 
   vbo_numquad = 6, 
   vbo_basequad = 4 
};



union quad
{
   float4 f ;
   int4   i ;
   uint4  u ;
};

__constant__ int4   g_mode ;
__constant__ int4   g_mask ;
__constant__ int4   g_mate ;
__constant__ int4   g_surf ;
__constant__ float4 g_anim ;




__device__ void pload( Photon& p, float4& color, float4* vbo,  int offset )
{
    /* Loads photon `p` and `color` from `vbo` at `offset` */

    union quad qflags, qlht ;

    float4 post = vbo[offset+vbo_position_time];     
    float4 dirw = vbo[offset+vbo_direction_wavelength]; 
    float4 polw = vbo[offset+vbo_polarization_weight];  
    float4 ccol = vbo[offset+vbo_ccolor];  

    qflags.f    = vbo[offset+vbo_flags]; 
    qlht.f      = vbo[offset+vbo_last_hit_triangle];    

    color = make_float4(ccol.x,ccol.y,ccol.z,ccol.w);

    p.position = make_float3(post.x,post.y,post.z);
    p.direction = make_float3(dirw.x,dirw.y,dirw.z);
    p.polarization = make_float3(polw.x,polw.y,polw.z);
    p.wavelength = dirw.w ;
    p.time = post.w ;

    p.last_hit_triangle = qlht.i.x;
    p.history = qflags.u.x ;
    p.weight = polw.w ;

    p.direction /= norm(p.direction);
    p.polarization /= norm(p.polarization);
}


__device__ float4 history_color( int history  )
{
    float4 color ; 
    if      (history & (NO_HIT|NAN_ABORT))    color = make_float4( 0.5, 0.5, 0.5, 1.0 ) ; // grey
    else if (history & (BULK_ABSORB))         color = make_float4( 1.0, 0.0, 0.0, 1.0 ) ; // bright red
    else if (history & (SURFACE_DETECT))      color = make_float4( 1.0, 0.0, 0.0, 1.0 ) ; // bright red
    else if (history & (SURFACE_ABSORB))      color = make_float4( 1.0, 0.0, 0.0, 1.0 ) ; // bright red
    else if (history & (RAYLEIGH_SCATTER ))   color = make_float4( 0.0, 0.0, 1.0, 1.0 ) ; // blue, like the sky 
    else if (history & (BULK_REEMIT))         color = make_float4( 0.0, 1.0, 0.0, 1.0 ) ; // light green
    else if (history & (SURFACE_REEMIT))      color = make_float4( 0.0, 0.5, 0.0, 1.0 ) ; // dark green
    else if (history & (SURFACE_TRANSMIT))    color = make_float4( 0.0, 0.7, 0.0, 1.0 ) ; // mid green
    else if (history & (REFLECT_SPECULAR))    color = make_float4( 1.0, 0.0, 1.0, 1.0 ) ; // magenta
    else if (history & (REFLECT_DIFFUSE))     color = make_float4( 0.7, 0.0, 1.0, 1.0 ) ; // purple 
    else if (history == 0)                    color = make_float4( 1.0, 1.0, 1.0, 1.0 ) ; // white
    else                                      color = make_float4( 1.0, 0.5, 0.0, 1.0 ) ; // orange 

    return color ;
}

__device__ void psave( Photon& p, float4* vbo, int photon_offset, int slot, quad& qlht, quad& qflags, bool meta_only )
{


    // Save photon info into VBO array which sets aside max_slots per photon:
    //
    //      slot 
    //           0            loaded from file/MQ with Geant4 starting info
    //           1
    //           ...
    //       max_slots-2      reserved for last photon state without any truncation
    //       max_slots-1      reserved as interpolation scratchpad
    //
    //  When there is no truncation (ie desired states to save < max_slots-2) 
    //  the last photon state is repeated in the body and at -2. 
    //  NOT SO, THE LAST JUST GOES INTO -2.
    //
    //  When there is truncation the "interpolation" between -3 and -2 will be
    //  "drunken" but it still works with unchanged presentation/interpolation code.
    //
    //  The advantage is that the final photon positions are available at a fixed location
    //  allowing photon picking to be implemented more easily.
    //
    //  http://www.rapidtables.com/web/color/RGB_Color.htm
    //  color based on most currently cogent aspects of history, 
    //  will need to change ordering as interested in different aspects
    //
    // NB the color is assigned on saving, ie whilst propagation history is evolving 
    //
    //

    int offset = photon_offset + slot*%(numquad)s ;

    vbo[offset+vbo_last_hit_triangle] = make_float4( qlht.f.x, qlht.f.y, qlht.f.z, qlht.f.w);    
    vbo[offset+vbo_flags ]            = make_float4( qflags.f.x, qflags.f.y, qflags.f.z, qflags.f.w);    
    vbo[offset+vbo_ccolor]            = history_color( p.history );

    if( meta_only ) return ; // NB slot 0 gets metadata only

    vbo[offset+vbo_position_time]        = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );     
    vbo[offset+vbo_direction_wavelength] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.wavelength ); 
    vbo[offset+vbo_polarization_weight]  = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.weight );  
}




__device__ float4 position_time_interpolate( float4& a , float4& b, float time )
{
    // interpolate positions with fraction based on times of the endpoints and the input time
    float tfrac = ( time - a.w ) / ( b.w - a.w ) ;      
    return make_float4(
          (b.x - a.x)*tfrac + a.x ,
          (b.y - a.y)*tfrac + a.y ,
          (b.z - a.z)*tfrac + a.z ,
          time  ) ;

    // TODO: adapt to use builtin lerp

}


__global__ void present_vbo( int nthreads, int max_slots, float4 *vbo )
{
    // invoked per photon, so need to spin over slots  
    // ccol.w (alpha) zero signals invisibility, 
    // it is detected in shader which sets position.w=0, 
    // to scoot the vertex off to infinity and beyond
    //
    int photon_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (photon_id >= nthreads) return;

    // slot offsets for 0,-2,-1 : in units of quad/float4

    unsigned int photon_offset = photon_id*max_slots*%(numquad)s  ; 
    unsigned int last_offset = photon_offset + (max_slots-2)*%(numquad)s ; 
    unsigned int max_offset = photon_offset  + (max_slots-1)*%(numquad)s ; 

    // constant memory "uniforms" parameter inputs

    float t            = g_anim.x ;  // daephotonskernelfunc.py
    float cohort_start = g_anim.y ; 
    float cohort_end   = g_anim.z ; 
    float cohort_mode  = g_anim.w ; 

    int mask = g_mask.x ;   // daephotonsparam.py
    int bits = g_mask.y ; 
    int pid  = g_mask.z ; 
    int sid  = g_mask.w ; 

    int mode = g_mode.x ; 

    int g_material1 = g_mate.x ; 
    int g_material2 = g_mate.y ; 
    int g_surface   = g_surf.x ; 


    // UNPACK FROM SLOT -2 : "fixed last" which was filled at tail of propagate_vbo 


    union quad tail_qlht ;
    tail_qlht.f      = vbo[last_offset + vbo_last_hit_triangle];   

    int tail_photon_id         = tail_qlht.u.x ; 
    //int tail_spare           = tail_qlht.i.y ; 
    unsigned int tail_flags    = tail_qlht.u.z ; 
    //int tail_channel_id      = tail_qlht.i.w ; 


    union quad tail_qflags ;
    tail_qflags.f    = vbo[last_offset + vbo_flags];   

    int            tail_slots = tail_qflags.i.x ; 
    float          tail_birth = tail_qflags.f.y ; // float time ranges are stuffed in the flags
    float          tail_death = tail_qflags.f.z ; 
    //unsigned int tail_steps = tail_qflags.u.w ; 


#ifdef VBO_DEBUG
 //   if ( photon_id == pid )
 //   {
 //         printf("photon_id %%d tail_slots %%d g_material1 %%d g_material2 %%d max_slots %(max_slots)s \n", photon_id, tail_slots, g_material1, g_material2 ); 
 //   }
#endif 


    // reasons to make invisible, based on selection parameters

    float mask_alpha  = ((sid  > -1 ) && ( photon_id != sid )) ||
                        ((mask > -1 ) && ( tail_flags & mask ) == 0 ) ||
                        ((bits > -1 ) && ( tail_flags != bits )) ||
                        ((cohort_start > -1 && cohort_end > -1) && ( tail_birth < cohort_start || tail_birth > cohort_end )) ? 0. : 1. ; 

    if( photon_id == pid ) mask_alpha = -mask_alpha ;    // negate to highlight the photon 

#ifdef VBO_DEBUG
    if( photon_id == sid ) printf("sid %%d mask_alpha %%f  \n ", sid, mask_alpha );
#endif 



    if(cohort_mode > 0. && mask_alpha > 0.){
        printf("I: photon_id %%d tail_birth %%f tail_death %%f  cohort %%f %%f %%f \n", photon_id, tail_birth, tail_death, cohort_start, cohort_end, cohort_mode );
    } 

   //
   // Modify "prev" slot visibility based on mode
   //
   //    * `--mode -1` (default) non-slot based mask_alpha applies
   //      ie final history/flags selecton 
   //
   //    * `--mode 0` "cumulative confetti" effect as vary animation time
   //      get time dependant slot visibiity.  
   //      Prior to time reaching the step keep invisible
   //      after time reaches the step the selection mask applies
   //
   //    * `--mode 1/2/3/...`  base visibility on the value of mode matching
   //      the slow qflags.w value (eg for a from material index)
   //
   // (fr,to) start at (0,1) so always have a pair 
   // Remember "to" will becomes "fr" except on last pair, so 
   // confusing to change "to"
   //   
   // position time interpolation of slot -1 for animation effect 
   // using "fr" direction and polarization makes more sense  as those only change at the interaction
   // also "to" is in the future, so better not to use it yet ?
   //

    float4 straddle_post ;
    int straddle_offset = -1 ;
    int fr_offset = -1 ; 
    int to_offset = -1 ; 
    int slot = 1 ;    

    while (slot <= tail_slots )
    {
        fr_offset = photon_offset + (slot-1)*%(numquad)s ;    
        to_offset = photon_offset +     slot*%(numquad)s ;    

        float4 to_post  = vbo[to_offset + vbo_position_time];
        float4 fr_post  = vbo[fr_offset + vbo_position_time];

        if ( t > fr_post.w && t < to_post.w ) 
        { 
            straddle_offset = fr_offset ;
            straddle_post = position_time_interpolate( fr_post, to_post, t );
        }

        union quad fr_qlht, fr_qflags ; 
        fr_qflags.f     = vbo[fr_offset + vbo_flags];           
        fr_qlht.f       = vbo[fr_offset + vbo_last_hit_triangle];           
        float4& fr_ccol = vbo[fr_offset + vbo_ccolor];

#ifdef VBO_DEBUG
       //if( photon_id == pid ) printf("pid %%d g_surface %%d fr_qlht.i.w  %%d \n ", pid, g_surface, fr_qlht.i.w );
#endif 

        fr_ccol.w = ((g_material1 > -1)  && ( fr_qlht.i.y != g_material1)) ||   // y:material1_index
                    ((g_material2 > -1)  && ( fr_qlht.i.z != g_material2)) ||   // z:material2_index
                    ((g_surface > -1)    && ( fr_qlht.i.w != g_surface  )) ||   // w:surface_index
                    ((mode > 0)  && ( fr_qflags.u.w != mode)) ||
                    ((mode == 0) && ( t < fr_post.w ))  ? 0. : mask_alpha ; 

        slot++ ;
    }


    if( to_offset > -1 && fr_offset > -1 ) // set visibility of last slot ("to" of the  last pair)
    {
        vbo[to_offset + vbo_ccolor].w = vbo[fr_offset + vbo_ccolor].w ;  
    }


    //  populate animation slot -1, with interpolated or last photon step

    if( straddle_offset > -1  )
    {
        memcpy( vbo + max_offset, vbo + straddle_offset, sizeof(float4)*vbo_basequad ); 
        vbo[max_offset+vbo_position_time] = straddle_post ;
    }
    else
    {
        memcpy( vbo + max_offset, vbo + last_offset, sizeof(float4)*vbo_numquad );

        union quad last_qflags, last_qlht ; 
        last_qflags.f    = vbo[last_offset + vbo_flags];           
        last_qlht.f      = vbo[last_offset + vbo_last_hit_triangle];           
        float4 last_post = vbo[last_offset + vbo_position_time];

        vbo[max_offset+vbo_ccolor].w = ((mode > 0)         && ( last_qflags.u.w != mode))      ||
                                       ((g_material1 > -1) && ( last_qlht.i.x != g_material1)) ||
                                       ((g_material2 > -1) && ( last_qlht.i.y != g_material2)) || 
                                       ((g_surface > -1)   && ( last_qlht.i.w != g_surface  )) ||   // w:surface_index
                                       ( t < last_post.w ) ?  0. : mask_alpha ; 

        // invisible prior to death OR if mode selection is active and does not match 
        // based on untruncated last photon time (slot -2) and last step material
    }
}




__device__ void sdump( State& s, Photon& p, Geometry* g, int photon_id, int steps )
{

   /*
        chroma/gpu/geometry.py

        145         material_codes = (((geometry.material1_index & 0xff) << 24) |
        146                           ((geometry.material2_index & 0xff) << 16) |
        147                           ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
        148         self.material_codes = ga.to_gpu(material_codes)

 
    convert_   = lambda c:0xFFFFFF00 | c if c & 0x80 else c ## sign bit diddling to negate the packed byte 
    codes_     = lambda c:(convert_(0xFF & (c >> 24)),convert_(0xFF & (c >> 16)),convert_(0xFF & (c >> 8)))

    In [25]: codes_(169547520)
    Out[25]: (10, 27, 23)

    In [26]: codes_(134872832)
    Out[26]: (8, 9, 4294967295)

    In [29]: np.int32(4294967295)
    Out[29]: -1

    In [36]: 0x80
    Out[36]: 128

                      wavelength = sample_cdf(&rng, 
                                          s.material1->n, 
                                          s.material1->wavelength0,
                                          s.material1->step,
                                          s.material1->reemission_cdf);

    */
 
    if(p.last_hit_triangle == -1) return ;
    unsigned int material_code = g->material_codes[p.last_hit_triangle];
    int inner_material_index = convert(0xFF & (material_code >> 24)); // grab bytes 3,2,1 (byte 0 unused?)  
    int outer_material_index = convert(0xFF & (material_code >> 16));
    int surface_index        = convert(0xFF & (material_code >> 8));

    int ncdf = s.material1->n ; 

    printf("[%%3d] %%6d material_code %%d inner %%d outer %%d si %%d ri1 %%f ri2 %%f abs %%f sca %%f rem %%f ncdf %%d w0 %%f st %%f cdf lo/up %%f %%f \n", 
           steps, 
           photon_id, 
           material_code,
           inner_material_index,
           outer_material_index,
           s.surface_index,
           s.refractive_index1,
           s.refractive_index2, 
           s.absorption_length, 
           s.scattering_length, 
           s.reemission_prob,
           ncdf,
           s.material1->wavelength0,
           s.material1->step, 
           s.material1->reemission_cdf[0], 
           s.material1->reemission_cdf[ncdf-1]); 


} 



/* 

VBO slot structure for max_slots = 10 example
==============================================

::

      0       ploaded, only meta data saved (TODO: eliminate the save?)
      1       body 
      2       body
      3       body
      4       body  
      5       body
      6       body psave only up to slot < max_slots - 3
      7 -3    post-BREAK/after-while psave into -3 or lesser slot
      8 -2    fixed last (will duplicate -3 if truncated, will duplicate a body slot -4/-5/-6/... if not truncated )
      9 -1    kept for animation 

Boundary conditions:

#. Chroma defaults to max_steps of 100, wasteful to record all those
   so truncation from steps to recorded slots is inevitable

#. need fixed positions for easy access to

   * first step (0)
   * last step (-2) : so must duplicate the last step into -2

#. need contiguous slots for OpenGL glMultiDrawArrays 
   whether truncated OR not.
   Cannot write into -3 in the body 
   as that is needed for a contiguous last 

It is tempting to avoid the duplication between the body slot 
and fixed last -2. BUT then it is not possible to have contiguous 
slots for drawing in the case of not being truncated.


To debug structure changes use::

   daephotonsanalyzer.sh --load 1 

 
*/



__device__ void fill_meta( State& s, Photon& p, quad& qlht, quad& qflags )
{

    qlht.i.x = p.last_hit_triangle ;  
    qlht.i.y = s.material1_index ;
    qlht.i.z = s.material2_index ;
    qlht.i.w = s.surface_index ;

    qflags.u.x = p.history ;
    qflags.f.y = s.distance_to_boundary ;   
    qflags.f.z = 0. ;         
    qflags.f.w = 0. ;   
}




	      
__global__ void
propagate_vbo( int first_photon, 
               int nthreads, 
               unsigned int *input_queue,
	           unsigned int *output_queue, 
               curandState *rng_states,
               float4 *vbo,
	           int max_steps, 
	           int max_slots, 
               int use_weights, 
               int scatter_first,
	           Geometry *g,
               int* solid_map, 
               int* solid_id_to_channel_id )
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nthreads) return;

    g = &sg;

    curandState rng = rng_states[id];

    int photon_id = input_queue[first_photon + id];
    int photon_offset = photon_id*max_slots*%(numquad)s ;

    Photon p;
    float4 color ; 
    pload(p, color, vbo, photon_offset ); 

#ifdef VBO_DEBUG
    p.id = photon_id ;
#endif
    

    union quad qlht, qflags ;

    float t0 = p.time ; 
    int status = STATUS_UNPACK ;
    int command = START ;
    int steps = 0;
    int slot = 0 ;

    if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)){
        status = STATUS_HISTORY_COMPLETE ; 
        command = RETURN ; 
        PDUMP_ALL( p, photon_id, status, steps, command, slot);
        return;
    }


    State s;


    while (steps < max_steps) {

       steps++;

       if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) // check for NaN and fail
       {
           p.history |= NO_HIT | NAN_ABORT;
           status = STATUS_NAN_FAIL ;
           command = BREAK ;
           break;
       }

       // *fill_state* 
       //
       // sets p.last_hit_triangle set from intersection with mesh 
       // finds triangle material, works out which of two materials currently in from normal  
       // sets s.surface_index and wavelength dependant material property lookups:
       // s.absorption_length, s.scattering_length, s.reemission_probability
       // (no photon changes other than p.last_hit_triangle, just sets state properties)
       // 
       fill_state(s, p, g); 
       status = STATUS_FILL_STATE ;  

       SDUMP( s, p, g, photon_id, steps );

       // snapshot capturing changes from below that CONTINUE around to here 
       // contrary to first impressions this does not mix different steps as
       // following the CONTINUE need to fill_state to see where the photon 
       // is headed in last_hit_triangle

       if( slot < max_slots-3 )
       {   
           // see VBO structure docs above for explanation of max_slots - 3
           bool meta_only = slot == 0 ; 
           fill_meta( s, p, qlht, qflags );
           psave( p, vbo, photon_offset, slot, qlht, qflags, meta_only );
           slot++ ;  // 0-based slot incremented after save
       }


       PDUMP( p, photon_id, status, steps, command, slot); 

       if (p.last_hit_triangle == -1){
            status = STATUS_NO_INTERSECTION ; 
            command = BREAK ;
            break;
       }

       //
       // **propagate_to_boundary** 
       //  
       // #. absorption: advance .time .position to absorption point, then either:
       //
       //    * BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
       //    * BULK_ABSORB(BREAK)    .last_hit_triangle -1
       //
       // #. scattering: advance .time .position to scatter point 
       //
       //    * RAYLEIGH_SCATTER(CONTINUE) change .direction .polarization
       //
       // #. survive: advance .time .position to boundary  (not absorbed/reemitted/scattered)
       // 
       //    * PASS
       //


       command = propagate_to_boundary(p, s, rng, use_weights, scatter_first);
       status = STATUS_TO_BOUNDARY ;  

       PDUMP( p, photon_id, status, steps, command, -1); 

       scatter_first = 0; // Only use the scatter_first value once
       if (command == BREAK) break;
       if (command == CONTINUE) continue;

       // PASS goes on to either at_surface/at_boundary handling 
       // depending on surface association 
       // NB .position .time are not changed by these,
       // at most  .direction .polarization .history are changed

       if (s.surface_index != -1) 
       {
           // **propagate_at_surface**  for default surface model
           //
           // #. SURFACE_ABSORB(BREAK)
           // #. SURFACE_DETECT(BREAK)
           // #. REFLECT_DIFFUSE(CONTINUE) .direction .polarization
           // #. REFLECT_SPECULAR(CONTINUE) .direction
           // #. NO other option, so never PASS? 
           //
           command = propagate_at_surface(p, s, rng, g, use_weights);
           status = STATUS_AT_SURFACE ; 

           PDUMP( p, photon_id, status, steps, command, -1); 

           if (command == BREAK) break ;
           if (command == CONTINUE) continue;

           status = STATUS_AT_SURFACE_UNEXPECTED ; 
           PDUMP_ALL( p, photon_id, status, steps, command, -1);  
       }

       // **propagate_at_boundary** 
       // depending on materials refractive indices and incidence angle
       //
       // #. "CONTINUE" REFLECT_SPECULAR  .direction .polarization
       // #. "CONTINUE" "REFRACT"         .direction .polarization
       // #. NO other option

       propagate_at_boundary(p, s, rng);
       status = STATUS_AT_BOUNDARY ; 

       PDUMP( p, photon_id, status, steps, CONTINUE, -1); 


    } // while (steps < max_steps)

    rng_states[id] = rng;

    //PDUMP( p, photon_id, status, steps, command, -1);

    if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) 
    {
        // Not done, put photon in output queue
        int out_idx = atomicAdd(output_queue, 1);
        output_queue[out_idx] = photon_id;
        status = STATUS_ENQUEUE ; 
    } 
    else
    {
        status = STATUS_DONE ; 
    } 


    fill_meta( s, p, qlht, qflags );
    psave( p, vbo, photon_offset, slot, qlht, qflags, false ); // last photon step info into body slot up to -3


    // hit PMT?
    int solid_id = 0 ;
    int channel_id = 0 ; 

    if ((p.history & SURFACE_DETECT) != 0) 
    {
	    if (p.last_hit_triangle > -1) {
            solid_id = solid_map[p.last_hit_triangle]; 
	        channel_id = solid_id_to_channel_id[solid_id];
        }
    }

    // duplicate the final photon info into fixed position -2, 
    // BUT use meta quads for photon rather than step level qtys 
    //
    // NB layout here needs to match that used in 
    //
    //     * chroma/event.py Photons.as_npl
    //     * daephotonsanalyzer.py      
    //
    // NB only the qlht makes it back to last quad of 4x4 NPL, 
    //    this is used to compare against the non-vbo propagation so must
    //    only include qtys relevant to both
    //

    qlht.i.x   = photon_id ;   //              
    qlht.i.y   = 0 ;           //  was slot, propagation presentation of propagation stuck at initial position if set to zero for non-vbo match
    qlht.u.z   = p.history ;   // 
    qlht.i.w   = channel_id ;  // ESSENTIAL: for hit selection

    qflags.i.x = slot ;        // used in propagation presentation, so gl.MultiDrawArrays knows which ranges of VBO structure to present 
    qflags.f.y = t0 ;         
    qflags.f.z = p.time ;   
    qflags.u.w = steps ; 

    psave( p, vbo, photon_offset, max_slots - 2, qlht, qflags, false ); // last photon into fixed slot -2


} // propagate




} // extern "C"
