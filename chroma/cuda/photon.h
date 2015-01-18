#ifndef __PHOTON_H__
#define __PHOTON_H__

#include "stdio.h"
#include "linalg.h"
#include "rotate.h"
#include "random.h"
#include "physical_constants.h"
#include "mesh.h"
#include "geometry.h"
#include "cx.h"

#define WEIGHT_LOWER_THRESHOLD 0.0001f

struct Photon
{
    float3 position;
    float3 direction;
    float3 polarization;
    float wavelength;
    float time;
  
    float weight;
  
    unsigned int history;

    int last_hit_triangle;

#ifdef VBO_DEBUG
    int id ;
#endif

};

struct State
{
    bool inside_to_outside;

    float3 surface_normal;

    float refractive_index1, refractive_index2;
    float absorption_length;
    float scattering_length;
    float reemission_prob;
    Material *material1;

    int surface_index;

    float distance_to_boundary;

    // SCB debug addition
    int material1_index ; 
    int material2_index ; 

};

enum
{
    NO_HIT           = 0x1 << 0,
    BULK_ABSORB      = 0x1 << 1,
    SURFACE_DETECT   = 0x1 << 2,
    SURFACE_ABSORB   = 0x1 << 3,
    RAYLEIGH_SCATTER = 0x1 << 4,
    REFLECT_DIFFUSE  = 0x1 << 5,
    REFLECT_SPECULAR = 0x1 << 6,
    SURFACE_REEMIT   = 0x1 << 7,
    SURFACE_TRANSMIT = 0x1 << 8,
    BULK_REEMIT      = 0x1 << 9,
    NAN_ABORT        = 0x1 << 31
}; // processes





enum { BREAK, CONTINUE, PASS, START, RETURN }; // return value from propagate_to_boundary

enum {
   STATUS_NONE,
   STATUS_HISTORY_COMPLETE,
   STATUS_UNPACK, 
   STATUS_NAN_FAIL, 
   STATUS_FILL_STATE,
   STATUS_NO_INTERSECTION,
   STATUS_TO_BOUNDARY,
   STATUS_AT_SURFACE,
   STATUS_AT_SURFACE_UNEXPECTED,
   STATUS_AT_BOUNDARY,
   STATUS_BREAKOUT,
   STATUS_ENQUEUE,
   STATUS_DONE,
}; // propagate_vbo status for debug tracing 


__device__ void 
pdump( Photon& p, int photon_id, int status, int steps, int command, int slot )
{
#if __CUDA_ARCH__ >= 200
  // printf only supported for computer capability > 2.0
    printf("STATUS: ");
    switch(status)
    {
       case STATUS_NONE                   : printf("NONE             ")  ; break ;
       case STATUS_HISTORY_COMPLETE       : printf("HISTORY_COMPLETE ")  ; break ;
       case STATUS_UNPACK                 : printf("UNPACK           ")  ; break ;
       case STATUS_NAN_FAIL               : printf("NAN_FAIL         ")  ; break ;
       case STATUS_FILL_STATE             : printf("FILL_STATE       ")  ; break ;
       case STATUS_NO_INTERSECTION        : printf("NO_INTERSECTION  ")  ; break ;
       case STATUS_TO_BOUNDARY            : printf("TO_BOUNDARY      ")  ; break ;
       case STATUS_AT_SURFACE             : printf("AT_SURFACE       ")  ; break ;
       case STATUS_AT_BOUNDARY            : printf("AT_BOUNDARY      ")  ; break ;
       case STATUS_BREAKOUT               : printf("BREAKOUT         ")  ; break ;
       case STATUS_ENQUEUE                : printf("ENQUEUE          ")  ; break ;
       case STATUS_DONE                   : printf("DONE             ")  ; break ;
       case STATUS_AT_SURFACE_UNEXPECTED  : printf("AT_SURFACE_UNEXPECTED ") ; break ;
       default                            : printf("STATUS_UNKNOWN_ENUM_VALUE (%d) ",status) ; break ;
    } 

    printf("COM: ");
    switch(command)
    {
       case  CONTINUE : printf("CONTINUE ") ; break ;
       case     BREAK : printf("BREAK    ") ; break ;
       case      PASS : printf("PASS     ") ; break ;
       case     START : printf("START    ") ; break ;
       case    RETURN : printf("RETURN   ") ; break ;
       default        : printf("         ") ;
    }

    printf("[%6d] slot %2d steps %2d ", photon_id, slot, steps );
    printf("lht %6d tpos %8.3f %10.2f %10.2f %10.2f ", p.last_hit_triangle, p.time, p.position.x, p.position.y, p.position.z );
    printf("   w %7.2f   dir %8.2f %8.2f %8.2f ", p.wavelength, p.direction.x, p.direction.y, p.direction.z );
    printf("pol %8.3f %8.3f %8.3f ",   p.polarization.x, p.polarization.y, p.polarization.z );
    //printf("flg %4d ", p.history );

    // maybe arrange to show history changes ?
    if (p.history & NO_HIT )           printf("NO_HIT ");
    if (p.history & RAYLEIGH_SCATTER ) printf("RAYLEIGH_SCATTER ");
    if (p.history & REFLECT_DIFFUSE )  printf("REFLECT_DIFFUSE ");
    if (p.history & REFLECT_SPECULAR ) printf("REFLECT_SPECULAR ");
    if (p.history & SURFACE_REEMIT )   printf("SURFACE_REEMIT ");
    if (p.history & BULK_REEMIT )      printf("BULK_REEMIT ");
    if (p.history & SURFACE_TRANSMIT ) printf("SURFACE_TRANSMIT ");
    if (p.history & SURFACE_ABSORB )   printf("SURFACE_ABSORB ");
    if (p.history & SURFACE_DETECT )   printf("SURFACE_DETECT ");
    if (p.history & BULK_ABSORB )      printf("BULK_ABSORB ");

    // if (p.history & NAN_ABORT )      printf("NAN_ABORT "); 
    unsigned int one = 0x1 ;   // avoids warning: integer conversion resulted in a change of sign
    unsigned int U_NAN_ABORT = one << 31 ;
    if (p.history & U_NAN_ABORT )      printf("NAN_ABORT ");

    printf("\n");
#endif
}




__device__ int
convert(int c)
{
    if (c & 0x80)
	return (0xFFFFFF00 | c);
    else
	return c;
}

__device__ float
get_theta(const float3 &a, const float3 &b)
{
    return acosf(fmaxf(-1.0f,fminf(1.0f,dot(a,b))));
}

__device__ void
fill_state(State &s, Photon &p, Geometry *g)
{

    p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
                                         s.distance_to_boundary, 
                                         p.last_hit_triangle);

    if (p.last_hit_triangle == -1) {
        s.material1_index = 999;  
        s.material2_index = 999;  
        p.history |= NO_HIT;
        return;
    }

    Triangle t = get_triangle(g, p.last_hit_triangle);

    unsigned int material_code = g->material_codes[p.last_hit_triangle];

    int inner_material_index = convert(0xFF & (material_code >> 24));
    int outer_material_index = convert(0xFF & (material_code >> 16));
    s.surface_index = convert(0xFF & (material_code >> 8));

    float3 v01 = t.v1 - t.v0;
    float3 v12 = t.v2 - t.v1;

    s.surface_normal = normalize(cross(v01, v12));

    int material1_index, material2_index ;
    Material *material1, *material2;
    if (dot(s.surface_normal,-p.direction) > 0.0f) {
        // outside to inside
        material1 = g->materials[outer_material_index];
        material2 = g->materials[inner_material_index];
        material1_index = outer_material_index ; 
        material2_index = inner_material_index ; 

        s.inside_to_outside = false;
    }
    else {
        // inside to outside
        material1 = g->materials[inner_material_index];
        material2 = g->materials[outer_material_index];
        material1_index = inner_material_index; 
        material2_index = outer_material_index ; 
        s.surface_normal = -s.surface_normal;

        s.inside_to_outside = true;
    }

    s.refractive_index1 = interp_property(material1, p.wavelength,
                                          material1->refractive_index);
    s.refractive_index2 = interp_property(material2, p.wavelength,
                                          material2->refractive_index);
    s.absorption_length = interp_property(material1, p.wavelength,
                                          material1->absorption_length);
    s.scattering_length = interp_property(material1, p.wavelength,
                                          material1->scattering_length);
    s.reemission_prob = interp_property(material1, p.wavelength,
                                        material1->reemission_prob);

    s.material1 = material1;
    s.material1_index = material1_index;  
    s.material2_index = material2_index;  
} // fill_state

__device__ float3
pick_new_direction(float3 axis, float theta, float phi)
{
    // Taken from SNOMAN rayscatter.for
    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    float cos_phi, sin_phi;
    sincosf(phi, &sin_phi, &cos_phi);
	
    float sin_axis_theta = sqrt(1.0f - axis.z*axis.z);
    float cos_axis_phi, sin_axis_phi;
	
    if (isnan(sin_axis_theta) || sin_axis_theta < 0.00001f) {
	cos_axis_phi = 1.0f;
	sin_axis_phi = 0.0f;
    }
    else {
	cos_axis_phi = axis.x / sin_axis_theta;
	sin_axis_phi = axis.y / sin_axis_theta;
    }

    float dirx = cos_theta*axis.x +
	sin_theta*(axis.z*cos_phi*cos_axis_phi - sin_phi*sin_axis_phi);
    float diry = cos_theta*axis.y +
	sin_theta*(cos_phi*axis.z*sin_axis_phi - sin_phi*cos_axis_phi);
    float dirz = cos_theta*axis.z - sin_theta*cos_phi*sin_axis_theta;

    return make_float3(dirx, diry, dirz);
}

__device__ void
rayleigh_scatter(Photon &p, curandState &rng)
{
    float cos_theta = 2.0f*cosf((acosf(1.0f - 2.0f*curand_uniform(&rng))-2*PI)/3.0f);
    if (cos_theta > 1.0f)
	cos_theta = 1.0f;
    else if (cos_theta < -1.0f)
	cos_theta = -1.0f;

    float theta = acosf(cos_theta);
    float phi = uniform(&rng, 0.0f, 2.0f * PI);

    p.direction = pick_new_direction(p.polarization, theta, phi);

    if (1.0f - fabsf(cos_theta) < 1e-6f) {
	p.polarization = pick_new_direction(p.polarization, PI/2.0f, phi);
    }
    else {
	// linear combination of old polarization and new direction
	p.polarization = p.polarization - cos_theta * p.direction;
    }

    p.direction /= norm(p.direction);
    p.polarization /= norm(p.polarization);
} // scatter

__device__
int propagate_to_boundary(Photon &p, State &s, curandState &rng,
                          bool use_weights=false, int scatter_first=0)
{
    float absorption_distance = -s.absorption_length*logf(curand_uniform(&rng));
    float scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && s.reemission_prob == 0) // Prevent absorption
        absorption_distance = 1e30;
    else
        use_weights = false;

    if (scatter_first == 1) // Force scatter
    {
        float scatter_prob = 1.0f - expf(-s.distance_to_boundary/s.scattering_length);

        if (scatter_prob > WEIGHT_LOWER_THRESHOLD) {
            int i=0;
            const int max_i = 1000;
            while (i < max_i && scattering_distance > s.distance_to_boundary) 
            {
                scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));
                i++;
            }
            p.weight *= scatter_prob;
        }
    } 
    else if (scatter_first == -1)  // Prevent scatter
    {
        float no_scatter_prob = expf(-s.distance_to_boundary/s.scattering_length);

        if (no_scatter_prob > WEIGHT_LOWER_THRESHOLD) {
            int i=0;
            const int max_i = 1000;
            while (i < max_i && scattering_distance <= s.distance_to_boundary) 
            {
               scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));
               i++;
            }
            p.weight *= no_scatter_prob;
        }
    }

    // absorption 
    //   #. advance .time and .position to absorption point
    //   #. if BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
    //   #. if BULK_ABSORB(BREAK)  .last_hit_triangle -1  
    //
    //  huh, branch BULK_REEMIT(CONTINUE) does not set .last_hit_triangle -1 ?
    //
    if (absorption_distance <= scattering_distance) {
        if (absorption_distance <= s.distance_to_boundary) 
        {
            p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
            p.position += absorption_distance*p.direction;

            float uniform_sample_reemit = curand_uniform(&rng);
            if (uniform_sample_reemit < s.reemission_prob) 
            {
                p.wavelength = sample_cdf(&rng, s.material1->n, 
                                          s.material1->wavelength0,
                                          s.material1->step,
                                          s.material1->reemission_cdf);
                p.direction = uniform_sphere(&rng);
                p.polarization = cross(uniform_sphere(&rng), p.direction);
                p.polarization /= norm(p.polarization);
                p.history |= BULK_REEMIT;
                return CONTINUE;
            } // photon is reemitted isotropically
            else 
            {
                p.last_hit_triangle = -1;
                p.history |= BULK_ABSORB;
                return BREAK;
            } // photon is absorbed in material1
        }
    }

    //  RAYLEIGH_SCATTER(CONTINUE)  .time .position advanced to scatter point .direction .polarization twiddled 
    //
    else
    {
        if (scattering_distance <= s.distance_to_boundary) {

            // Scale weight by absorption probability along this distance
            if (use_weights) p.weight *= expf(-scattering_distance/s.absorption_length);

            p.time += scattering_distance/(SPEED_OF_LIGHT/s.refractive_index1);
            p.position += scattering_distance*p.direction;

            rayleigh_scatter(p, rng);

            p.history |= RAYLEIGH_SCATTER;

            p.last_hit_triangle = -1;

            return CONTINUE;
        } // photon is scattered in material1
    } // if scattering_distance < absorption_distance


    // Scale weight by absorption probability along this distance
    if (use_weights) p.weight *= expf(-s.distance_to_boundary/s.absorption_length);

    //  Survive to boundary(PASS)  .position .time advanced to boundary 
    //
    p.position += s.distance_to_boundary*p.direction;
    p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.refractive_index1);

    return PASS;

} // propagate_to_boundary

__noinline__ __device__ void
propagate_at_boundary(Photon &p, State &s, curandState &rng)
{
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    // Photons at normal incidence do not have a unique plane of incidence,
    // so we have to pick the plane normal to be the polarization vector
    // to get the correct logic below
    if (incident_plane_normal_length < 1e-6f)
        incident_plane_normal = p.polarization;
    else
        incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient*normal_coefficient;

    float reflection_coefficient;
    if (curand_uniform(&rng) < normal_probability) 
    {
        // photon polarization normal to plane of incidence
        reflection_coefficient = -sinf(incident_angle-refracted_angle)/sinf(incident_angle+refracted_angle);

        if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) 
        {
            p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
            p.history |= REFLECT_SPECULAR;
        }
        else 
        {
            // hmm maybe add REFRACT_? flag for this branch  
            p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
        }
        p.polarization = incident_plane_normal;
    }
    else 
    {
        // photon polarization parallel to plane of incidence
        reflection_coefficient = tanf(incident_angle-refracted_angle)/tanf(incident_angle+refracted_angle);

        if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) 
        {
            p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
            p.history |= REFLECT_SPECULAR;
        }
        else 
        {
            // hmm maybe add REFRACT_? flag for this branch  
            p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
        }

        p.polarization = cross(incident_plane_normal, p.direction);
        p.polarization /= norm(p.polarization);
    }

} // propagate_at_boundary

__device__ int
propagate_at_specular_reflector(Photon &p, State &s)
{
   
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float n_incident_plane_normal = norm(incident_plane_normal); 

#ifdef VBO_DEBUG
    if( p.id == VBO_DEBUG_PHOTON_ID )
    {
        printf("id %d incident_angle          %f \n"      , p.id, incident_angle );
        printf("id %d s.surface_normal        %f %f %f \n", p.id, s.surface_normal.x, s.surface_normal.y, s.surface_normal.z ); 
        printf("id %d p.direction             %f %f %f \n", p.id, p.direction.x, p.direction.y, p.direction.z ); 
        printf("id %d incident_plane_normal   %f %f %f \n", p.id, incident_plane_normal.x, incident_plane_normal.y, incident_plane_normal.z ); 
        printf("id %d n_incident_plane_..) %f \n", p.id, n_incident_plane_normal ); 
    }
#endif

    // at normal incidence 
    //
    //   * incident_angle = 0. 
    //   * incident_plane_normal starts (0.,0.,0.) gets "normalized" by 0. to (nan,nan,nan)
    //   * collinear surface normal and direction,  
    //   * results in n_incident_plane_normal being zero
    //     

    if( n_incident_plane_normal != 0. )
    {
        incident_plane_normal /= n_incident_plane_normal;
        p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
    }
    else  // collinear surface_normal and direction, so just avoid the NAN and just flip direction
    {
        p.direction = -p.direction ;
    }

    p.history |= REFLECT_SPECULAR;

    return CONTINUE;
} // propagate_at_specular_reflector

__device__ int
propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
{
    float ndotv;
    do {
	p.direction = uniform_sphere(&rng);
	ndotv = dot(p.direction, s.surface_normal);
	if (ndotv < 0.0f) {
	    p.direction = -p.direction;
	    ndotv = -ndotv;
	}
    } while (! (curand_uniform(&rng) < ndotv) );

    p.polarization = cross(uniform_sphere(&rng), p.direction);
    p.polarization /= norm(p.polarization);

    p.history |= REFLECT_DIFFUSE;

    return CONTINUE;
} // propagate_at_diffuse_reflector

__device__ int
propagate_complex(Photon &p, State &s, curandState &rng, Surface* surface, bool use_weights=false)
{
    float detect = interp_property(surface, p.wavelength, surface->detect);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float n2_eta = interp_property(surface, p.wavelength, surface->eta);
    float n2_k = interp_property(surface, p.wavelength, surface->k);

    // thin film optical model, adapted from RAT PMT optical model by P. Jones
    cuFloatComplex n1 = make_cuFloatComplex(s.refractive_index1, 0.0f);
    cuFloatComplex n2 = make_cuFloatComplex(n2_eta, n2_k);
    cuFloatComplex n3 = make_cuFloatComplex(s.refractive_index2, 0.0f);

    float cos_t1 = dot(p.direction, s.surface_normal);
    if (cos_t1 < 0.0f)
        cos_t1 = -cos_t1;
    float theta = acosf(cos_t1);

    cuFloatComplex cos1 = make_cuFloatComplex(cosf(theta), 0.0f);
    cuFloatComplex sin1 = make_cuFloatComplex(sinf(theta), 0.0f);

    float e = 2.0f * PI * surface->thickness / p.wavelength;
    cuFloatComplex ratio13sin = cuCmulf(cuCmulf(cuCdivf(n1, n3), cuCdivf(n1, n3)), cuCmulf(sin1, sin1));
    cuFloatComplex cos3 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio13sin));
    cuFloatComplex ratio12sin = cuCmulf(cuCmulf(cuCdivf(n1, n2), cuCdivf(n1, n2)), cuCmulf(sin1, sin1));
    cuFloatComplex cos2 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio12sin));
    float u = cuCrealf(cuCmulf(n2, cos2));
    float v = cuCimagf(cuCmulf(n2, cos2));

    // s polarization
    cuFloatComplex s_n1c1 = cuCmulf(n1, cos1);
    cuFloatComplex s_n2c2 = cuCmulf(n2, cos2);
    cuFloatComplex s_n3c3 = cuCmulf(n3, cos3);
    cuFloatComplex s_r12 = cuCdivf(cuCsubf(s_n1c1, s_n2c2), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_r23 = cuCdivf(cuCsubf(s_n2c2, s_n3c3), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_t12 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n1c1), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_t23 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n2c2), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_g = cuCdivf(s_n3c3, s_n1c1);

    float s_abs_r12 = cuCabsf(s_r12);
    float s_abs_r23 = cuCabsf(s_r23);
    float s_abs_t12 = cuCabsf(s_t12);
    float s_abs_t23 = cuCabsf(s_t23);
    float s_arg_r12 = cuCargf(s_r12);
    float s_arg_r23 = cuCargf(s_r23);
    float s_exp1 = exp(2.0f * v * e);

    float s_exp2 = 1.0f / s_exp1;
    float s_denom = s_exp1 +
                    s_abs_r12 * s_abs_r12 * s_abs_r23 * s_abs_r23 * s_exp2 +
                    2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 + s_arg_r12 + 2.0f * u * e);

    float s_r = s_abs_r12 * s_abs_r12 * s_exp1 + s_abs_r23 * s_abs_r23 * s_exp2 +
                2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 - s_arg_r12 + 2.0f * u * e);
    s_r /= s_denom;

    float s_t = cuCrealf(s_g) * s_abs_t12 * s_abs_t12 * s_abs_t23 * s_abs_t23;
    s_t /= s_denom;

    // p polarization
    cuFloatComplex p_n2c1 = cuCmulf(n2, cos1);
    cuFloatComplex p_n3c2 = cuCmulf(n3, cos2);
    cuFloatComplex p_n2c3 = cuCmulf(n2, cos3);
    cuFloatComplex p_n1c2 = cuCmulf(n1, cos2);
    cuFloatComplex p_r12 = cuCdivf(cuCsubf(p_n2c1, p_n1c2), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_r23 = cuCdivf(cuCsubf(p_n3c2, p_n2c3), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_t12 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n1), cos1), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_t23 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n2), cos2), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_g = cuCdivf(cuCmulf(n3, cos3), cuCmulf(n1, cos1));

    float p_abs_r12 = cuCabsf(p_r12);
    float p_abs_r23 = cuCabsf(p_r23);
    float p_abs_t12 = cuCabsf(p_t12);
    float p_abs_t23 = cuCabsf(p_t23);
    float p_arg_r12 = cuCargf(p_r12);
    float p_arg_r23 = cuCargf(p_r23);
    float p_exp1 = exp(2.0f * v * e);

    float p_exp2 = 1.0f / p_exp1;
    float p_denom = p_exp1 +
                    p_abs_r12 * p_abs_r12 * p_abs_r23 * p_abs_r23 * p_exp2 +
                    2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 + p_arg_r12 + 2.0f * u * e);

    float p_r = p_abs_r12 * p_abs_r12 * p_exp1 + p_abs_r23 * p_abs_r23 * p_exp2 +
                2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 - p_arg_r12 + 2.0f * u * e);
    p_r /= p_denom;

    float p_t = cuCrealf(p_g) * p_abs_t12 * p_abs_t12 * p_abs_t23 * p_abs_t23;
    p_t /= p_denom;

    // calculate s polarization fraction, identical to propagate_at_boundary
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    if (incident_plane_normal_length < 1e-6f)
        incident_plane_normal = p.polarization;
    else
        incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient * normal_coefficient; // i.e. s polarization fraction

    float transmit = normal_probability * s_t + (1.0f - normal_probability) * p_t;
    if (!surface->transmissive)
        transmit = 0.0f;

    float reflect = normal_probability * s_r + (1.0f - normal_probability) * p_r;
    float absorb = 1.0f - transmit - reflect;

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;

        detect /= survive;
        reflect /= survive;
        transmit /= survive;
    }

    if (use_weights && detect > 0.0f) {
        p.history |= SURFACE_DETECT;
        p.weight *= detect;
        return BREAK;
    }

    float uniform_sample = curand_uniform(&rng);

    if (uniform_sample < absorb) {
        // detection probability is conditional on absorption here
        float uniform_sample_detect = curand_uniform(&rng);
        if (uniform_sample_detect < detect)
            p.history |= SURFACE_DETECT;
        else
            p.history |= SURFACE_ABSORB;

        return BREAK;
    }
    else if (uniform_sample < absorb + reflect || !surface->transmissive) {
        // reflect, specularly (default) or diffusely
        float uniform_sample_reflect = curand_uniform(&rng);
        if (uniform_sample_reflect < reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else
            return propagate_at_specular_reflector(p, s);
    }
    else {
        // refract and transmit
        p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
        p.polarization = cross(incident_plane_normal, p.direction);
        p.polarization /= norm(p.polarization);
        p.history |= SURFACE_TRANSMIT;
        return CONTINUE;
    }
} // propagate_complex

__device__ int
propagate_at_wls(Photon &p, State &s, curandState &rng, Surface *surface, bool use_weights=false)
{
    float absorb = interp_property(surface, p.wavelength, surface->absorb);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float reemit = interp_property(surface, p.wavelength, surface->reemit);

    float uniform_sample = curand_uniform(&rng);

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;
        reflect_diffuse /= survive;
        reflect_specular /= survive;
    }

    if (uniform_sample < absorb) {
        float uniform_sample_reemit = curand_uniform(&rng);
        if (uniform_sample_reemit < reemit) {
            p.history |= SURFACE_REEMIT;
            p.wavelength = sample_cdf(&rng, surface->n, surface->wavelength0, surface->step, surface->reemission_cdf);
	    p.direction = uniform_sphere(&rng);
	    p.polarization = cross(uniform_sphere(&rng), p.direction);
	    p.polarization /= norm(p.polarization);
            return CONTINUE;
        } else {
	  p.history |= SURFACE_ABSORB;
	  return BREAK;
	}
    }
    else if (uniform_sample < absorb + reflect_specular + reflect_diffuse) {
        // choose how to reflect, defaulting to diffuse
        float uniform_sample_reflect = curand_uniform(&rng) * (reflect_specular + reflect_diffuse);
        if (uniform_sample_reflect < reflect_specular)
            return propagate_at_specular_reflector(p, s);
        else
            return propagate_at_diffuse_reflector(p, s, rng);
    }
    else {
        p.history |= SURFACE_TRANSMIT;
        return CONTINUE;
    }
} // propagate_at_wls

__device__ int
propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
                     bool use_weights=false)
{
    Surface *surface = geometry->surfaces[s.surface_index];

    if (surface->model == SURFACE_COMPLEX)
        return propagate_complex(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_WLS)
        return propagate_at_wls(p, s, rng, surface, use_weights);
    else 
    {
        // use default surface model: do a combination of specular and
        // diffuse reflection, detection, and absorption based on relative
        // probabilties

        // since the surface properties are interpolated linearly, we are
        // guaranteed that they still sum to 1.0.
        float detect = interp_property(surface, p.wavelength, surface->detect);
        float absorb = interp_property(surface, p.wavelength, surface->absorb);
        float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
        float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);

        float uniform_sample = curand_uniform(&rng);

        if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) 
        {
            // Prevent absorption and reweight accordingly
            float survive = 1.0f - absorb;
            absorb = 0.0f;
            p.weight *= survive;

            // Renormalize remaining probabilities
            detect /= survive;
            reflect_diffuse /= survive;
            reflect_specular /= survive;
        }

        // For default surface model
        //   SURFACE_ABSORB(BREAK)
        //   SURFACE_DETECT(BREAK)
        //   REFLECT_DIFFUSE(CONTINUE) .direction .polarization
        //   REFLECT_SPECULAR(CONTINUE) .direction
        //

        if (use_weights && detect > 0.0f) {
            p.history |= SURFACE_DETECT;
            p.weight *= detect;
            return BREAK;
        }

        if (uniform_sample < absorb) {
            p.history |= SURFACE_ABSORB;
            return BREAK;
        }
        else if (uniform_sample < absorb + detect) {
            p.history |= SURFACE_DETECT;
            return BREAK;
        }
        else if (uniform_sample < absorb + detect + reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else
            return propagate_at_specular_reflector(p, s);
    }

} // propagate_at_surface

#endif

