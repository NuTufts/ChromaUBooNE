#ifndef __PHOTON_H__
#define __PHOTON_H__

#include "linalg.h"
#include "geometry.h"
#include "rotate.h"
#include "random.cl"
#include "physical_constants.h"
#include "mesh.h"
//#include "cx.h" // complex functions

#define WEIGHT_LOWER_THRESHOLD 0.0001f

typedef struct Photon
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

} Photon;

typedef struct State
{
  bool inside_to_outside;

  float3 surface_normal;
  
  float refractive_index1, refractive_index2;
  float absorption_length;
  float scattering_length;
  float reemission_prob;
  //Material *material1; // we try to use the material struct here
  __global float *mat1_refractive_index;  // addresses to what memory space?
  __global float *mat1_absorption_length;
  __global float *mat1_scattering_length;
  __global float *mat1_reemission_prob;
  __global float *mat1_reemission_cdf;
  unsigned int n;
  float step;
  float wavelength0;
  
  int surface_index;

  float distance_to_boundary;

  // SCB debug addition
  int material1_index ; 
  int material2_index ; 
  
} State;

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

// -----------------------------------------------------------------
// declarations
void pdump( Photon* p, int photon_id, int status, int steps, int command, int slot );
int convert(int c);
float get_theta(const float3 *a, const float3 *b);
void fill_state(State* s, Photon* p, __local Geometry *g);
float3 pick_new_direction(float3 axis, float theta, float phi);
void rayleigh_scatter(Photon *p, __global clrandState *rng);
int propagate_to_boundary(Photon* p, State* s, __global clrandState* rng, bool use_weights, int scatter_first);
void propagate_at_boundary(Photon *p, State *s, __global clrandState *rng);
int propagate_at_specular_reflector(Photon *p, State *s);
int propagate_at_diffuse_reflector(Photon *p, State *s, __global clrandState *rng);
int propagate_complex(Photon *p, State *s, __global clrandState *rng, Surface* surface, bool use_weights);
int propagate_at_wls(Photon *p, State *s, __global clrandState *rng, Surface *surface, bool use_weights);
int propagate_at_surface(Photon *p, State *s, __global clrandState *rng, __local Geometry *geometry, bool use_weights);

// -----------------------------------------------------------------
// definitions

void pdump( Photon* p, int photon_id, int status, int steps, int command, int slot )
{
  // printf only supported for computer capability > 2.0
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
       case STATUS_AT_SURFACE_UNEXPECTED  : printf("AT_SURFACE_UNEXPECTED") ; break ;
       default                            : printf("STATUS_UNKNOWN_ENUM_VALUE") ; break ;
    } 

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
    printf("lht %6d tpos %8.3f %10.2f %10.2f %10.2f ", p->last_hit_triangle, p->time, p->position.x, p->position.y, p->position.z );
    printf("   w %7.2f   dir %8.2f %8.2f %8.2f ", p->wavelength, p->direction.x, p->direction.y, p->direction.z );
    printf("pol %8.3f %8.3f %8.3f ",   p->polarization.x, p->polarization.y, p->polarization.z );
    //printf("flg %4d ", p->history );

    // maybe arrange to show history changes ?
    if (p->history & NO_HIT )           printf("NO_HIT ");
    if (p->history & RAYLEIGH_SCATTER ) printf("RAYLEIGH_SCATTER ");
    if (p->history & REFLECT_DIFFUSE )  printf("REFLECT_DIFFUSE ");
    if (p->history & REFLECT_SPECULAR ) printf("REFLECT_SPECULAR ");
    if (p->history & SURFACE_REEMIT )   printf("SURFACE_REEMIT ");
    if (p->history & BULK_REEMIT )      printf("BULK_REEMIT ");
    if (p->history & SURFACE_TRANSMIT ) printf("SURFACE_TRANSMIT ");
    if (p->history & SURFACE_ABSORB )   printf("SURFACE_ABSORB ");
    if (p->history & SURFACE_DETECT )   printf("SURFACE_DETECT ");
    if (p->history & BULK_ABSORB )      printf("BULK_ABSORB ");

    // if (p->history & NAN_ABORT )      printf("NAN_ABORT "); 
    unsigned int one = 0x1 ;   // avoids warning: integer conversion resulted in a change of sign
    unsigned int U_NAN_ABORT = one << 31 ;
    if (p->history & U_NAN_ABORT )      printf("NAN_ABORT ");

    printf("\n");
}

int convert(int c)
{
    if (c & 0x80)
	return (0xFFFFFF00 | c);
    else
	return c;
}

float get_theta(const float3 *a, const float3 *b)
{
    return acosf(fmaxf(-1.0f,fminf(1.0f,dot(*a,*b))));
}

void fill_state(State* s, Photon* p, __local Geometry *g)
{
  p->last_hit_triangle = intersect_mesh(&(p->position), &(p->direction), g,
					&(s->distance_to_boundary),
					p->last_hit_triangle);
  
  if (p->last_hit_triangle == -1) {
    s->material1_index = 999;
    s->material2_index = 999;
    p->history |= NO_HIT;
    return;
  }

  Triangle t = get_triangle(g, p->last_hit_triangle);
  
  unsigned int material_code = g->material_codes[p->last_hit_triangle];
  
  int inner_material_index = convert(0xFF & (material_code >> 24));
  int outer_material_index = convert(0xFF & (material_code >> 16));
  s->surface_index = convert(0xFF & (material_code >> 8));
  
  float3 v01 = t.v1 - t.v0;
  float3 v12 = t.v2 - t.v1;
  
  s->surface_normal = normalize(cross(v01, v12));
  
  int material1_index, material2_index ;
  Material material1, material2;
  if (dot(s->surface_normal,-p->direction) > 0.0f) {
    // outside to inside
    //material1 = g->materials[outer_material_index];
    //material2 = g->materials[inner_material_index];
    material1_index = outer_material_index ;
    material2_index = inner_material_index ;
    s->inside_to_outside = false;
  }
  else {
    // inside to outside
    //material1 = g->materials[inner_material_index];
    //material2 = g->materials[outer_material_index];
    material1_index = inner_material_index;
    material2_index = outer_material_index ;
    s->surface_normal = -s->surface_normal;
    s->inside_to_outside = true;
  }
  fill_material_struct( material1_index, &material1, g);
  fill_material_struct( material2_index, &material2, g);

  s->refractive_index1 = interp_material_property(&material1, p->wavelength, material1.refractive_index);
  s->refractive_index2 = interp_material_property(&material2, p->wavelength, material2.refractive_index);
  s->absorption_length = interp_material_property(&material1, p->wavelength, material1.absorption_length);
  s->scattering_length = interp_material_property(&material1, p->wavelength, material1.scattering_length);
  s->reemission_prob   = interp_material_property(&material1, p->wavelength, material1.reemission_prob);

  // we do this instead of mallocing instant of Material struct
  s->mat1_refractive_index = g->refractive_index + g->nwavelengths*s->material1_index;
  s->mat1_absorption_length = g->absorption_length + g->nwavelengths*s->material1_index;
  s->mat1_scattering_length = g->scattering_length + g->nwavelengths*s->material1_index;
  s->mat1_reemission_prob = g->reemission_prob + g->nwavelengths*s->material1_index;
  s->mat1_reemission_cdf = g->reemission_cdf + g->nwavelengths*s->material1_index;

  s->n = g->nwavelengths;
  s->step = g->step;
  s->wavelength0 = g->wavelength0;
					 
} // fill_state

float3 pick_new_direction(float3 axis, float theta, float phi)
{
    // Taken from SNOMAN rayscatter.for
    float cos_theta, sin_theta;
    //sincosf(theta, &sin_theta, &cos_theta);
    sin_theta = sincos( theta, &cos_theta );
    float cos_phi, sin_phi;
    //sincosf(phi, &sin_phi, &cos_phi);
    sin_phi = sincos( phi, &cos_phi );
	
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

void rayleigh_scatter(Photon *p, __global clrandState *rng)
{
  float cos_theta = 2.0f*cos((acos(1.0f - 2.0f*clrand_uniform(rng, 0.0f, 1.0f))-2.0f*M_PI_F)/3.0f);
  if (cos_theta > 1.0f)
    cos_theta = 1.0f;
  else if (cos_theta < -1.0f)
    cos_theta = -1.0f;

  float theta = acos(cos_theta);
  float phi = clrand_uniform(rng, 0.0f, 2.0f * M_PI_F);

  p->direction = pick_new_direction(p->polarization, theta, phi);

  if (1.0f - fabs(cos_theta) < 1e-6f) {
    p->polarization = pick_new_direction(p->polarization, M_PI_F/2.0f, phi);
  }
  else {
    // linear combination of old polarization and new direction
    p->polarization = p->polarization - cos_theta * p->direction;
  }

  p->direction /= length(p->direction);
  p->polarization /= length(p->polarization);
} // scatter

int propagate_to_boundary(Photon* p, State* s, __global clrandState* rng, bool use_weights, int scatter_first)
{
  float absorption_distance = -s->absorption_length*logf(clrand_uniform(rng,0.0f,1.0f));
  float scattering_distance = -s->scattering_length*logf(clrand_uniform(rng,0.0f,1.0f));

  if (use_weights && p->weight > WEIGHT_LOWER_THRESHOLD && s->reemission_prob == 0) // Prevent absorption
    absorption_distance = 1.0e30f;
  else
    use_weights = false;

  if (scatter_first == 1) // Force scatter
    {
      float scatter_prob = 1.0f - expf(-s->distance_to_boundary/s->scattering_length);

      if (scatter_prob > WEIGHT_LOWER_THRESHOLD) {
	int i=0;
	const int max_i = 1000;
	while (i < max_i && scattering_distance > s->distance_to_boundary) 
	  {
	    scattering_distance = -s->scattering_length*logf( clrand_uniform(rng, 0.0f, 1.0f) );
	    i++;
	  }
	p->weight *= scatter_prob;
      }
    } 
  else if (scatter_first == -1)  // Prevent scatter
    {
      float no_scatter_prob = expf(-s->distance_to_boundary/s->scattering_length);

      if (no_scatter_prob > WEIGHT_LOWER_THRESHOLD) {
	int i=0;
	const int max_i = 1000;
	while (i < max_i && scattering_distance <= s->distance_to_boundary) 
	  {
	    scattering_distance = -s->scattering_length*logf( clrand_uniform(rng, 0.0f, 1.0f) );
	    i++;
	  }
	p->weight *= no_scatter_prob;
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
    if (absorption_distance <= s->distance_to_boundary) 
      {
	p->time += absorption_distance/(SPEED_OF_LIGHT/s->refractive_index1);
	p->position += absorption_distance*p->direction;

	float uniform_sample_reemit = clrand_uniform(rng,0.0f,1.0f);
	if (uniform_sample_reemit < s->reemission_prob) 
	  {
	    p->wavelength = sample_cdf_interp(rng, s->n, 
					      s->wavelength0,
					      s->step,
					      s->mat1_reemission_cdf);
	    p->direction = uniform_sphere(rng);
	    p->polarization = cross(uniform_sphere(rng), p->direction);
	    p->polarization /= length(p->polarization);
	    p->history |= BULK_REEMIT;
	    return CONTINUE;
	  } // photon is reemitted isotropically
	else 
	  {
	    p->last_hit_triangle = -1;
	    p->history |= BULK_ABSORB;
	    return BREAK;
	  } // photon is absorbed in material1
      }
  }
  
    //  RAYLEIGH_SCATTER(CONTINUE)  .time .position advanced to scatter point .direction .polarization twiddled 
    //
    else
    {
        if (scattering_distance <= s->distance_to_boundary) {

            // Scale weight by absorption probability along this distance
            if (use_weights) p->weight *= expf(-scattering_distance/s->absorption_length);

            p->time += scattering_distance/(SPEED_OF_LIGHT/s->refractive_index1);
            p->position += scattering_distance*p->direction;

            rayleigh_scatter(p, rng);

            p->history |= RAYLEIGH_SCATTER;

            p->last_hit_triangle = -1;

            return CONTINUE;
        } // photon is scattered in material1
    } // if scattering_distance < absorption_distance


    // Scale weight by absorption probability along this distance
    if (use_weights) p->weight *= expf(-s->distance_to_boundary/s->absorption_length);

    //  Survive to boundary(PASS)  .position .time advanced to boundary 
    //
    p->position += s->distance_to_boundary*p->direction;
    p->time += s->distance_to_boundary/(SPEED_OF_LIGHT/s->refractive_index1);

    return PASS;

} // propagate_to_boundary

void propagate_at_boundary(Photon *p, State *s, __global clrandState *rng)
{
  float3 inv_dir = -p->direction;
  float incident_angle = get_theta( &(s->surface_normal),&inv_dir );
  float refracted_angle = asinf(sinf(incident_angle)*s->refractive_index1/s->refractive_index2);

  float3 incident_plane_normal = cross(p->direction, s->surface_normal);
  float incident_plane_normal_length = length(incident_plane_normal);
  
  // Photons at normal incidence do not have a unique plane of incidence,
  // so we have to pick the plane normal to be the polarization vector
  // to get the correct logic below
  if (incident_plane_normal_length < 1e-6f)
    incident_plane_normal = p->polarization;
  else
    incident_plane_normal /= incident_plane_normal_length;
  
  float normal_coefficient = dot(p->polarization, incident_plane_normal);
  float normal_probability = normal_coefficient*normal_coefficient;
  
  float reflection_coefficient;
  if (clrand_uniform(rng, 0.0f, 1.0f) < normal_probability) 
    {
      // photon polarization normal to plane of incidence
      reflection_coefficient = -sinf(incident_angle-refracted_angle)/sinf(incident_angle+refracted_angle);
      
      if ((clrand_uniform(rng, 0.0f, 1.0f) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) 
        {
	  p->direction = rotate_with_vec( &(s->surface_normal), incident_angle, &incident_plane_normal);
	  p->history |= REFLECT_SPECULAR;
        }
      else 
        {
	  // hmm maybe add REFRACT_? flag for this branch  
	  p->direction = rotate_with_vec(&(s->surface_normal), M_PI_F-refracted_angle, &incident_plane_normal);
        }
      p->polarization = incident_plane_normal;
    }
  else 
    {
      // photon polarization parallel to plane of incidence
      reflection_coefficient = tanf(incident_angle-refracted_angle)/tanf(incident_angle+refracted_angle);
      
      if ((clrand_uniform(rng, 0.0f, 1.0f) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) 
        {
	  p->direction = rotate_with_vec(&(s->surface_normal), incident_angle, &incident_plane_normal);
	  p->history |= REFLECT_SPECULAR;
        }
      else 
        {
	  // hmm maybe add REFRACT_? flag for this branch  
	  p->direction = rotate_with_vec(&(s->surface_normal), M_PI_F-refracted_angle, &incident_plane_normal);
        }
      
      p->polarization = cross(incident_plane_normal, p->direction);
      p->polarization /= length(p->polarization);
    }
  
} // propagate_at_boundary

int propagate_at_specular_reflector(Photon *p, State *s)
{
  
  float3 inv_dir = -p->direction;
  float incident_angle = get_theta( &(s->surface_normal), &inv_dir );
  float3 incident_plane_normal = cross(p->direction, s->surface_normal);
  float n_incident_plane_normal = length(incident_plane_normal); 

#ifdef VBO_DEBUG
    if( p->id == VBO_DEBUG_PHOTON_ID )
    {
        printf("id %d incident_angle          %f \n"      , p->id, incident_angle );
        printf("id %d s.surface_normal        %f %f %f \n", p->id, s->surface_normal.x, s->surface_normal.y, s->surface_normal.z ); 
        printf("id %d p.direction             %f %f %f \n", p->id, p->direction.x, p->direction.y, p->direction.z ); 
        printf("id %d incident_plane_normal   %f %f %f \n", p->id, incident_plane_normal.x, incident_plane_normal.y, incident_plane_normal.z ); 
        printf("id %d n_incident_plane_..) %f \n", p->id, n_incident_plane_normal ); 
    }
#endif

    // at normal incidence 
    //
    //   * incident_angle = 0. 
    //   * incident_plane_normal starts (0.,0.,0.) gets "normalized" by 0. to (nan,nan,nan)
    //   * collinear surface normal and direction,  
    //   * results in n_incident_plane_normal being zero
    //     

    if( n_incident_plane_normal != 0.0f )
    {
        incident_plane_normal /= n_incident_plane_normal;
        p->direction = rotate_with_vec( &(s->surface_normal), incident_angle, &incident_plane_normal );
    }
    else  // collinear surface_normal and direction, so just avoid the NAN and just flip direction
    {
        p->direction = -p->direction ;
    }

    p->history |= REFLECT_SPECULAR;

    return CONTINUE;
} // propagate_at_specular_reflector

int propagate_at_diffuse_reflector(Photon *p, State *s, __global clrandState *rng)
{
    float ndotv;
    do {
	p->direction = uniform_sphere(rng);
	ndotv = dot(p->direction, s->surface_normal);
	if (ndotv < 0.0f) {
	    p->direction = -p->direction;
	    ndotv = -ndotv;
	}
    } while (! (clrand_uniform(rng, 0.0f, 1.0f) < ndotv) );

    p->polarization = cross(uniform_sphere(rng), p->direction);
    p->polarization /= length(p->polarization);

    p->history |= REFLECT_DIFFUSE;

    return CONTINUE;
} // propagate_at_diffuse_reflector

int propagate_complex(Photon *p, State *s, __global clrandState *rng, Surface* surface, bool use_weights)
{
  float detect           = interp_surface_property(surface, p->wavelength, surface->detect);
  //float reflect_specular = interp_surface_property(surface, p->wavelength, surface->reflect_specular);
  float reflect_diffuse  = interp_surface_property(surface, p->wavelength, surface->reflect_diffuse);
  float n2_eta           = interp_surface_property(surface, p->wavelength, surface->eta);
  float n2_k             = interp_surface_property(surface, p->wavelength, surface->k);
  
  // thin film optical model, adapted from RAT PMT optical model by P. Jones
  //cfloat_t n1 = make_cuFloatComplex(s.refractive_index1, 0.0f);
  //cfloat_t n2 = make_cuFloatComplex(n2_eta, n2_k);
  //cfloat_t n3 = make_cuFloatComplex(s.refractive_index2, 0.0f);

/*   cfloat_t n1  = (cfloat_t)( s->refractive_index1, 0.0f ); */
/*   cfloat_t n2  = (cfloat_t)( n2_eta, n2_k ); */
/*   cfloat_t n3  = (cfloat_t)( s->refractive_index2, 0.0f ); */
/*   cfloat_t one = (cfloat_t)( 1.0f, 0.0f ); */
/*   cfloat_t two = (cfloat_t)( 2.0f, 0.0f ); */
  
/*   float cos_t1 = dot(p->direction, s->surface_normal); */
/*   if (cos_t1 < 0.0f) */
/*     cos_t1 = -cos_t1; */
/*   float theta = acos(cos_t1); */
  
/*   cfloat_t cos1 = ( cos(theta), 0.0f ); // make_cfloat_t(cosf(theta), 0.0f); */
/*   cfloat_t sin1 = ( sin(theta), 0.0f ); // make_cfloat_t(sinf(theta), 0.0f); */
  
/*   float e = 2.0f * M_PI_F * (*(surface->thickness)) / p->wavelength; */
/* /\*   cfloat_t ratio13sin = cuCmulf(cuCmulf(cuCdivf(n1, n3), cuCdivf(n1, n3)), cuCmulf(sin1, sin1)); *\/ */
/* /\*   cfloat_t cos3 = cuCsqrtf(cuCsubf(make_cfloat_t(1.0f,0.0f), ratio13sin)); *\/ */
/* /\*   cfloat_t ratio12sin = cuCmulf(cuCmulf(cuCdivf(n1, n2), cuCdivf(n1, n2)), cuCmulf(sin1, sin1)); *\/ */
/* /\*   cfloat_t cos2 = cuCsqrtf(cuCsubf(make_cfloat_t(1.0f,0.0f), ratio12sin)); *\/ */
/* /\*   float u = cuCrealf(cuCmulf(n2, cos2)); *\/ */
/* /\*   float v = cuCimagf(cuCmulf(n2, cos2)); *\/ */
/*   cfloat_t ratio13sin = cfloat_mul(cfloat_mul(cfloat_divide(n1, n3), cfloat_divide(n1, n3)), cfloat_mul(sin1, sin1)); */
/*   cfloat_t cos3       = cfloat_sqrt(one-ratio13sin); */
/*   cfloat_t ratio12sin = cfloat_mul(cfloat_mul(cfloat_divide(n1, n2), cfloat_divide(n1, n2)), cfloat_mul(sin1, sin1)); */
/*   cfloat_t cos2       = cfloat_sqrt(one-ratio12sin); */
/*   float u             = cfloat_real(cfloat_mul(n2, cos2)); */
/*   float v             = cfloat_real(cfloat_mul(n2, cos2)); */
  
/*   // s polarization */
/*   cfloat_t s_n1c1 = cfloat_mul(n1, cos1); */
/*   cfloat_t s_n2c2 = cfloat_mul(n2, cos2); */
/*   cfloat_t s_n3c3 = cfloat_mul(n3, cos3); */
/*   cfloat_t s_r12  = cfloat_divide( (s_n1c1-s_n2c2), (s_n1c1+s_n2c2) ); */
/*   cfloat_t s_r23  = cfloat_divide( (s_n2c2-s_n3c3), (s_n2c2+s_n3c3) ); */
/*   cfloat_t s_t12  = cfloat_divide( cfloat_mul(two, s_n1c1), (s_n1c1+s_n2c2) ); */
/*   cfloat_t s_t23  = cfloat_divide( cfloat_mul(two, s_n2c2), (s_n2c2+s_n3c3) ); */
/*   cfloat_t s_g    = cfloat_divide( s_n3c3, s_n1c1 ); */
  
/*   float s_abs_r12 = cfloat_abs(s_r12); */
/*   float s_abs_r23 = cfloat_abs(s_r23); */
/*   float s_abs_t12 = cfloat_abs(s_t12); */
/*   float s_abs_t23 = cfloat_abs(s_t23); */
/*   float s_arg_r12 = clCargf(s_r12); */
/*   float s_arg_r23 = clCargf(s_r23); */
/*   float s_exp1    = exp(2.0f * v * e); */
  
/*   float s_exp2 = 1.0f / s_exp1; */
/*   float s_denom = s_exp1 + */
/*     s_abs_r12 * s_abs_r12 * s_abs_r23 * s_abs_r23 * s_exp2 + */
/*     2.0f * s_abs_r12 * s_abs_r23 * cos(s_arg_r23 + s_arg_r12 + 2.0f * u * e); */
  
/*   float s_r = s_abs_r12 * s_abs_r12 * s_exp1 + s_abs_r23 * s_abs_r23 * s_exp2 + */
/*     2.0f * s_abs_r12 * s_abs_r23 * cos(s_arg_r23 - s_arg_r12 + 2.0f * u * e); */
/*   s_r /= s_denom; */
  
/*   float s_t = cfloat_real(s_g) * s_abs_t12 * s_abs_t12 * s_abs_t23 * s_abs_t23; */
/*   s_t /= s_denom; */
  
/*   // p polarization */
/*   cfloat_t p_n2c1 = cfloat_mul(n2, cos1); */
/*   cfloat_t p_n3c2 = cfloat_mul(n3, cos2); */
/*   cfloat_t p_n2c3 = cfloat_mul(n2, cos3); */
/*   cfloat_t p_n1c2 = cfloat_mul(n1, cos2); */
/*   cfloat_t p_r12  = cfloat_divide((p_n2c1-p_n1c2), (p_n2c1+p_n1c2)); */
/*   cfloat_t p_r23  = cfloat_divide((p_n3c2-p_n2c3), (p_n3c2+p_n2c3)); */
/*   cfloat_t p_t12  = cfloat_divide(cfloat_mul(cfloat_mul(two, n1), cos1), (p_n2c1+p_n1c2)); */
/*   cfloat_t p_t23  = cfloat_divide(cfloat_mul(cfloat_mul(two, n2), cos2), (p_n3c2+p_n2c3)); */
/*   cfloat_t p_g    = cfloat_divide(cfloat_mul(n3, cos3), cfloat_mul(n1, cos1)); */
  
/*   float p_abs_r12 = cfloat_abs(p_r12); */
/*   float p_abs_r23 = cfloat_abs(p_r23); */
/*   float p_abs_t12 = cfloat_abs(p_t12); */
/*   float p_abs_t23 = cfloat_abs(p_t23); */
/*   float p_arg_r12 = clCargf(p_r12); */
/*   float p_arg_r23 = clCargf(p_r23); */
/*   float p_exp1    = exp(2.0f * v * e); */
  
/*   float p_exp2 = 1.0f / p_exp1; */
/*   float p_denom = p_exp1 + */
/*     p_abs_r12 * p_abs_r12 * p_abs_r23 * p_abs_r23 * p_exp2 + */
/*     2.0f * p_abs_r12 * p_abs_r23 * cos(p_arg_r23 + p_arg_r12 + 2.0f * u * e); */
  
/*   float p_r = p_abs_r12 * p_abs_r12 * p_exp1 + p_abs_r23 * p_abs_r23 * p_exp2 + */
/*     2.0f * p_abs_r12 * p_abs_r23 * cos(p_arg_r23 - p_arg_r12 + 2.0f * u * e); */
/*   p_r /= p_denom; */
  
/*   float p_t = cfloat_real(p_g) * p_abs_t12 * p_abs_t12 * p_abs_t23 * p_abs_t23; */
/*   p_t /= p_denom; */
  
/*   // calculate s polarization fraction, identical to propagate_at_boundary */
/*   float3 inv_dir = -p->direction; */
/*   float incident_angle  = get_theta( &(s->surface_normal), &inv_dir); */
/*   float refracted_angle = asin(sin(incident_angle)*s->refractive_index1/s->refractive_index2); */
  
/*   float3 incident_plane_normal       = cross(p->direction, s->surface_normal); */
/*   float incident_plane_normal_length = length(incident_plane_normal); */
  
/*   if (incident_plane_normal_length < 1e-6f) */
/*     incident_plane_normal = p->polarization; */
/*   else */
/*     incident_plane_normal /= incident_plane_normal_length; */
  
/*   float normal_coefficient = dot(p->polarization, incident_plane_normal); */
/*   float normal_probability = normal_coefficient * normal_coefficient; // i.e. s polarization fraction */
  
/*   float transmit = normal_probability * s_t + (1.0f - normal_probability) * p_t; */
/*   if (!surface->transmissive) */
/*     transmit = 0.0f; */
  
/*   float reflect = normal_probability * s_r + (1.0f - normal_probability) * p_r; */
/*   float absorb = 1.0f - transmit - reflect; */
  
/*   if (use_weights && p->weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) { */
/*     // Prevent absorption and reweight accordingly */
/*     float survive = 1.0f - absorb; */
/*     absorb = 0.0f; */
/*     p->weight *= survive; */
    
/*     detect /= survive; */
/*     reflect /= survive; */
/*     transmit /= survive; */
/*   } */
  
/*   if (use_weights && detect > 0.0f) { */
/*     p->history |= SURFACE_DETECT; */
/*     p->weight *= detect; */
/*     return BREAK; */
/*   } */
  
/*   float uniform_sample = clrand_uniform(rng, 0.0f, 1.0f); */
  
/*   if (uniform_sample < absorb) { */
/*     // detection probability is conditional on absorption here */
/*     float uniform_sample_detect = clrand_uniform(rng, 0.0f, 1.0f); */
/*     if (uniform_sample_detect < detect) */
/*       p->history |= SURFACE_DETECT; */
/*     else */
/*       p->history |= SURFACE_ABSORB; */
    
/*     return BREAK; */
/*   } */
/*   else if (uniform_sample < absorb + reflect || !surface->transmissive) { */
/*     // reflect, specularly (default) or diffusely */
/*     float uniform_sample_reflect = clrand_uniform(rng, 0.0f, 1.0f); */
/*     if (uniform_sample_reflect < reflect_diffuse) */
/*       return propagate_at_diffuse_reflector(p, s, rng); */
/*     else */
/*       return propagate_at_specular_reflector(p, s); */
/*   } */
/*   else { */
/*     // refract and transmit */
/*     p->direction = rotate_with_vec( &(s->surface_normal), M_PI_F-refracted_angle, &incident_plane_normal); */
/*     p->polarization = cross(incident_plane_normal, p->direction); */
/*     p->polarization /= length(p->polarization); */
/*     p->history |= SURFACE_TRANSMIT; */
/*     return CONTINUE; */
/*   } */


} // propagate_complex

int propagate_at_wls(Photon *p, State *s, __global clrandState *rng, Surface *surface, bool use_weights)
{
  float absorb = interp_surface_property(surface, p->wavelength, surface->absorb);
  float reflect_specular = interp_surface_property(surface, p->wavelength, surface->reflect_specular);
  float reflect_diffuse = interp_surface_property(surface, p->wavelength, surface->reflect_diffuse);
  float reemit = interp_surface_property(surface, p->wavelength, surface->reemit);

  float uniform_sample = clrand_uniform(rng, 0.0f, 1.0f);
  
  if (use_weights && p->weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
    // Prevent absorption and reweight accordingly
    float survive = 1.0f - absorb;
    absorb = 0.0f;
    p->weight *= survive;
    reflect_diffuse /= survive;
    reflect_specular /= survive;
  }

  if (uniform_sample < absorb) {
    float uniform_sample_reemit = clrand_uniform(rng, 0.0f, 1.0f);
    if (uniform_sample_reemit < reemit) {
      p->history |= SURFACE_REEMIT;
      p->wavelength = sample_cdf_interp( rng, surface->n, surface->wavelength0, surface->step, surface->reemission_cdf);
      p->direction = uniform_sphere(rng);
      p->polarization = cross(uniform_sphere(rng), p->direction);
      p->polarization /= length(p->polarization);
      return CONTINUE;
    } else {
      p->history |= SURFACE_ABSORB;
      return BREAK;
    }
  }
  else if (uniform_sample < absorb + reflect_specular + reflect_diffuse) {
    // choose how to reflect, defaulting to diffuse
    float uniform_sample_reflect = clrand_uniform(rng,0.0f,1.0f) * (reflect_specular + reflect_diffuse);
    if (uniform_sample_reflect < reflect_specular)
      return propagate_at_specular_reflector(p, s);
    else
      return propagate_at_diffuse_reflector(p, s, rng);
  }
  else {
    p->history |= SURFACE_TRANSMIT;
    return CONTINUE;
  }
} // propagate_at_wls

int propagate_at_surface(Photon *p, State *s, __global clrandState *rng, __local Geometry *geometry, bool use_weights)
{
  //Surface* surface = geometry->surfaces[s.surface_index];
  Surface surface;
  fill_surface_struct( s->surface_index, &surface, geometry );
  
  if (*(surface.model) == SURFACE_COMPLEX)
    return propagate_complex(p, s, rng, &surface, use_weights);
  else if ( *(surface.model) == SURFACE_WLS)
    return propagate_at_wls(p, s, rng, &surface, use_weights);
  else 
    {
      // use default surface model: do a combination of specular and
      // diffuse reflection, detection, and absorption based on relative
      // probabilties
      
      // since the surface properties are interpolated linearly, we are
      // guaranteed that they still sum to 1.0.
      float detect           = interp_surface_property( &surface, p->wavelength, surface.detect);
      float absorb           = interp_surface_property( &surface, p->wavelength, surface.absorb);
      float reflect_diffuse  = interp_surface_property( &surface, p->wavelength, surface.reflect_diffuse);
      float reflect_specular = interp_surface_property( &surface, p->wavelength, surface.reflect_specular);
      
      float uniform_sample = clrand_uniform(rng, 0.0f, 1.0f);
      
      if (use_weights && p->weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) 
        {
	  // Prevent absorption and reweight accordingly
	  float survive = 1.0f - absorb;
	  absorb = 0.0f;
	  p->weight *= survive;
	  
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
	p->history |= SURFACE_DETECT;
	p->weight *= detect;
	return BREAK;
      }
      
      if (uniform_sample < absorb) {
	p->history |= SURFACE_ABSORB;
	return BREAK;
      }
      else if (uniform_sample < absorb + detect) {
	p->history |= SURFACE_DETECT;
	return BREAK;
      }
      else if (uniform_sample < absorb + detect + reflect_diffuse)
	return propagate_at_diffuse_reflector(p, s, rng);
      else
	return propagate_at_specular_reflector(p, s);
    }
  
} // propagate_at_surface

#endif

