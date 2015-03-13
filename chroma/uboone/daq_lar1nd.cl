// -*-c++-*-
#include "detector.h"
#include "random.h"

__kernel void
run_daq(__global clrandState *s, const unsigned int detection_state,
	// photon data
	const int first_photon, const int nphotons, __global float *photon_times, __global float* photon_pos,
	__global unsigned int *photon_histories, __global int *last_hit_triangles,
	__global float *weights,
	// detector info
	__global int *solid_map,
	__global int* solid_id_to_channel_index,
	// output
	__global uint* uint_adc, int nchannels, int ntdcs, float ns_per_tdc, float sim_t0_offset, // offset is a bit opaque. todo: remove or add parameter
	__global uint* channel_history,
	// geometry transforms
	__global float* ch_inv_rot, __global float* ch_inv_trans,
	// misc constants
	const float global_weight)
{
  // Simply histogram photons
  // We can also apply jitter if we want here (separate from the electronics reponse we will convolve)
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  union {
    unsigned int intval;
    float floatval;
  } charge;

  int photon_id = first_photon + id ;
  if (photon_id < nphotons) {
    __global clrandState* rng = s+id;
    int triangle_id = last_hit_triangles[photon_id];
		
    if (triangle_id > -1) {
      int solid_id = solid_map[triangle_id];
      unsigned int history = photon_histories[photon_id];
      int channel_index = solid_id_to_channel_index[solid_id];
      if (channel_index >= 0 && (history & detection_state)) {
	__global uint* pChannelADC = uint_adc + channel_index*ntdcs;
	float bar_pos[3];
	__global float* pos = photon_pos + 3*photon_id;
	__global float* inv_rot   = ch_inv_rot + 9*channel_index;  // 3x3 matrix
	__global float* inv_trans = ch_inv_trans + 3*channel_index; // 3-vector
	for (int i=0; i<3; i++) {
	  bar_pos[i] = 0.0f;
	  for (int j=0; j<3; j++) {
	    bar_pos[i]  += inv_rot[3*i + j]*pos[j];
	  }
	  bar_pos[i] += inv_trans[i];
	}

	float weight = weights[photon_id] * global_weight;
	if (clrand_uniform(rng, 0.0f, 1.0f) < weight) {
	  //float time = photon_times[photon_id] + sample_cdf(&rng, detector->time_cdf_len, detector->time_cdf_x, detector->time_cdf_y);
	  //float charge = sample_cdf(&rng, detector->charge_cdf_len, detector->charge_cdf_x, detector->charge_cdf_y);
	  float time =  photon_times[photon_id];
	  charge.floatval = weight;

	  // find bin
	  int tdc_bin = (int) (time-sim_t0_offset)/ns_per_tdc;
	  if ( tdc_bin>=0 && tdc_bin<ntdcs ){
	    //atomic_add( pChannelADC + tdc_bin, charge.intval );
	    atomic_add( pChannelADC + tdc_bin, 1 );
	    atomic_or( channel_history + channel_index, history );
	  }

	} // if weighted photon contributes
	
      } // if photon detected by a channel
      
    } // if photon terminated on surface
    
    //s[id] = rng; // update the random generator state, not needed
	
  }
};


__kernel void
convert_adc( __global uint* uint_adc, global float* adc, 
	     const int nchannels, const int ntdcs ) {

  int threadid = get_local_size(0)*get_group_id(0) + get_local_id(0);
  if ( threadid>=nchannels )
    return;
  union {
    unsigned int intval;
    float floatval;
  } charge;

  for (int itdc=0; itdc<ntdcs; itdc++) {
    //charge.intval = uint_adc[ threadid*ntdcs + itdc ];
    //adc[ threadid ] = charge.floatval;
    //adc[ threadid*ntdcs + itdc ] = charge.floatval;
    adc[ threadid*ntdcs + itdc ] = (float)uint_adc[ threadid*ntdcs + itdc ]; 
  }
};

__kernel void
get_earliest_hit_time( const int nchannels, const int ntdcs, const float ns_per_tdc, 
		       __global float* adc, __global uint* channel_history, 
		       __global float* earliest_time ) {
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  if ( id>=nchannels )
    return;
  earliest_time[id] = 1.0e9;

  __global float* pChannelADC = adc + ntdcs*id;
  for (int itdc=0; itdc<ntdcs; itdc++) {
    if ( pChannelADC[itdc]>0 ) {
      float ftdc = (float)itdc;
      earliest_time[id] = ftdc*ns_per_tdc;
      return;
    }
  }

};

