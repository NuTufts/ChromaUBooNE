// -*-c++-*-
#include "detector.h"
#include "random.h"

unsigned int float_to_sortable_int(float f);
float sortable_int_to_float(unsigned int i);

unsigned int float_to_sortable_int(float f)
{
  return as_uint(f);
  //return __float_as_int(f);
    //int i = __float_as_int(f);
    //unsigned int mask = -(int)(i >> 31) | 0x80000000;
    //return i ^ mask;
}

float sortable_int_to_float(unsigned int i)
{
  return as_float(i);
  //return __int_as_float(i);
    //unsigned int mask = ((i >> 31) - 1) | 0x80000000;
    //return __int_as_float(i ^ mask);
}


__kernel void reset_earliest_time_int(float maxtime, int ntime_ints, __global unsigned int *time_ints)
{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  if (id < ntime_ints) {
    unsigned int maxtime_int = float_to_sortable_int(maxtime);
    time_ints[id] = maxtime_int;
  }
}

__kernel void run_daq(__global clrandState *s, 
		      unsigned int detection_state, int first_photon, int nphotons, 
		      __global float *photon_times,
		      __global unsigned int *photon_histories, __global int *last_hit_triangles,
		      __global float *weights, __global int *solid_map,
		      // -- Detector *detector, --
		      __global int* solid_id_to_channel_index,
		      __global float* time_cdf_x,   __global float* time_cdf_y,
		      __global float *charge_cdf_x, __global float *charge_cdf_y,
		      int nchannels, int time_cdf_len, int charge_cdf_len, float charge_unit,
		      // -------------------------
		      __global unsigned int *earliest_time_int,
		      __global unsigned int *channel_q_int, __global unsigned int *channel_histories,
		      float global_weight)
{
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);

  __local Detector detector;
  if ( get_local_id(0)==0 ) {
    fill_detector_struct( &detector, solid_id_to_channel_index, 
			  time_cdf_x, time_cdf_y,
			  charge_cdf_x, charge_cdf_y,
			  nchannels, time_cdf_len, charge_cdf_len, charge_unit );
  }
  barrier( CLK_LOCAL_MEM_FENCE );

  if (id < nphotons) {
    __global clrandState* rng = s+id;
    int photon_id    = id + first_photon;
    int triangle_id  = last_hit_triangles[photon_id];
		
    if (triangle_id > -1) {
      int solid_id         = solid_map[triangle_id];
      unsigned int history = photon_histories[photon_id];
      int channel_index    = detector.solid_id_to_channel_index[solid_id];

      if (channel_index >= 0 && (history & detection_state)) {

	float weight = weights[photon_id] * global_weight;
	if (clrand_uniform(rng,0.0f,1.0f) < weight) {
	  float time = photon_times[photon_id] + sample_cdf(rng, detector.time_cdf_len, detector.time_cdf_x, detector.time_cdf_y);

	  unsigned int time_int = float_to_sortable_int(time);
	  
	  float charge = sample_cdf(rng, detector.charge_cdf_len,
				    detector.charge_cdf_x,
				    detector.charge_cdf_y);
	  unsigned int charge_int = round( charge / detector.charge_unit );
	  
	  //atomicMin(earliest_time_int + channel_index, time_int);
	  //atomicAdd(channel_q_int + channel_index, charge_int);
	  //atomicOr(channel_histories + channel_index, history);

	  atomic_min( earliest_time_int + channel_index, time_int );
	  atomic_add( channel_q_int + channel_index, charge_int );
	  atomic_or( channel_histories + channel_index, history );
	} // if weighted photon contributes
	
      } // if photon detected by a channel
      
    } // if photon terminated on surface
    
    //s[id] = rng; // no need
    
  } // (id < nphotons)
  
}

__kernel void run_daq_many(__global clrandState *s, unsigned int detection_state,
			   int first_photon, int nphotons, __global float *photon_times,
			   __global unsigned int *photon_histories, __global int *last_hit_triangles,
			   __global float *weights,
			   __global int *solid_map,
			   // -- Detector *detector, --
			   __global int* solid_id_to_channel_index,
			   __global float* time_cdf_x,   __global float* time_cdf_y,
			   __global float *charge_cdf_x, __global float *charge_cdf_y,
			   int nchannels, int time_cdf_len, int charge_cdf_len, float charge_unit,
			   // -------------------------
			   __global unsigned int *earliest_time_int,
			   __global unsigned int *channel_q_int, __global unsigned int *channel_histories,
			   int ndaq, int channel_stride, float global_weight)
{
  __local int photon_id;
  __local int triangle_id;
  __local int solid_id;
  __local int channel_index;
  __local unsigned int history;
  __local float photon_time;
  __local float weight;
  
  __local Detector detector;

  if (get_local_id(0) == 0) {

    fill_detector_struct( &detector,
			  solid_id_to_channel_index, 
			  time_cdf_x, time_cdf_y,
			  charge_cdf_x, charge_cdf_y, 
			  nchannels, time_cdf_len, charge_cdf_len, charge_unit );

    //photon_id = first_photon + blockIdx.x;
    photon_id = first_photon + get_group_id(0);
    triangle_id = last_hit_triangles[photon_id];
	
    if (triangle_id > -1) {
      solid_id = solid_map[triangle_id];
      history = photon_histories[photon_id];
      channel_index = detector.solid_id_to_channel_index[solid_id];
      photon_time = photon_times[photon_id];
      weight = weights[photon_id] * global_weight;
    }
  }

  barrier( CLK_LOCAL_MEM_FENCE );

  if (triangle_id == -1 || channel_index < 0 || !(history & detection_state))
    return;

  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
  __global clrandState* rng = s+id;

  //for (int i = threadIdx.x; i < ndaq; i += blockDim.x) {
  for ( int i = get_local_id(0); i < ndaq; i += get_local_size(0) ) {
    int channel_offset = channel_index + i * channel_stride;

    if (clrand_uniform(rng,0.0f,1.0f) < weight) {
      float time = photon_time + clrand_normal(rng) + sample_cdf(rng, detector.time_cdf_len, detector.time_cdf_x, detector.time_cdf_y);
      
      //unsigned int time_int = float_to_sortable_int(time);
      unsigned int time_int = as_uint(time);
	
      float charge = sample_cdf(rng, detector.charge_cdf_len, detector.charge_cdf_x,detector.charge_cdf_y);

      unsigned int charge_int = round(charge / detector.charge_unit);
	    
      //atomicMin(earliest_time_int + channel_offset, time_int);
      //atomicAdd(channel_q_int + channel_offset, charge_int);
      //atomicOr(channel_histories + channel_offset, history);
      atomic_min(earliest_time_int + channel_offset, time_int);
      atomic_add(channel_q_int + channel_offset, charge_int);
      atomic_or(channel_histories + channel_offset, history);
    }
  }

  //s[id] = rng;
}

__kernel void convert_sortable_int_to_float(int n, __global unsigned int *sortable_ints, __global float *float_output)
{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
    
  if (id < n) {
    //float_output[id] = sortable_int_to_float(sortable_ints[id]);
    float_output[id] = sortable_int_to_float(sortable_ints[id]);
  }
}

__kernel void convert_charge_int_to_float(// --- Detector *detector, ---
					  __global int* solid_id_to_channel_index,
					  __global float* time_cdf_x,   __global float* time_cdf_y,
					  __global float *charge_cdf_x, __global float *charge_cdf_y,
					  int nchannels, int time_cdf_len, int charge_cdf_len, float charge_unit,
					  // ---------------------------
					  __global unsigned int *charge_int,
					  __global float *charge_float)
{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
	
  //if (id < detector->nchannels)
  //charge_float[id] = charge_int[id] * detector->charge_unit;

  if (id < nchannels)
    charge_float[id] = charge_int[id] * charge_unit;

}
