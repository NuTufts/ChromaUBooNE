// -*-c++-*-
//#include <curand_kernel.h>
#include "random.h"
#include "sorting.h"

__kernel void bin_hits(int nchannels, 
		       __global float *channel_q, __global float *channel_time, __global unsigned int *hitcount, 
		       int tbins, float tmin, float tmax, int qbins,
		       float qmin, float qmax, 
		       __global unsigned int *pdf)
{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);

  if (id >= nchannels)
    return;

  unsigned int q = channel_q[id];
  float t = channel_time[id];
	
  if (t < 1.0e8f && t >= tmin && t < tmax && q >= qmin && q < qmax) {
    hitcount[id] += 1;
		
    int tbin = (t - tmin) / (tmax - tmin) * tbins;
    int qbin = (q - qmin) / (qmax - qmin) * qbins;
	    
    // row major order (channel, t, q)
    int bin = id * (tbins * qbins) + tbin * qbins + qbin;
    pdf[bin] += 1;
  }
}

__kernel void accumulate_bincount(int nchannels, int ndaq,
				  __global unsigned int *event_hit,
				  __global float *event_time, 
				  __global float *mc_time,
				  __global unsigned int *hitcount, __global unsigned int *bincount,
				  float min_twidth, float tmin, float tmax,
				  int min_bin_content,
				  __global unsigned int *map_channel_id_to_hit_offset,
				  __global unsigned int *work_queues)
{
  //int channel_id = threadIdx.x + blockDim.x * blockIdx.x;
  int channel_id = get_local_size(0)*get_group_id(0) + get_local_id(0);
	
  if (channel_id >= nchannels)
    return;

  float channel_hitcount = hitcount[channel_id];
  float channel_bincount = bincount[channel_id];
  float channel_event_time = event_time[channel_id];
  int channel_event_hit = event_hit[channel_id];
    
  __global unsigned int *work_queue = work_queues + map_channel_id_to_hit_offset[channel_id] * (ndaq + 1);
  unsigned int next_slot = work_queue[0];
	
  for (int i=0; i < ndaq; i++) {
    int read_offset = nchannels * i + channel_id;

    // Was this channel hit in the Monte Carlo?
    float channel_mc_time = mc_time[read_offset];
    if (channel_mc_time >= 1.0e8f)
      continue; // Nothing else to do
	
    // Is this channel inside the range of the PDF?
    float distance;
    if (channel_mc_time < tmin || channel_mc_time > tmax)
      continue;  // Nothing else to do
		
    channel_hitcount += 1;

    // Was this channel hit in the event-of-interest?
    if (!channel_event_hit)
      continue; // No need to update PDF value for unhit channel
    
    // Are we inside the minimum size bin?
    distance = fabs(channel_mc_time - channel_event_time);
    if (distance < min_twidth/2.0f) {
      channel_bincount += 1;
    }

    // Add this hit to the work queue if we also need to sort it into the 
    // nearest_mc_list
    if (channel_bincount < min_bin_content) {
      work_queue[next_slot] = read_offset;
      next_slot++;
    }
  }
  
  hitcount[channel_id] = channel_hitcount;
  bincount[channel_id] = channel_bincount;
  if (channel_event_hit)
    work_queue[0] = next_slot;
}
	
__kernel void accumulate_nearest_neighbor(int nhit,
					  int ndaq,
					  __global unsigned int *map_hit_offset_to_channel_id,
					  __global unsigned int *work_queues,
					  __global float *event_time,
					  __global float *mc_time,
					  __global float *nearest_mc, 
					  int min_bin_content)
{
  //int hit_id = threadIdx.x + blockDim.x * blockIdx.x;
  int hit_id = get_local_size(0)*get_group_id(0) + get_local_id(0);
    
  if (hit_id >= nhit)
    return;

  __global unsigned int *work_queue = work_queues + hit_id * (ndaq + 1);
  int queue_items = work_queue[0] - 1;

  int channel_id = map_hit_offset_to_channel_id[hit_id];
  float channel_event_time = event_time[channel_id];
    
  float distance_table[1000];
  int distance_table_len = 0;
  
  // Load existing distance table
  int offset = min_bin_content * hit_id;    
  for (int i=0; i < min_bin_content; i++) {
    float d = nearest_mc[offset + i];
    if (d > 1.0e8f)
      break;
    
    distance_table[distance_table_len] = d;
    distance_table_len++;
  }
    
  // append new entries
  for (int i=0; i < queue_items; i++) {
    unsigned int read_offset = work_queue[i+1];
    float channel_mc_time = mc_time[read_offset];
    float distance = fabs(channel_mc_time - channel_event_time);

    distance_table[distance_table_len] = distance;
    distance_table_len++;
  }

  // Sort table
  piksrt_device(distance_table_len, distance_table);
    
  // Copy first section of table back out to global memory
  distance_table_len = min(distance_table_len, min_bin_content);
  for (int i=0; i < distance_table_len; i++) {
    nearest_mc[offset + i] = distance_table[i];
  }
}

__kernel void accumulate_nearest_neighbor_block(int nhit,
						int ndaq,
						__global unsigned int *map_hit_offset_to_channel_id,
						__global unsigned int *work_queues,
						__global float *event_time,
						__global float *mc_time,
						__global float *nearest_mc, 
						int min_bin_content)
{
  //int hit_id = blockIdx.x;
  int hit_id = get_group_id(0);
    
  __local float distance_table[1000];
  //__local unsigned int *work_queue;
  __global unsigned int *work_queue;
  __local int queue_items;
  __local int channel_id;
  __local float channel_event_time;
  __local int distance_table_len;
  __local int offset;

  //if (threadIdx.x == 0) {
  if (get_local_id(0) == 0) {
    work_queue = work_queues + hit_id * (ndaq + 1);
    queue_items = work_queue[0] - 1;

    channel_id = map_hit_offset_to_channel_id[hit_id];
    channel_event_time = event_time[channel_id];
    distance_table_len = min_bin_content;
    offset = min_bin_content * hit_id;    
  }

  //__syncthreads();
  barrier( CLK_LOCAL_MEM_FENCE );

  // Load existing distance table
  //for (int i=threadid.x; i < min_bin_content; i += blockDim.x) {
  for (int i=get_local_id(0); i < min_bin_content; i += get_local_size(0)) {
    float d = nearest_mc[offset + i];
    if (d > 1.0e8f) {
      atomic_min(&distance_table_len, i);
      break;
    }
    distance_table[i] = d;
  }
  
  //__syncthreads();
  barrier( CLK_LOCAL_MEM_FENCE );

  // append new entries
  //for (int i=threadIdx.x; i < queue_items; i += blockDim.x) {
  for (int i=get_local_id(0); i < queue_items; i +=  get_local_size(0)) {
    unsigned int read_offset = work_queue[i+1];
    float channel_mc_time = mc_time[read_offset];
    float distance = fabs(channel_mc_time - channel_event_time);
    distance_table[distance_table_len + i] = distance;
  }

  //__syncthreads();
  barrier( CLK_LOCAL_MEM_FENCE );
  
  //if (threadIdx.x == 0) {
  if ( get_local_id(0)==0 ) {
    distance_table_len += queue_items;
      // Sort table
    piksrt(distance_table_len, distance_table);
    // Copy first section of table back out to global memory
    distance_table_len = min(distance_table_len, min_bin_content);
  }
    
  //__syncthreads();
  barrier( CLK_LOCAL_MEM_FENCE );

  //for (int i=threadIdx.x; i < distance_table_len; i += blockDim.x) {
  for (int i=get_local_id(0); i < distance_table_len; i += get_local_size(0) ) {
    nearest_mc[offset + i] = distance_table[i];
  }
}

__kernel void accumulate_moments(int time_only, int nchannels,
				 __global float *mc_time,
				 __global float *mc_charge,
				 float tmin, float tmax,
				 float qmin, float qmax,
				 __global unsigned int *mom0,
				 __global float *t_mom1,
				 __global float *t_mom2,
				 __global float *q_mom1,
				 __global float *q_mom2)

{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
	
  if (id >= nchannels)
    return;
	
  // Was this channel hit in the Monte Carlo?
  float channel_mc_time = mc_time[id];

  // Is this channel inside the range of the PDF?
  if (time_only) {
    if (channel_mc_time < tmin || channel_mc_time > tmax)
      return;  // Nothing else to do
    
    mom0[id] += 1;
    t_mom1[id] += channel_mc_time;
    t_mom2[id] += channel_mc_time*channel_mc_time;
  }
  else { // time and charge PDF
    float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
	
    if (channel_mc_time < tmin || channel_mc_time > tmax ||
	channel_mc_charge < qmin || channel_mc_charge > qmax)
      return;  // Nothing else to do
    
    mom0[id] += 1;
    t_mom1[id] += channel_mc_time;
    t_mom2[id] += channel_mc_time*channel_mc_time;
    q_mom1[id] += channel_mc_charge;
    q_mom2[id] += channel_mc_charge*channel_mc_charge;
  }
}

__constant static const float invroot2 = 0.70710678118654746f; // 1/sqrt(2)
__constant static const float rootPiBy2 = 1.2533141373155001f; // sqrt(M_PI/2)

__kernel void accumulate_kernel_eval(int time_only, int nchannels, 
				     __global unsigned int *event_hit,
				     __global float *event_time, __global float *event_charge, __global float *mc_time,
				     __global float *mc_charge,
				     float tmin, float tmax,
				     float qmin, float qmax,
				     __global float *inv_time_bandwidths,
				     __global float *inv_charge_bandwidths,
				     __global unsigned int *hitcount,
				     __global float *time_pdf_values,
				     __global float *charge_pdf_values)
		       
{
  //int id = threadIdx.x + blockDim.x * blockIdx.x;
  int id = get_local_size(0)*get_group_id(0) + get_local_id(0);
	
  if (id >= nchannels)
    return;
	
  // Was this channel hit in the Monte Carlo?
  float channel_mc_time = mc_time[id];
	
  // Is this channel inside the range of the PDF?
  if (time_only) {
    if (channel_mc_time < tmin || channel_mc_time > tmax)
      return;  // Nothing else to do
    
    // This MC information is contained in the PDF
    hitcount[id] += 1;
    
    // Was this channel hit in the event-of-interest?
    int channel_event_hit = event_hit[id];
    if (!channel_event_hit)
      return; // No need to update PDF value for unhit channel
    
    // Kernel argument
    float channel_event_time = event_time[id];
    float inv_bandwidth = inv_time_bandwidths[id];
    float arg = (channel_mc_time - channel_event_time) * inv_bandwidth;

    // evaluate 1D Gaussian normalized within time window
    float term = exp(-0.5f * arg * arg) * inv_bandwidth;
    
    float norm = tmax - tmin;
    if (inv_bandwidth > 0.0f) {
      float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
      float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
      norm = (erf(hiarg) - erf(loarg)) * rootPiBy2;
    }
    time_pdf_values[id] += term / norm;
  }
  else { // time and charge PDF
    float channel_mc_charge = mc_charge[id]; // int->float conversion because DAQ just returns an integer
    
    if (channel_mc_time < tmin || channel_mc_time > tmax ||
	channel_mc_charge < qmin || channel_mc_charge > qmax)
      return;  // Nothing else to do
    
    // This MC information is contained in the PDF
    hitcount[id] += 1;
    
    // Was this channel hit in the event-of-interest?
    int channel_event_hit = event_hit[id];
    if (!channel_event_hit)
      return; // No need to update PDF value for unhit channel
    
    
    // Kernel argument: time dim
    float channel_event_obs = event_time[id];
    float inv_bandwidth = inv_time_bandwidths[id];
    float arg = (channel_mc_time - channel_event_obs) * inv_bandwidth;
    
    float norm = tmax - tmin;
    if (inv_bandwidth > 0.0f) {
      float loarg = (tmin - channel_mc_time)*inv_bandwidth*invroot2;
      float hiarg = (tmax - channel_mc_time)*inv_bandwidth*invroot2;
      norm = (erf(hiarg) - erf(loarg)) * rootPiBy2;
    }
    float term = exp(-0.5f * arg * arg);
    
    time_pdf_values[id] += term / norm;

    // Kernel argument: charge dim
    channel_event_obs = event_charge[id];
    inv_bandwidth = inv_charge_bandwidths[id];
    arg = (channel_mc_charge - channel_event_obs) * inv_bandwidth;
    
    norm = qmax - qmin;
    if (inv_bandwidth > 0.0f) {
      float loarg = (qmin - channel_mc_charge)*inv_bandwidth*invroot2;
      float hiarg = (qmax - channel_mc_charge)*inv_bandwidth*invroot2;
      norm = (erf(hiarg) - erf(loarg)) * rootPiBy2;
    }
    
    term = exp(-0.5f * arg * arg);
    
    charge_pdf_values[id] += term / norm;
  }
}
