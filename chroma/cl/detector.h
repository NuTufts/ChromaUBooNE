#ifndef __DETECTOR_H__
#define __DETECTOR_H__

typedef struct Detector
{
  // Order in decreasing size to avoid alignment problems
  __global int *solid_id_to_channel_index;

  __global float *time_cdf_x;
  __global float *time_cdf_y;

  __global float *charge_cdf_x;
  __global float *charge_cdf_y;

  int nchannels;
  int time_cdf_len;
  int charge_cdf_len;
  float charge_unit; 
  // Convert charges to/from quantized integers with
  // q_int = (int) roundf(q / charge_unit )
  // q = q_int * charge_unit
} Detector;

void fill_detector_struct( __local Detector* det,
			   __global int* solid_id_to_channel_index, 
			   __global float* time_cdf_x, __global float* time_cdf_y,
			   __global float *charge_cdf_x, __global float *charge_cdf_y,
			   int nchannels, int time_cdf_len, int charge_cdf_len, float charge_unit );

void fill_detector_struct( __local Detector* det,
			   __global int* solid_id_to_channel_index, 
			   __global float* time_cdf_x, __global float* time_cdf_y,
			   __global float *charge_cdf_x, __global float *charge_cdf_y,
			   int nchannels, int time_cdf_len, int charge_cdf_len, float charge_unit ) {
  det->solid_id_to_channel_index = solid_id_to_channel_index;
  det->time_cdf_x                = time_cdf_x;
  det->time_cdf_y                = time_cdf_y;
  det->charge_cdf_x              = charge_cdf_x;
  det->charge_cdf_y              = charge_cdf_y;
  det->nchannels                 = nchannels;
  det->time_cdf_len              = time_cdf_len;
  det->charge_cdf_len            = charge_cdf_len;
  det->charge_unit               = charge_unit;
};

#endif // __DETECTOR_H__
