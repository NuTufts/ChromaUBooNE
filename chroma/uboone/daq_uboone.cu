// -*-c++-*-
#include "detector.h"
#include "random.h"


extern "C"
{

__global__ void
run_daq(curandState *s, unsigned int detection_state,
	// photon data
	int first_photon, int nphotons, float *photon_times,
	unsigned int *photon_histories, int *last_hit_triangles,
	float *weights,
	// detector info
	int *solid_map,
	Detector *detector,
	// output
	float* adc, int nchannels, int ntdcs, float ns_per_tdc, float sim_t0_offset, // offset is a bit opaque. todo: remove or add parameter
	uint* channel_history,
	// misc constants
	float global_weight)
{
  // Simply histogram photons
  // We can also apply jitter if we want here (separate from the electronics reponse we will convolve)
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < nphotons) {
    curandState rng = s[id];
    int photon_id = id + first_photon;
    int triangle_id = last_hit_triangles[photon_id];
		
    if (triangle_id > -1) {
      int solid_id = solid_map[triangle_id];
      unsigned int history = photon_histories[photon_id];
      int channel_index = detector->solid_id_to_channel_index[solid_id];
      if (channel_index >= 0 && (history & detection_state)) {
	float* pChannelADC = adc + channel_index*ntdcs;
	float weight = weights[photon_id] * global_weight;
	if (curand_uniform(&rng) < weight) {
	  //float time = photon_times[photon_id] + sample_cdf(&rng, detector->time_cdf_len, detector->time_cdf_x, detector->time_cdf_y);
	  //float charge = sample_cdf(&rng, detector->charge_cdf_len, detector->charge_cdf_x, detector->charge_cdf_y);
	  float time =  photon_times[photon_id];
	  float charge = weight;

	  // find bin
	  int tdc_bin = (int) (time-sim_t0_offset)/ns_per_tdc;
	  if ( tdc_bin>=0 && tdc_bin<ntdcs ){
	    atomicAdd( pChannelADC + tdc_bin, charge );
	    atomicOr( channel_history + channel_index, history );
	  }

	} // if weighted photon contributes
	
      } // if photon detected by a channel
      
    } // if photon terminated on surface
    
    s[id] = rng; // update the random generator state
	
  }
}

__global__ void
get_earliest_hit_time( const int nchannels, const int ntdcs, const float ns_per_tdc, 
		       float* adc, uint* channel_history, 
		       float* earliest_time ) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if ( id>=nchannels )
    return;
  earliest_time[id] = 1.0e9;

  float* pChannelADC = adc + ntdcs*id;
  for (int itdc=0; itdc<ntdcs; itdc++) {
    if ( pChannelADC[itdc]>0 ) {
      float ftdc = (float)itdc;
      earliest_time[id] = ftdc*ns_per_tdc;
      return;
    }
  }

};

} // extern "C"
