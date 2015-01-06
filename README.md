## Chroma Uboone

Fork of Chroma developed for the use with the MicroBooNE neutrino experiment.  
Actually, this is a fork of Simon C. Blythe's fork or Chroma.

# Modifications by this fork:

* Modified gpu/tools.py to be able to load context with cuda context flags.  By default, loads the flag for device-mapped host memory.
* Fixed what seemed like a bug in detector.py where the pdf was defined with one too many bins.
* Fixed a bug in gpu/tools.py: make_gpu_struct.  There were problems with the numpy buffer interface.
* Modifications have been made to test functions that were used to help explore program behavior.  Some functions have been added.

# BVH Precision differences between opencl and cuda:

* I see some precision effects that causes differences in the BVH tree between opencl and cuda.  Tests used the companion cube mesh.
* Get differences in quantize centroid values which then propagates to rest of tree structure.
* From my studies, differences are only off by one quantum along any axis
* Example: Mismatch: ID=286  cuda= 114847925804877  cl= 114847925804876
  CUDA:  [37663, 55944, 4589]  CL:  [37662, 55944, 4589]  diff= [1 0 0]
* CUDA value always seems to be bigger
* For companion cube test about 1.9% of all morton codes are different

# Random Number Generator

* Unlike CUDA, OpenCL does not have a built-in random number generator
* Implemented RNG using Random123 package: http://www.deshawresearch.com/downloads/download_random123.cgi/. Included in repo.
* Random123 is in principle stateless. Created state vector in gpu/clrandstate.py. Implemented uniform sampler in cl/random.cl
* tests, cltest/test_randomgen.py and cltest/test_sample_cdf.py, generate flat and gaussian distribution, respectively. Seems to work.
