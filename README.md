# Chroma Uboone

Fork of Chroma developed for the use with the MicroBooNE neutrino experiment.  
Actually, this is a fork of Simon C. Blyth's fork of Chroma.

## Modifications by this fork:

* Modified gpu/tools.py to be able to load context with cuda context flags.  
  By default, loads the flag for device-mapped host memory. 
  Needed for running on an older card (C1080).
* Fixed what seemed like a bug in detector.py where the pdf was defined with one too many bins.
* Fixed a bug in gpu/tools.py: make_gpu_struct.  There were problems with the numpy buffer interface.
* Modifications have been made to test functions that were used to help explore program behavior.  Some functions have been added.
* Added abilit to run with OpenCL, using pyOpenCL
* Providing new module, importgeo, that uses tools written by S.C. Blyth to import geometries described in
  COLLADA file output by G4DAE C++ library for Geant4
* Having ROOT with PyROOT extension will allow some tests to make plots. 
  Not required (or at least tried not to make it required).

## OpenCL capability

* API, CUDA or OpenCL, is selectable using functions in api.py
* api.py also has functions to test which API
* opencl has some big differences that affect chroma: no device-mapped memory, no built-in random number generator, and different struct creation/definition.
* because of these differences, port was not done so cleanly.  many modules have been reorganized and classes changed. merging this fork may require coordination.

### BVH Precision differences between opencl and cuda:

* I see some precision effects which causes differences in the BVH tree between opencl and cuda.  Tests used the companion cube mesh.
* Get differences in quantize centroid values which then propagates to rest of tree structure.
* From my studies, differences are only off by one quantum along any axis
* Example Mismatch: Triangle ID=286, morton codes are  cuda= 114847925804877  cl= 114847925804876
* Bounding box for mismatch is:
  CUDA=[37663, 55944, 4589],  CL=[37662, 55944, 4589], diff= [1 0 0]
* CUDA value always seems to be bigger
* For companion cube mesh about 1.9% of all morton codes are different

### Random Number Generator

* Unlike CUDA, OpenCL does not have a built-in random number generator
* Implemented RNG using Random123 package: http://www.deshawresearch.com/downloads/download_random123.cgi/. Included in repo.
* Random123 is in principle stateless. Created state vector in gpu/clrandstate.py. Implemented uniform sampler in cl/random.cl. 
Require state, because being able to reproduce results is a requirement for particle physics MC.
* tests, cltest/test_randomgen.py and cltest/test_sample_cdf.py, generate flat and gaussian distribution, respectively. Seems to work.
