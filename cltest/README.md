# Descriptions of different tests

test_api.py: tests the chroma.api module
* tested to work

test_bvh_simple.py: tests if BVH can be built for test meshes
* tested to work
* slight differences in structure of trees
* seems to be due to precision differences, as comparion between CUDA and OpenCL (on different machines) gives slightly different morton codes.

test_generator_photon.py: tests generation of photons
* requires pyG4 to have been built and linked in
* not tested

test_randomgen.py: tests implementation of random number generator
* produces ROOT file, output_testRNG.root, which should contain flat distribution between [0,1)
* works

test_sample_cdf.py: tests sampling from CDF
* produces ROOT File, output_sample_cdf.root, which contains gaussian distribution

test_propagation.py: checks for propagation bug
* also is test that machinery is working
* does not work, yet
* requres geant4!