## Chroma Uboone

Fork of Chroma developed for the use with the MicroBooNE neutrino experiment.  
Actually, this is a fork of Simon C. Blythe's fork or Chroma.

Modifications by this fork:

* Modified gpu/tools.py to be able to load context with cuda context flags.  By default, loads the flag for device-mapped host memory.
* Fixed what seemed like a bug in detector.py where the pdf was defined with one too many bins.
* Fixed a bug in gpu/tools.py: make_gpu_struct.  There were problems with the numpy buffer interface.
* Modifications have been made to test functions that were used to help explore program behavior.  Some functions have been added.
