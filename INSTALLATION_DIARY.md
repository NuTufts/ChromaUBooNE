# Installation Diary

List of steps I took to install Chroma from scratch.
Note, I suggest trying to use ShrinkWrap according to
http://chroma.bitbucket.org.  But I've included these notes
here just in case it is useful for others.  

I installed Chroma on a machine without root privlegdes.
My local python packages went into ~/.local. Any other packages went into
~/software/bin, ~/software/lib, ~/software/include, etc. Using gcc 4.8.


### BIP2, bz2

* needed it to get proper python build
* also needed for BOOST
* installed into ~/software

### Python (2.7.8)

Starting with a python build.

* wget https://www.python.org/ftp/python/2.7.8/Python-2.7.8.tgz
* Added ~/software/lib and ~/software/include to LDFLAGS and CPPFLAGS
* Also added libbz2.so sim link in ~/software/lib so that python would find it
* configure
* make, make install
* Set path to search ~/.local/bin first in ~/.bashrc so this python active

### setup_tools

* followed instructions here: https://forcecarrier.wordpress.com/2013/07/26/installing-pip-virutalenv-in-sudo-free-way/
* wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
* python ez_setup.py --user

### PIP

* wget https://bootstrap.pypa.io/get-pip.py
* python get-pip.py --user

### virtualenv

* pip install virtualenv

### numpy

* pip install numpy

### geant4 (4.9.5.p02)

*  cmake -DCMAKE_INSTALL_PREFIX=/Users/twongjirad/software/geant4.9.5.p02 /Users/twongjirad/software/geant4.9.5.p02/geant4.9.5.p02-src -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_RAYTRACER_X11=ON -Wno-dev -DGEANT4_USE_GDML=ON

### Chroma (pre-install)

* At this point I cloned my chroma repo and built a virtual environment for chroma.  
  You can choose to do this at any point once virtualenv is installed.
* environment folder is chroma_env

### Boost

* using chroma_dep/boost-1.57.0/setup.py.
* set environment variables: BZIP2_INCLUDE, BZIP2_LIBPATH
* BZIP2_INCLUDE=/home/tmw/software/include
* BZIP2_LIBPATH=/home/tmw/software/lib
* compiled without a problem

### PyUblas

* cloned git repo
* python configure.py --boost-lib-dir=/home/tmw/chroma_uboone/chroma_env/lib --boost-inc-dir=/home/tmw/chroma_uboone/chroma_env/include --boost-python-libname=boost_python
* installed in chroma_env

### PyCUDA

* python configure.py --boost-lib-dir=/home/tmw/chroma_uboone/chroma_env/lib --boost-inc-dir=/home/tmw/chroma_uboone/chroma_env/include --boost-python-libname=boost_python --cudadrv-lib-dir=/usr/lib64/nvidia
* note that I had to specify where my NVIDIA libraries were.  other machines did not require this
* ran examples n pycuda/examples to test installation


### CLHEP

* downloaded from: http://proj-clhep.web.cern.ch/proj-clhep/DISTRIBUTION/tarFiles/clhep-2.2.0.4.tgz
* followed instructions. built with no problem. installed in ~/software

### g4py_chroma

* this took some work.  
* made minor modifications for it to compile
* my fork: https://bitbucket.org/renatechordate/g4py/src
* used modified setup.py script from http://mtrr.org/chroma_pkgs/g4py_chroma/, so that it would call configure flags that would specify dependencies.


### xserces

* wget http://www.carfab.com/apachesoftware//xerces/c/3/sources/xerces-c-3.1.1.tar.gz

### freetype (for matplotlib)

* followed instructions on web

### Chroma, finally

* used setup.py




