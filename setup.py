from setuptools import setup, find_packages, Extension
import subprocess
import os
import ROOT # avoids KeyError: 'ROOT' from pyroot atexit cleanup  +593 `facade = sys.modules[ __name__ ]` while running tests 

libraries = ['boost_python']
extra_objects = []

if 'VIRTUAL_ENV' in os.environ:
    boost_lib = os.path.join(os.environ['VIRTUAL_ENV'],'lib','libboost_python.so')
    if os.path.exists(boost_lib):
        # use local copy of boost
        extra_objects.append(boost_lib)
        libraries.remove('boost_python')

def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output

geant4_cflags = check_output(['geant4-config','--cflags']).split()
geant4_libs = check_output(['geant4-config','--libs']).split()
# For GEANT4.9.4 built without cmake
try:
    clhep_libs = check_output(['clhep-config','--libs']).split()
except OSError:
    clhep_libs = []


include_dirs=['src']

##### figure out location of pyublas headers
#try:
from imp import find_module
file, pathname, descr = find_module("pyublas")
#pathname = "./chroma_env/lib/python2.7/site-packages/pyublas/"
from os.path import join
include_dirs.append(join(pathname, "include"))
print "pyublas headers: ",join(pathname, "include")
#except:
#    pass  # Don't throw exceptions if prereqs not installed yet

#####

if 'VIRTUAL_ENV' in os.environ:
    include_dirs.append(os.path.join(os.environ['VIRTUAL_ENV'], 'include'))
try:
    import numpy.distutils
    include_dirs += numpy.distutils.misc_util.get_numpy_include_dirs()
except:
    pass # if numpy doesn't exist yet

setup(
    name = 'Chroma',
    version = '0.5',
    packages = find_packages(),
    include_package_data=True,
    package_data = { 'chroma':['models/*.stl*',
                               'cuda/*.cu','cuda/*.h',
                               'cl/*.cl','cl/*.h','cl/Random123/*.h','cl/Random123/conventional/*.h','cl/Random123/features/*.h', 
                               'uboone/*.dat','uboone/*.cu','uboone/*.cl'],
                     },
    scripts = ['bin/chroma-sim', 'bin/chroma-cam',
               'bin/chroma-geo', 'bin/chroma-bvh',
               'bin/chroma-server'],
    ext_modules = [
        Extension('chroma.generator._g4chroma',
                  ['src/G4chroma.cc'],
                  include_dirs=include_dirs,
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs+clhep_libs,
                  extra_objects=extra_objects,
                  libraries=libraries,
                  ),
        Extension('chroma.generator.mute',
                  ['src/mute.cc'],
                  include_dirs=include_dirs,
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs+clhep_libs,
                  extra_objects=extra_objects,
                  libraries=libraries),
        ],
 
    setup_requires = ['pyublas'],
#    install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda', 
#                        'numpy>=1.6', 'pygame', 'nose', 'sphinx', 'unittest2'],
    install_requires = ['uncertainties','pyzmq-static','spnav', 
                        'numpy>=1.6', 'pygame', 'nose', 'sphinx', 'unittest2'],
    test_suite = 'nose.collector',
    
)
