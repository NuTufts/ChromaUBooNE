# Installation Diary

List of steps I took to install Chroma from scratch.
Note, I suggest trying to use ShrinkWrap according to
http://chroma.bitbucket.org.  But I've included these notes
here just in case it is useful for others.  

I installed Chroma on two machines without root priveleges.

## Installation 1

### Python (2.7.8)

Starting with a python build.

* wget https://www.python.org/ftp/python/2.7.8/Python-2.7.8.tgz
* configure
* make, make install
* Worked without problems.
* Set path to search ~/.local/bin first

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

## geant4 (4.9.5.p02)

*  cmake -DCMAKE_INSTALL_PREFIX=/Users/twongjirad/software/geant4.9.5.p02 /Users/twongjirad/software/geant4.9.5.p02/geant4.9.5.p02-src -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_RAYTRACER_X11=ON -Wno-dev -DGEANT4_USE_GDML=ON

