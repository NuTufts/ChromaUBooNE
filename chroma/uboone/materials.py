import os,sys
import numpy as np

# This module has functions and defintions to load the optical 
# properties required by the MicroBooNE detector

materialnames = ["LAr",                       # liquid argon [ may have its own module one day ]
                 "Titanium",                  # for the wires (fancy)
                 "STEEL_STAINLESS_Fe7Cr2Ni",  # cryostat walls
                 "Acrylic",                   # wavelength shifting plates
                 "Glass",                     # pmt window
                 "bialkali",                  # pmt photocathode
                 "Vacuum",
                 "PU_foam_light",             # mastic insulation. Irrelevant.
                 "PU_foam_dense",             # mastic insulation. Irrelevant.
                 "Air",                       # lab air, Irrelevant
                 "Concrete",]                 # Irrelevant
# --------------------------------------------------------------------------------
# what needs to be specified.
# Materials need:
#   - refractive_index (can be function of wavelength)
#   - absorption_length (function of wavelength)
#   - scattering_length (function of wavelength)
# See chroma.geometry: class Material for more information
# --------------------------------------------------------------------------------
# LAr: Liquid Argon
# * Refractice index from
# Sinnock, A. C. Refractive indices of the condensed rare gases, argon, krypton and xenon. 
# Journal of Physics C: Solid State Physics 13, 2375 (1980).
# Measured at 83 K at 546.1 nm
# Values at 260 and 400 are dummpy values
# * Scattering Length from
# Ishida et al. NIMA 384 (1997) 380-386: 66+/-3 cm
# [USED] Seidel et al. NIMA 489 (2002) 189–194: 90 cm (calculated)
# * Absorption Length
# Going to be a function of puity and other inputs. 
# Taking the easy route and using 80.9 cm for now (from C. Rubbia)

lar_refractive_index = np.array( [ (260.0, 1.2316),
                                   (400.0, 1,2316),
                                   (546.1, 1.2316) ] )

def load_lar_material_info( matclass ):
    matclass.set( 'refractive_index', lar_refractive_index[:,1], lar_refractive_index[:,0] )
    matclass.set( 'scattering_length', 90.0 )
    matclass.set( 'absorption_length', 80.9 )

# --------------------------------------------------------------------------------
# Acrylic
# This can vary based on mnufacturer, even batch to batch...especially bellow 440 nm
# We use data from RPT #1 from MiniClean report in 
# Bodmer et al., http://arxiv.org/pdf/1310.6454v2.pdf

def load_acrylic_material_info( matclass ):
    matclass.set('refractive_index', 1.49)
    matclass.absorption_length = np.array( [(375.0,29.0), (405.0, 155.0), (440.0, 261.0), (543, 3360.0), (632.0, 1650.0), (800, 1650.0)] )
    matclass.set('scattering_length', 1000.0 )

# --------------------------------------------------------------------------------
# Matclass

def load_glass_material_info( matclass ):
    # Taken from chroma.demo.optics as a starting point
    matclass.set('refractive_index', 1.49)
    matclass.absorption_length = \
        np.array([(200, 0.1e-6), (300, 0.1e-6), (330, 1000.0), (500, 2000.0), (600, 1000.0), (770, 500.0), (800, 0.1e-6)])
    matclass.set('scattering_length', 1e6)

# --------------------------------------------------------------------------------
# Vacuum

def load_vacuum_material_info( matclass ):
    # Taken from chroma.demo.optics as a starting point
    matclass.set('refractive_index', 1.0)
    matclass.set('absorption_length', 1.0e6)
    matclass.set('scattering_length', 1.0e6)

# --------------------------------------------------------------------------------
# Dummy values for non-transmissive materials

def load_stainless_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_titanium_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_bialkali_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_concrete_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_air_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_pufoam_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )

def load_dummy_material_info( matclass ):
    # using dummy values, as we never expect photons to be propagating here
    matclass.set( 'refractive_index',  1.0 )
    matclass.set( 'scattering_length', 1.0 )
    matclass.set( 'absorption_length', 1.0 )


# --------------------------------------------------------------------------------
def load_uboone_materials( c2cclass ):
    """
    c2cclass: collada_to_chroma class instance
    """
    if not isinstance(c2class, ColladaToChroma):
        raise TypeError('input to function should be instance of ColladaToChroma')
    loaders = { "LAr":load_lar_material_info,
                "Titanium":load_titanium_material_info,
                "Acrylic":load_acrylic_material_info,
                "Glass":load_glass_material_info,
                "Vacuum":load_vacuum_material_info,
                "STEEL_STAINLESS_Fe7Cr2Ni":load_vacuum_material_info,
                "PU_foam_light":load_pufoam_material_info,
                "PU_foam_dense":load_pufoam_material_info,
                "Concrete":load_concrete_material_info }
    
def clean_material_name( matname ):
    # pointer addresses attached to names
    return matname.split("0x")[0]
