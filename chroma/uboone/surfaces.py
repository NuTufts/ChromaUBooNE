import os,sys
import numpy as np
from chroma.geometry import Surface
import chroma.uboone.materials as materials
import chroma.uboone.load_ub_spectral_data as datatools

# This module contains the data needed to describe surface processes
# Meant for MicroBooNE geometry
# whatever surface prob remains below 1.0 is transmission prob

# ----------------------------------------------------
# Steel-LAr surface (cryostat)
# We use simple mixtures of diffuse and specular reflection.
# Hard to know what it should be for 128 nm light
# Using values from larproperties.fcl
def make_steel_surface():
    steel_surface = Surface("steel_surface")
    steel_surface.set('reflect_diffuse', 0.125)
    steel_surface.set('reflect_specular',0.125)
    steel_surface.set('detect',0.0)
    steel_surface.set('absorb',0.75)
    steel_surface.set('reemit',0.0)
    steel_surface.transmissive = 0
    # eta and kappa not set
    return steel_surface

# ----------------------------------------------------
# Titanium-LAr surface (wires)
# We use simple mixtures of diffuse and specular reflection.
# Hard to know what it should be for 128 nm light
def make_titanium_surface():
    titanium_surface = Surface("titanium_surface")
    titanium_surface.set('reflect_diffuse', 0.0)
    titanium_surface.set('reflect_specular',1.0)
    titanium_surface.set('detect',0.0)
    titanium_surface.set('absorb',0.0)
    titanium_surface.set('reemit',0.0)
    titanium_surface.transmissive = 0
    return titanium_surface

# ----------------------------------------------------
# Glass (PMT surface)
# We use simple mixtures of diffuse and specular reflection.
# Hard to know what it should be for 128 nm light
def make_glass_surface():
    # mostly fake numbers based on http://www.shimadzu.com/an/uv/support/uv/ap/measuring_solar2.html
    # glass mostly absorbs until 300 nm, then gradually transmits.  reflected light assumed diffuse
    glass_surface = Surface("glass_surface")
    glass_surface.set('reflect_diffuse',  np.array( [ (100, 0.0), (280.0, 0.0), (350.0, 0.1), (1000.0, 0.1) ] ) )
    glass_surface.set('reflect_specular', np.array( [ (100, 0.0), (280.0, 0.0), (350.0, 0.0), (1000.0, 0.0) ] ) )
    glass_surface.set('absorb',           np.array( [ (100, 1.0), (280.0, 1.0), (350.0, 0.0), (1000.0, 0.0) ] ) )
    glass_surface.set('detect',0.0)
    glass_surface.set('reemit',0.0)
    glass_surface.transmissive = 1
    return glass_surface

# ----------------------------------------------------
# Acrylic: Wavelength-shifting plates
# We use simple mixtures of diffuse and specular reflection.
# Hard to know what it should be for 128 nm light
def make_acrylic_surface_detectmode():
    """
    this version of acrylic surface detects photons and acts as our counting unit.
    """
    acrylic_surface = Surface("acrylic_surface_detector")
    acrylic_surface.set('reflect_diffuse', 0.0)
    acrylic_surface.set('reflect_specular',0.0)
    acrylic_surface.set('detect',1.0)
    acrylic_surface.set('absorb',0.0)
    acrylic_surface.set('reemit',0.0)
    acrylic_surface.transmissive = 0
    return acrylic_surface

def make_acrylic_surface_wlsmode():
    """
    this version of acrylic surface wavelength shifts reemitted light
    """
    acrylic_surface = Surface("acrylic_surface_detector")
    acrylic_surface.set('reflect_diffuse', 0.0)
    acrylic_surface.set('reflect_specular',0.0)
    acrylic_surface.set('detect',0.0)
    acrylic_surface.set('absorb',0.0)
    acrylic_surface.set('reemit', datatools.load_hist_data( os.path.dirname(__file__)+"/raw_tpb_emission.dat", 350, 640 ) ) # 100% reemission. Actually, should be 120%!! Need to think about this.
    acrylic_surface.transmissive = 1
    return acrylic_surface

# ----------------------------------------------------
# G10: Guessing!
def make_G10_surface():
    g10_surface = Surface("g10_surface")
    g10_surface.set('reflect_diffuse', 0.5)
    g10_surface.set('reflect_specular',0.0)
    g10_surface.set('detect',0.0)
    g10_surface.set('absorb',0.5)
    g10_surface.set('reemit',0.0)
    g10_surface.transmissive = 0
    return g10_surface


# ----------------------------------------------------
# absorbing surface
def make_absorbing_surface(name="absorbing_surface"):
    # mostly fake numbers based on http://www.shimadzu.com/an/uv/support/uv/ap/measuring_solar2.html
    # glass mostly absorbs until 300 nm, then gradually transmits.  reflected light assumed diffuse
    black_surface = Surface(name)
    black_surface.set('reflect_diffuse',  0.0 )
    black_surface.set('reflect_specular', 0.0 )
    black_surface.set('absorb', 1.0 )
    black_surface.set('detect', 0.0)
    black_surface.set('reemit', 0.0)
    black_surface.transmissive = 0
    return black_surface


# ----------------------------------------------------
# Boundary Surface Definitions

boundary_surfaces = { ("STEEL_STAINLESS_Fe7Cr2Ni", "LAr"): make_steel_surface(),
                      ("Titanium", "LAr"): make_titanium_surface(),
                      ("Acrylic", "LAr"):make_acrylic_surface_detectmode(),
                      ("G10", "LAr"):make_G10_surface(),
                      ("Glass", "LAr"):make_glass_surface(),
                      ("Glass", "STEEL_STAINLESS_Fe7Cr2Ni"):make_steel_surface(),
                      ("Glass","Vacuum"):make_absorbing_surface("pmt_inner_surface"),
                      ("Absorb"):make_absorbing_surface() }
                      
# if undefined, passes black surface
def get_boundary_surface( matname1, matname2 ):
    m1 = materials.clean_material_name( matname1 )
    m2 = materials.clean_material_name( matname2 )
    if m1==m2:
        return None
    if m1 not in materials.materialnames or m2 not in materials.materialnames:
        missing = []
        if m1 not in materials.materialnames:
            missing.append(m1)
        if m2 not in materials.materialnames:
            missing.append(m2)
        raise ValueError( "bounary between materials not defined. Missing: %s"%(missing) )
    if (m1,m2) in boundary_surfaces:
        return boundary_surfaces[(m1,m2)]
    elif (m2,m1) in boundary_surfaces:
        return boundary_surfaces[(m2,m1)]
    else:
        return boundary_surfaces["Absorb"]

def get_uboone_surfaces():
    surface_list = []
    keys = boundary_surfaces.keys()
    keys.sort()
    for k in keys:
        surface_list.append( boundary_surfaces[k] )
    return surface_list
