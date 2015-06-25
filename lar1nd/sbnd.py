import os,sys
from chroma.importgeo import UserVG4DEAGeo, load_hist_data
from chroma.geometry import Surface
import numpy as np

wireplane = Surface( 'wireplane' )
wireplane.nplanes = 3.0
wireplane.wire_pitch = 0.3
wireplane.wire_diameter = 0.015
wireplane.transmissive = 1
wireplane.model = Surface.SURFACE_WIREPLANE


class SBND( UserVG4DEAGeo ):

    def __init__(self):
        super(SBND,self).__init__( "SBND", "dae/lar1nd_lightguides_nowires_chroma.dae" )

    def surfacesdict(self):
        # Steel/LAr
        steel_surface = Surface("steel_surface")
        steel_surface.set('reflect_diffuse', 0.25)
        steel_surface.set('reflect_specular',0.0)
        steel_surface.set('detect',0.0)
        steel_surface.set('absorb',0.75)
        steel_surface.set('reemit',0.0)
        steel_surface.transmissive = 0
        # Titanium/LAr
        titanium_surface = Surface("titanium_surface")
        titanium_surface.set('reflect_diffuse', 0.125)
        titanium_surface.set('reflect_specular',0.125)
        titanium_surface.set('detect',0.0)
        titanium_surface.set('absorb',0.75)
        titanium_surface.set('reemit',0.0)
        titanium_surface.transmissive = 0
        # Glass/LAr  
        glass_surface = Surface("glass_surface")
        glass_surface.set('reflect_diffuse',  np.array( [ (100, 0.0), (280.0, 0.0), (350.0, 0.5), (1000.0, 0.5) ] ) )
        glass_surface.set('reflect_specular', np.array( [ (100, 0.0), (280.0, 0.0), (350.0, 0.5), (1000.0, 0.5) ] ) )
        glass_surface.set('absorb',           np.array( [ (100, 1.0), (280.0, 1.0), (350.0, 0.0), (1000.0, 0.0) ] ) )
        glass_surface.set('detect',0.0)
        glass_surface.set('reemit',0.0)
        glass_surface.transmissive = 1
        # Acrylic: detecting surface
        acrylic_surface = Surface("acrylic_surface_detector")
        acrylic_surface.set('reflect_diffuse', 0.0)
        acrylic_surface.set('reflect_specular',0.0)
        acrylic_surface.set('detect',1.0)
        acrylic_surface.set('absorb',0.0)
        acrylic_surface.set('reemit',0.0)
        acrylic_surface.transmissive = 0
        # Acrylic: wavelength shifting
        #acrylic_surface_wls = Surface("acrylic_surface_wls")
        #acrylic_surface_wls.set('reflect_diffuse', 0.0)
        #acrylic_surface_wls.set('reflect_specular',0.0)
        #acrylic_surface_wls.set('detect',0.0)
        #acrylic_surface_wls.set('absorb',0.0)
        #acrylic_surface_wls.set('reemit', load_hist_data( os.path.dirname(__file__)+"/raw_tpb_emission.dat", 350, 640 ) ) # 100% reemission. Actually, should be 120%!! Need to think about this.
        #acrylic_surface_wls.transmissive = 1
        # G10
        g10_surface = Surface("g10_surface")
        g10_surface.set('reflect_diffuse', 0.5)
        g10_surface.set('reflect_specular',0.0)
        g10_surface.set('detect',0.0)
        g10_surface.set('absorb',0.5)
        g10_surface.set('reemit',0.0)
        g10_surface.transmissive = 0
        # Black surface
        black_surface = Surface("black_surface")
        black_surface.set('reflect_diffuse',  0.0 )
        black_surface.set('reflect_specular', 0.0 )
        black_surface.set('absorb', 1.0 )
        black_surface.set('detect', 0.0)
        black_surface.set('reemit', 0.0)
        black_surface.transmissive = 0

        boundary_surfaces = { ("STEEL_STAINLESS_Fe7Cr2Ni", "LAr"):steel_surface,
                              ("STEEL_STAINLESS_Fe7Cr2Ni", "Air"):steel_surface,
                              ("Titanium", "LAr"):titanium_surface,
                              ("Acrylic", "LAr"):acrylic_surface,
                              ("G10", "LAr"):g10_surface,
                              ("Glass", "LAr"):glass_surface,
                              ("Glass", "STEEL_STAINLESS_Fe7Cr2Ni"):steel_surface,
                              ("Glass","Vacuum"):black_surface, 
                              ("PU_foam_light","Air"):black_surface,
                              ("Concrete","Air"):black_surface
                              }

        return boundary_surfaces

    def  sensitiveLogicalVolumes(self):
        return ["vollightguidedetector"]

    def sensitivePhysicalVolumes(self):
        return ["vollightguidedetector_PV"]

    def channeldict(self):
        channelmap = {"vollightguidedetector_PV":0}
        return channelmap

    def wireplanevolumes(self):
        """Returns list of volumes with wire planes"""
        return ['volTPCPlaneVert_PV']

    def setaswireplane(self,name,solid):
        """Gives a chroma Solid. User can set any of the traingles surface to a wireplane. Return True if did. Return False is did not.
        set surface for triangles on x=-2023.25 and x=2023.25 planes"""
        if name in ["volTPCPlaneVert_PV"]:
            for n,triangle in enumerate(solid.mesh.triangles):
                #print [ solid.mesh.vertices[x] for x in triangle ] # for debug
                nxplane = 0
                for ivert in triangle:
                    if solid.mesh.vertices[ivert,0]==-2023.25 or solid.mesh.vertices[ivert,0]==2023.25:
                        nxplane += 1
                # if the numbr of vertices have the correct x value, we say we have the right traingle
                if nxplane==3: 
                    print [ solid.mesh.vertices[x] for x in triangle ]
                    solid.surface[ n ] = wireplane
                    solid.unique_surfaces = np.unique( solid.surface )

            return True # modified
        return False # not modified
