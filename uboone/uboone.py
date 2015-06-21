from chroma.importgeo import UserVG4DEAGeo
from chroma.geometry import Surface
import numpy as np

uboone_wireplane = Surface( 'uboone_wireplane' )
uboone_wireplane.nplanes = 3.0
uboone_wireplane.wire_pitch = 0.3
uboone_wireplane.wire_diameter = 0.015
uboone_wireplane.transmissive = 1
uboone_wireplane.model = Surface.SURFACE_WIREPLANE


class uboone( UserVG4DEAGeo ):

    def __init__(self):
        super(uboone,self).__init__( "uboone", "dae/microboone_32pmts_nowires_cryostat.dae" )

    def surfacesdict(self):
        return {}
    
    def channeldict(self):
        return {}

    def wireplanevolumes(self):
        """Returns list of volumes with wire planes"""
        return ['volTPCPlane_PV']

    def setaswireplane(self,name,solid):
        """Gives a chroma Solid. User can set any of the traingles surface to a wireplane. Return True if did. Return False is did not."""
        if "volTPCPlane_PV" in name:
            for n,triangle in enumerate(solid.mesh.triangles):
                nxplane = 0
                for ivert in triangle:
                    if solid.mesh.vertices[ivert,0]==-1281.0:
                        nxplane += 1
                if nxplane==3:
                    print [ solid.mesh.vertices[x] for x in triangle ]
                    solid.surface[ n ] = uboone_wireplane
                    solid.unique_surfaces = np.unique( solid.surface )
            return True
        return False

