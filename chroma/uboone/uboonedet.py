import os,sys
import numpy as np
from chroma.geometry import Geometry
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
import chroma.uboone.surfaces as uboonesurfaces

# Geometry class representing the MicroBooNE geometry

class ubooneDet( Geometry ):
    def __init__(self, daefile, acrylic_detect=True, acrylic_wls=False):
        if acrylic_detect and acrylic_wls:
            raise ValueError('cannot have acrylic act as both detector and wavelength shifter')
        # run constructor of base class
        if acrylic_detect:
            super( ubooneDet, self ).__init__( "Acrylic" )
        elif acrylic_wls:
            super( ubooneDet, self ).__init__( "bialkali" )
        else:
            print "Warning: no detector material specified"

        # We use g4dae tools to create geometry with mesh whose triangles have materials assigned to them
        DAENode.parse( daefile, sens_mats=[] )
        self.collada2chroma = ColladaToChroma(DAENode)
        geom = self.collada2chroma.convert_geometry()
        
        # copy member objects (maybe use copy module instead?)
        self.mesh = geom.mesh
        self.colors = geom.colors
        self.solid_id = geom.solid_id
        self.unique_materials = geom.unique_materials
        self.material1_index = geom.material1_index
        self.material2_index = geom.material2_index
        material_lookup = dict(zip(self.unique_materials, range(len(self.unique_materials))))

        # Next we need to go through all the triangles and make sure they have the right surfaces attached to them
        self.unique_surfaces = []
        surface_index_dict = {}
        surface_indices = []
        for id, mats in enumerate( (self.material1_index, self.material2_index) ):
            surface = uboonesurfaces.get_boundary_surface( self.unique_materials[mats[0]].name, self.unique_materials[mats[1]].name )
            if surface not in self.unique_surfaces:
                self.unique_surfaces.append( surface )
                surface_index_dict[ surface ] = self.unique_surfaces.index( surface )
            surface_indices.append( surface_index_dict[ surface ] )
        self.surface_index = np.array( surface_indices, dtype=np.uint32 )

        # OK, we should be ready to go
        
