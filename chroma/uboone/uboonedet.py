import os,sys
import numpy as np
from chroma.detector import Detector
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
import chroma.uboone.surfaces as uboonesurfaces
from chroma.loader import load_bvh

# Geometry class representing the MicroBooNE geometry

class ubooneDet( Detector ):
    def __init__(self, daefile, acrylic_detect=True, acrylic_wls=False,
                 bvh_name="uboone_bvh_default",
                 auto_build_bvh=True, read_bvh_cache=True,
                 update_bvh_cache=True, cache_dir=None,
                 cuda_device=None, cl_device=None ):

        if acrylic_detect and acrylic_wls:
            raise ValueError('cannot have acrylic act as both detector and wavelength shifter')
        # run constructor of base class
        if acrylic_detect:
            super( ubooneDet, self ).__init__( "Acrylic" )
        elif acrylic_wls:
            super( ubooneDet, self ).__init__( "bialkali" )
        else:
            print "Warning: no detector material specified"
        self.acrylic_detect = acrylic_detect
        self.acrylic_wls    = acrylic_wls

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
        for id, mats in enumerate( zip(self.material1_index, self.material2_index) ):
            surface = uboonesurfaces.get_boundary_surface( self.unique_materials[mats[0]].name, self.unique_materials[mats[1]].name )
            if surface not in self.unique_surfaces:
                self.unique_surfaces.append( surface )
                surface_index_dict[ surface ] = self.unique_surfaces.index( surface )
            surface_indices.append( surface_index_dict[ surface ] )
        self.surface_index = np.array( surface_indices, dtype=np.int )
        print "number of surface indicies: ",len(self.surface_index)

        if self.bvh is None:
            self.bvh = load_bvh(self, auto_build_bvh=auto_build_bvh,
                                read_bvh_cache=read_bvh_cache,
                                update_bvh_cache=update_bvh_cache,
                                cache_dir=cache_dir,
                                cuda_device=cuda_device, cl_device=cl_device)

        self._setup_photodetectors()

        # OK, we should be ready to go
    def _setup_photodetectors( self ):
        self.set_time_dist_gaussian( 1.2, -6.0, 6.0 )
        self.set_charge_dist_gaussian( 1.0, 0.1, 0.5, 1.5 )
        
