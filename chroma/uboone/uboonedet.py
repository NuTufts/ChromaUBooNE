import os,sys
import numpy as np
from chroma.detector import Detector
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
import chroma.uboone.surfaces as uboonesurfaces
from chroma.loader import load_bvh
from chroma.bvh.NodeDSAR import NodeDSARtree

# Geometry class representing the MicroBooNE geometry

class ubooneDet( Detector ):
    def __init__(self, daefile, acrylic_detect=True, acrylic_wls=False,
                 bvh_name="uboone_bvh_default", detector_volumes=[],
                 auto_build_bvh=True, read_bvh_cache=True,
                 update_bvh_cache=True, cache_dir=None, bvh_method='grid', bvh_target_degree='3',
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
        
        if len( detector_volumes )==0:
            raise ValueError( "No detector volumes specified!" )

        # We use g4dae tools to create geometry with mesh whose triangles have materials assigned to them
        DAENode.parse( daefile, sens_mats=[] )
        self.collada2chroma = ColladaToChroma(DAENode)
        geom = self.collada2chroma.convert_geometry()
        
        # copy member objects (maybe use copy module instead?)
        self.mesh = geom.mesh
        self.colors = geom.colors
        self.solids = geom.solids
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
            if surface==None:
                surface_indices.append( -1 )
            else:
                if surface not in self.unique_surfaces:
                    self.unique_surfaces.append( surface )
                    surface_index_dict[ surface ] = self.unique_surfaces.index( surface )
                surface_indices.append( surface_index_dict[ surface ] )
        self.surface_index = np.array( surface_indices, dtype=np.int )
        print "number of surface indicies: ",len(self.surface_index)

        # Finally, setup channels
        print "SEUTP UBOONE CHANNELS from solids list: ",len(self.solids)
        self.solid_id_to_channel_index.resize( len(self.solids) )
        self.solid_id_to_channel_index.fill(-1) # default no channels
        self.solid_id_to_channel_id.resize( len(self.solids) )
        self.solid_id_to_channel_id.fill(-1)

        print len( self.solid_id_to_channel_index ), len(  self.solid_id_to_channel_id )
        # prevous calls to add_solid by collada_to_chroma sized this array
        for n,solid in enumerate(self.solids):
            if acrylic_detect and  any( volnames in solid.node.lv.id for volnames in detector_volumes ):
                solid_id = n
                channel_index = len(self.channel_index_to_solid_id)
                channel_id = channel_index # later can do more fancy channel indexing/labeling
                self.solid_id_to_channel_index[solid_id] = channel_index
                self.solid_id_to_channel_id[solid_id] = channel_id

                # resize channel_index arrays before filling
                self.channel_index_to_solid_id.resize(channel_index+1)
                self.channel_index_to_solid_id[channel_index] = solid_id
                self.channel_index_to_channel_id.resize(channel_index+1)
                self.channel_index_to_channel_id[channel_index] = channel_id
                
                # dictionary does not need resizing
                self.channel_id_to_channel_index[channel_id] = channel_index
        print "Number of Channels Added: ",len(self.channel_index_to_solid_id)

        if self.bvh is None:
            self.bvh = load_bvh(self, auto_build_bvh=auto_build_bvh,
                                read_bvh_cache=read_bvh_cache,
                                update_bvh_cache=update_bvh_cache,
                                cache_dir=cache_dir, bvh_method=bvh_method, target_degree=bvh_target_degree,
                                cuda_device=cuda_device, cl_device=cl_device)
        self.node_dsar_tree = NodeDSARtree( self.bvh )
        self._setup_photodetectors()

        # OK, we should be ready to go
    def _setup_photodetectors( self ):
        self.set_time_dist_gaussian( 1.2, -6.0, 6.0 )
        self.set_charge_dist_gaussian( 1.0, 0.1, 0.5, 1.5 )
        
