import os,sys
import numpy as np
from chroma.detector import Detector
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
import chroma.uboone.surfaces as uboonesurfaces
from chroma.loader import load_bvh
from chroma.bvh.NodeDSAR import NodeDSARtree
import time
from sets import Set

# Geometry class representing the MicroBooNE geometry

class ubooneDet( Detector ):
    def __init__(self, daefile, acrylic_detect=True, acrylic_wls=False,
                 bvh_name="uboone_bvh_default", detector_volumes=[],
                 wireplane_volumes=[],
                 auto_build_bvh=True, read_bvh_cache=True, calculate_ndsar_tree=False,
                 update_bvh_cache=True, cache_dir=None, bvh_method='grid', bvh_target_degree='3',
                 cuda_device=None, cl_device=None, dump_node_info=False ):

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
        self.collada2chroma = ColladaToChroma( DAENode, dump_node_info=False )
        geom = self.collada2chroma.convert_geometry_partial()

        # Look for wireplane alogorithms
        if len(wireplane_volumes)>0:
            wireplaneset = map( lambda x: x[0], wireplane_volumes )
            for n,solid in enumerate(geom.solids):
                node = solid.node
                if node.pv.id in wireplaneset:
                    for wireplane in wireplane_volumes:
                        if wireplane[0]==node.pv.id:
                            wireplane[1]( solid )
            print "Found Solids to add wireplanes: ",len(wireplaneset)
                    
        geom = self.collada2chroma.finish_converting_geometry()

        if dump_node_info:
            for n,solid in enumerate(geom.solids):
                node = solid.node
                mesh = solid.mesh
                material2names = map( lambda x: x.name, np.unique(solid.material2) )
                material1names = map( lambda x: x.name, np.unique(solid.material1) )
                surfaces = map( lambda x: x.name, filter( lambda x: x!=None, solid.unique_surfaces ) )
                print "[SOLID %d, NODE %05d:%s,%s]"%(n, node.index,node.pv.id,node.lv.id)," NTriangles=%d OuterMat=%s InnerMat=%s Surface=%s"%(len(mesh.triangles), material2names, material1names,surfaces)
        
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
        # We use the material lists to loop over all triangles
        self.unique_surfaces = []
        surface_index_dict = {}
        surface_indices = []
        for id, mats in enumerate( zip(self.material1_index, self.material2_index) ):
            existing_surface = geom.surface_index[ id ]
            if existing_surface!=-1:
                print "Triangle %d already has specified surface: ",geom.unique_surfaces[ existing_surface ]
                #surface_indices.append( existing_surface )
                surface = geom.unique_surfaces[ existing_surface ]
            else:
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
        print "SETUP UBOONE CHANNELS from solids list: ",len(self.solids)
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

        # This tree helps with the navigation of the node tree
        #self.node_dsar_tree = NodeDSARtree( self.bvh )
        self._setup_photodetectors()

        if calculate_ndsar_tree:
            print "Calculate node DSAR tree ...",
            sndsar = time.time()
            self.node_dsar_tree = NodeDSARtree( self.bvh )
            endsar = time.time()
            print " done ",endsar-sndsar," secs."


        # OK, we should be ready to go
    def _setup_photodetectors( self ):
        self.set_time_dist_gaussian( 1.2, -6.0, 6.0 )
        self.set_charge_dist_gaussian( 1.0, 0.1, 0.5, 1.5 )
        
