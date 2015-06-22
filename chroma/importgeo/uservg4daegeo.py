import abc, os, sys, time
from chroma.detector import Detector
from chroma.geometry import Material, Surface
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
from chroma.loader import load_bvh
from sets import Set
import numpy as np
from itertools import imap

class UserVG4DEAGeo(Detector):
    __metaclass__ = abc.ABCMeta
    

    def __init__(self, name, daefile,
                 bvh_name="bvh_default", auto_build_bvh=True, read_bvh_cache=True, 
                 update_bvh_cache=True, cache_dir=None, bvh_method='grid', bvh_target_degree='3',
                 cuda_device=None, cl_device=None, dump_node_info=False ):
        super( UserVG4DEAGeo, self ).__init__()
        # save data members
        self.Name = name
        self.daefile = os.path.basename(daefile)
        # setup geometry cache

        # We use g4dae tools to create geometry with mesh whose triangles have materials assigned to them
        DAENode.parse( daefile, sens_mats=[] )
        self.collada2chroma = ColladaToChroma( DAENode, dump_node_info=False )
        geom = self.collada2chroma.convert_geometry_partial()
        
        # Apply wireplane alogorith
        if len(self.wireplanevolumes())>0:
            nwireplanes = 0
            for n,solid in enumerate(geom.solids):
                node = solid.node
                name = node.pv.id.split("0x")[0]
                if name in self.wireplanevolumes():
                    ok = self.setaswireplane( name, solid )
                    if ok:
                        nwireplanes += 1
            print "Number of solids with wireplane surface: ",nwireplanes
        
        # Finish Geometry
        geom = self.collada2chroma.finish_converting_geometry()
        self.mesh = geom.mesh
        self.colors = geom.colors
        self.solids = geom.solids
        self.solid_id = geom.solid_id
        self.unique_materials = geom.unique_materials
        self.material1_index = geom.material1_index
        self.material2_index = geom.material2_index
        print "Materials: ",len(self.material1_index),len(self.material2_index)

        # Next we need to go through all the triangles and make sure they have the right surfaces attached to them
        # We use the material lists to generate pairs of materials to that require a surface
        # Note material index list is over all triangles already
        # Then we make a list of surfaces
        self.user_surfaces_dict = self.surfacesdict()
        self.unique_surfaces = []
        surface_index_dict = {None:-1}
        surface_index_list = []
        for id, mats in enumerate( zip(self.material1_index, self.material2_index) ):
            surface = None
            if self.unique_materials[mats[0]].name!=self.unique_materials[mats[1]].name:
                existing_surface = geom.surface_index[ id ]
                if existing_surface!=-1:
                    print "Triangle %d already has specified surface: "%(id),geom.unique_surfaces[ existing_surface ]
                    surface = geom.unique_surfaces[ existing_surface ]
                else:
                    mat1 = self.unique_materials[mats[0]].name.split("0x")[0]
                    mat2 = self.unique_materials[mats[1]].name.split("0x")[0]
                    if (mat1,mat2) in self.user_surfaces_dict:
                        surface = self.user_surfaces_dict[ (mat1, mat2) ]
                    elif (mat2,mat1) in self.user_surfaces_dict:
                        surface = self.user_surfaces_dict[ (mat2,mat1) ]
                    else:
                        raise RuntimeError("Could not assign a surface between materials %s and %s. Must provide through sufacesdict(self) method."%(mat1,mat2))
            else:
                if geom.surface_index[id ]!=-1:
                    print "Triangle %d already has specified surface: "%(id),geom.unique_surfaces[ existing_surface ]
                    surface = geom.unique_surfaces[ existing_surface ]
                else:
                    surface = None
            if surface is not None and surface not in self.unique_surfaces:
                self.unique_surfaces.append( surface )
                surface_index_dict[ surface ] = self.unique_surfaces.index( surface )
                print "Registering new surface: [%d] %s"%(surface_index_dict[surface],surface.name)
            surface_index_list.append( surface_index_dict[ surface ] )

        print "number of unique surfaces: ",len(self.unique_surfaces)
        print self.unique_surfaces
        self.surface_index = np.array( surface_index_list, dtype=np.int32 )

        # Setup the channels
        self._setup_channels()

        # Setup the BVH
        if self.bvh is None:
            self.bvh = load_bvh(self, auto_build_bvh=auto_build_bvh,
                                read_bvh_cache=read_bvh_cache,
                                update_bvh_cache=update_bvh_cache,
                                cache_dir=cache_dir, bvh_method=bvh_method, target_degree=bvh_target_degree,
                                cuda_device=cuda_device, cl_device=cl_device)

    #@abc.abstractmethod
    #def materialsdict(self):
    #    """Return a dictionary of {'name':Material}"""
    #    raise RuntimeError("User is expected to return a dictionary of {'name':Material} where name is of type 'str' and Material is of type 'Material' (in chroma.geometry module)")

    @abc.abstractmethod
    def surfacesdict(self):
        """Return a dictionary of {('material1 name','material2 name'):Surface}
        Geant4 geometries must have materials specified.  Surfaces not so much. So we provide a dictionary to fill in the surfaces.
        """
        raise RuntimeError("User is expected to return a dictionary of {'name':Surface} where name is of type 'str' and Surface is of type 'Surface' (in chroma.geometry module)")

    @abc.abstractmethod
    def setaswireplane(self,name,solid):
        """Gives a chroma Solid and name of solid. User can set any of the traingles surface to a wireplane. Return True if did. Return False is did not."""
        return False

    @abc.abstractmethod
    def wireplanevolumes(self):
        """Returns list of volumes with wire planes"""
        return []

    @abc.abstractmethod
    def sensitiveLogicalVolumes(self):
        """Return list of sensitive logical volumes"""
        return []

    @abc.abstractmethod
    def sensitivePhysicalVolumes(self):
        """Return list of sensitive logical volumes"""
        return []

    @abc.abstractmethod
    def channeldict(self):
        """Return a dictionary of {ID:'volumename'}"""
        raise RuntimeError("User is expected to return a dictionary of {ID:'VolumeName'} where ID is an integer and VolumeName is of type 'str'")
    
    def _setup_photodetectors( self ):
        """ define the timing spread and single photoelectron distribution """
        self.set_time_dist_gaussian( 1.2, -6.0, 6.0 )
        self.set_charge_dist_gaussian( 1.0, 0.1, 0.5, 1.5 )

    def _setup_channels( self ):
        """
        fills in the solid_id_to_channel_index/solid_id_to_channel_id array look up array.
        we also make a dictionary of channel id number to node. we will need this to
        access information for various things.
        searches for nodes with a logical volume name that contains any of the strings in the list 'detector_volumes'.
        acrylic_detect is likely deprecated.
        """
        print "SETUP CHANNELS from solids list (",len(self.solids)," solids)"
        self.solid_id_to_channel_index.resize( len(self.solids) )
        self.solid_id_to_channel_index.fill(-1) # default no channels
        self.solid_id_to_channel_id.resize( len(self.solids) )
        self.solid_id_to_channel_id.fill(-1)
        self.user_channel_dict = self.channeldict()

        # prevous calls to add_solid by collada_to_chroma sized this array
        for n,solid in enumerate(self.solids):
            # loop through solids and find sensitive logical volumes
            if any( volnames in solid.node.lv.id for volnames in self.sensitiveLogicalVolumes() ):
                #print "Sensitive Detector Solid: ",solid.node," LV ID=",solid.node.lv.id
                # Need to find physical volume in order to assign channel ID
                found = False
                current = solid.node
                history = []
                while not found:
                    match = False
                    for pvname in self.sensitivePhysicalVolumes():
                        if pvname in current.pv.id:
                            match = True
                            break
                    #print "Match=",match," current=",current.pv.id," looking for ",self.sensitivePhysicalVolumes()
                    history.append( current.pv.id )
                    if match:
                        found = True
                    else:
                        current = current.parent
                    if current is None:
                        break
                if found==False:
                    raise RuntimeError("Could not find channel ID for logical volume: %s"%(str(history)))

                #print solid.node.boundgeom.matrix
                pvname = current.pv.id.split("0x")[0]
                if pvname not in self.user_channel_dict:
                    raise RuntimeError("Unassigned channel number for physical volume=%s that contains a sensitive detector volume"%(pvname))

                channel_id = self.user_channel_dict[pvname]
                    
                solid_id = n
                channel_index = len(self.channel_index_to_solid_id)
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
    
    
    
    
