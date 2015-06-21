import abc, os, sys, time
from chroma.detector import Detector
from chroma.geometry import Material, Surface
from chroma.g4daenode.collada_to_chroma import ColladaToChroma
from chroma.g4daenode.g4daenode import DAENode
from chroma.loader import load_bvh
from sets import Set
import numpy as np

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
        #material_lookup = dict(zip(self.unique_materials, range(len(self.unique_materials))))

        # Next we need to go through all the triangles and make sure they have the right surfaces attached to them
        # We use the material lists to loop over all triangles
        self.unique_surfaces = []
        surface_index_dict = {}
        surface_indices = []
        for id, mats in enumerate( zip(self.material1_index, self.material2_index) ):
            existing_surface = geom.surface_index[ id ]
            if existing_surface!=-1:
                print "Triangle %d already has specified surface: "%(id),geom.unique_surfaces[ existing_surface ]
                surface = geom.unique_surfaces[ existing_surface ]
            else:
                #surface = uboonesurfaces.get_boundary_surface( self.unique_materials[mats[0]].name, self.unique_materials[mats[1]].name )
                surface = None
            if surface==None:
                surface_indices.append( -1 )
            else:
                if surface not in self.unique_surfaces:
                    self.unique_surfaces.append( surface )
                    surface_index_dict[ surface ] = self.unique_surfaces.index( surface )
                surface_indices.append( surface_index_dict[ surface ] )
        self.surface_index = np.array( surface_indices, dtype=np.int )
        print "number of surface indicies: ",len(self.surface_index)


    #@abc.abstractmethod
    #def materialsdict(self):
    #    """Return a dictionary of {'name':Material}"""
    #    raise RuntimeError("User is expected to return a dictionary of {'name':Material} where name is of type 'str' and Material is of type 'Material' (in chroma.geometry module)")

    @abc.abstractmethod
    def surfacesdict(self):
        """Return a dictionary of {'name':Surface}"""
        raise RuntimeError("User is expected to return a dictionary of {'name':Surface} where name is of type 'str' and Surface is of type 'Surface' (in chroma.geometry module)")

    @abc.abstractmethod
    def channeldict(self):
        """Return a dictionary of {ID:'volumename'}"""
        raise RuntimeError("User is expected to return a dictionary of {ID:'VolumeName'} where ID is an integer and VolumeName is of type 'str'")
    
    @abc.abstractmethod
    def setaswireplane(self,name,solid):
        """Gives a chroma Solid and name of solid. User can set any of the traingles surface to a wireplane. Return True if did. Return False is did not."""
        return False

    @abc.abstractmethod
    def wireplanevolumes(self):
        """Returns list of volumes with wire planes"""
        return []
    
    
    
    
    
