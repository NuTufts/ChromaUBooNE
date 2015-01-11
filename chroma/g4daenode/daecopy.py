import collada

class DAECopy(object):
    """
    Non-Node objects, ie Effect, Material, Geometry have clearly defined places 
    to go within the `library_` elements and there is no need to place other
    elements inside those.

    The situation is not so clear with  the MaterialNode, GeometryNode, NodeNode, Node
    which live in a containment heirarcy, and for Node can contain others inside them.
    """
    def __init__(self, top, opts ):

        self.opts = opts 

        dae = collada.Collada()
        dae.assetInfo.upaxis = 'Z_UP'    # default is Y_UP

        self.dae = dae
        self.maxdepth = int(self.opts.get('maxdepth',0))   # default to no-recursion

        cpvtop = self( top )    # recursive copier
        self.cpvtop = cpvtop

        content = [cpvtop]
        if 'extra' in opts:
            cextra = collada.scene.ExtraNode( opts['extra']  )
            content.append(cextra) 


        cscene = collada.scene.Scene("DefaultScene", content )
        self.dae.scenes.append(cscene)
        self.dae.scene = cscene
        self.dae.save()             #  the save takes ~60% of total CPU time


    def load_effect( self, effect ):
        """
        :param effect: to be copied  

        Creates an effect from the xmlnode of an old one into 
        the new collada document being created
        """
        #ceffect = collada.material.Effect.load( self.dae, {},  effect.xmlnode ) 
        #ceffect = copy.copy( effect )
        ceffect = effect 
        ceffect.double_sided = True  
        self.dae.effects.append(ceffect)  # pycollada managed not adding duplicates 
        return ceffect

    def load_material( self, material ):    
        """
        :param material:

        must append the effect before can load the material that refers to it 
        """
        #cmaterial = collada.material.Material.load( self.dae, {} , material.xmlnode )
        #cmaterial = copy.copy( material )
        cmaterial = material
        cmaterial.double_sided = True   
        self.dae.materials.append(cmaterial)
        return cmaterial

    def load_geometry( self, geometry  ):
        """
        :param geometry:

        Profiling points to this consuming half the time
        attempts to use a deepcopy instead lead to 

        ::

             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 189, in deepcopy
                 y = _reconstruct(x, rv, 1, memo)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 329, in _reconstruct
                 y.append(item)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 226, in append
                 self._addindex(obj)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 152, in _addindex
                 _idx = self._index
             AttributeError: 'IndexedList' object has no attribute '_index'

        """
        #cgeometry = collada.geometry.Geometry.load( self.dae, {}, geometry.xmlnode)   # this consumes 43% of time
        #cgeometry = copy.deepcopy( geometry )
        #cgeometry = copy.copy( geometry )
        cgeometry = geometry

        self.dae.geometries.append(cgeometry)
        return cgeometry
 
    def copy_geometry_node( self, geonode ):
        """
        ::

            <instance_geometry url="#RPCStrip0x886a088">
               <bind_material>
                  <technique_common>
                      <instance_material symbol="WHITE" target="#__dd__Materials__MixGas0x8837740"/>
                 </technique_common>
               </bind_material>
            </instance_geometry>
        """
        cgeometry = self.load_geometry( geonode.geometry )
        cmaterials = []    # actually matnodes
        for matnode in geonode.materials:
            material = matnode.target
            ceffect = self.load_effect( material.effect )
            cmaterial = self.load_material( material )
            # cmaterial.double_sided = True   # geo node keeps a reference to the material, so changing this doesnt help ?
            cmatnode = collada.scene.MaterialNode( matnode.symbol, cmaterial, matnode.inputs )
            cmaterials.append(cmatnode)
        pass     
        cgeonode = collada.scene.GeometryNode( cgeometry, cmaterials )
        return cgeonode


    def faux_copy_geometry_node(self, geonode):
        """
        Not really copying, just borrowing objects owned from the original .dae into the subcopy one
        """
        self.dae.geometries.append( geonode.geometry )
        for matnode in geonode.materials:
            material = matnode.target
            self.dae.effects.append( material.effect ) 
            self.dae.materials.append( material ) 
        pass    
        return geonode

    def make_id(self, bid ):
        if self.opts.get('blender',False):
            return bid[-8:]     # just the ptr, eg xa8bffe8 as blender is long id challenged
        return bid 

    def __call__(self, vnode, depth=0, index=0 ):
        """
        The translation of the Geant4 model into Collada being used has:

        * LV nodes contain instance_geometry and 0 or more node(PV)  elements  
        * PV nodes contain matrix and instance_node (pointing to an LV node) **ONLY**
          they are merely placements within their holding LV node. 
          [observe problem of matrix being placed after instance_node for small numbers
           of nodes (possibly at leaves?)]
          
        DAENode are created by collada raw nodes traverse hitting leaves, ie
        with recursion node path  Node/NodeNode/GeometryNode or xml structure
        node/instance_node/instance_geometry 

        Thus DAENode instances correspond to::
        
             containing PV
                instance_node referenced LV
                    LV referenced geometry 

        NB this means the PV and LV are referring to different volumes, the PV
        being the containing parent PV. Because of this it is incorrect to recurse
        on the PV, its only the LV that (maybe) holds child PV that require to
        be recursed to.   Stating another way, the PV is the containing parent volume
        so its just plain wrong to recurse on it.

        NodeNode children are those of the referred to node

        :: 

             63868     <node id="__dd__Geometry__RPC__lvRPCGasgap140xa8c0268">
             63869       <instance_geometry url="#RPCGasgap140x886a0f0">
             63870         <bind_material>
             63871           <technique_common>
             63872             <instance_material symbol="WHITE" target="#__dd__Materials__Air0x8838278"/>
             63873           </technique_common>
             63874         </bind_material>
             63875       </instance_geometry>
             63876       <node id="__dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xa8c02c0">
             63877         <matrix>
             63878                 6.12303e-17 1 0 -910
             63879                 -1 6.12303e-17 0 0
             63880                  0 0 1 0
             63881                  0.0 0.0 0.0 1.0
             63882         </matrix>
             63883         <instance_node url="#__dd__Geometry__RPC__lvRPCStrip0xa8c01d8"/>
             63884       </node>


        Schematically::

              <node id="lvname1" >         
                 <instance_geometry url="#geo1" ... />
                 <node id="pvname1" >   "PV"  
                    <matrix/>
                    <instance_node url="#lvname2" >  "LV"
                        
                         metaphorically the instance node passes 
                         thru to the referred to node for the raw collada recurse
                         and makes that node element "invisble"
                         (not appearing in the nodepath used to create the DAENode)
                         hence  Node/NodeNode/GeometryNode
                                 pv     lv        geo      <<< SO PV is the parent of the LV, not the same volume ???

                           <node id="lvname2">      "LV"
                               <instance_geometry url="#geo2" />   "GEO"

                               <node id="pvname3" >
                                    <matrix/>
                                    <instance_node url="#lvname4" />
                               </node>
                               ...
                           </node> 

                    </instance_node>
                 </node>
                 <node id="pvname2" >            
                    <matrix/>
                    <instance_node url="#lvname3" />
                 </node>
                 ...
              </node>

        """
        #log.debug( "    " * depth + "[%d.%d] %s " % (depth, index, daenode))
        pvnode, lvnode, geonode = vnode.pv, vnode.lv, vnode.geo
        # NB the lvnode is a NodeNode instance


        # copy the instance_geometry node referred to by the LV  
        cnodes = []
        cgeonode = self.copy_geometry_node( geonode )
        #cgeonode = self.faux_copy_geometry_node( geonode )

        if depth == 0 and self.opts.get('skiproot', False) == True:
            log.info("skipping root geometry")
        else:
            cnodes.append(cgeonode)  

        # collect children of the referred to LV, ie the contained PV
        if not hasattr(vnode,'children') or len(vnode.children) == 0:# leaf
            pass
        else:
            if depth == self.maxdepth:  # stop the recursion when hit maxdepth
                pass
            else:
                for index, child in enumerate(vnode.children):  
                    cnode = self(child, depth + 1, index )       ####### THE RECURSIVE CALL ##########
                    cnodes.append(cnode)
            pass

        # bring together the LV copy , NB the lv a  NodeNode instance, hence the `.node` referral in the below 
        # (properties hide this referral for id/children but not for transforms : being explicit below for clarity )
        copy_ = lambda _:_.load(collada, _.xmlnode)     # create a collada object from the xmlnode representation of another
        clvnode = collada.scene.Node( self.make_id(lvnode.node.id) , children=cnodes, transforms=map(copy_,lvnode.node.transforms) ) 

        # unlike the other library_ pycollada does not prevent library_nodes/node duplication 
        if not clvnode.id in self.dae.nodes:
            self.dae.nodes.append(clvnode)
        
        # deal with the containing/parent PV, that references the above LV  
        refnode = self.dae.nodes[clvnode.id]  
        cnodenode = collada.scene.NodeNode( refnode ) 
        cpvnode = collada.scene.Node( self.make_id(pvnode.id) , children=[cnodenode], transforms=map(copy_,pvnode.transforms) ) 

        #log.debug("cpvnode %s " % tostring_(cpvnode) )
        return cpvnode


    def __str__(self):
        """
        Even after popping the top node id blender still complains::

            cannot find Object for Node with id=""
            cannot find Object for Node with id=""
            cannot find Object for Node with id=""
            cannot find Object for Node with id=""

        """
        #self.dae.save()   this stay save was almost doubling CPU time 

        if self.opts.get('blender',False):
            if self.cpvtop.xmlnode.attrib.has_key('id'):
                topid = self.cpvtop.xmlnode.attrib.pop('id')
                log.info("popped the top for blender %s " % topid )

        out = StringIO()
        writeXML(self.dae.xmlnode, out )
        return out.getvalue()
