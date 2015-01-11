import collada
from collada.common import DaeObject
from tools import tag
import sys, os, logging, hashlib, copy, re
log = logging.getLogger(__name__)

class DAEExtra(DaeObject):
    """

    Non-distributed extra nodes are conventrated at 

    ::

                <library_nodes>
                   <node.../> 
                   <node.../> 
                   <node.../> 
                   <extra>
                      <opticalsurface.../>
                      <skinsurface.../>
                      <bordersurface.../>
                      <meta>
                         <bsurf.../>   # ? debug only ?
                      </meta>
                   </extra>
                </library_nodes>

    ::

        066057   <library_nodes>
        066058     <node id="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060">
        066059       <instance_geometry url="#near_top_cover_box0xc23f970">
        066060         <bind_material>
        066061           <technique_common>
        066062             <instance_material symbol="PPE" target="#__dd__Materials__PPE0xc12f008"/>
        066063           </technique_common>
        066064         </bind_material>
        066065       </instance_geometry>
        066066     </node>
        ... 
        152905     <node id="World0xc15cfc0">
        152906       <instance_geometry url="#WorldBox0xc15cf40">
        152907         <bind_material>
        152908           <technique_common>
        152909             <instance_material symbol="Vacuum" target="#__dd__Materials__Vacuum0xbf9fcc0"/>
        152910           </technique_common>
        152911         </bind_material>
        152912       </instance_geometry>
        152913       <node id="__dd__Structure__Sites__db-rock0xc15d358">
        152914         <matrix>
        152915                 -0.543174 -0.83962 0 -16520
        152916 0.83962 -0.543174 0 -802110
        152917 0 0 1 -2110
        152918 0.0 0.0 0.0 1.0
        152919 </matrix>
        152920         <instance_node url="#__dd__Geometry__Sites__lvNearSiteRock0xc030350"/>
        152921         <extra>
        152922           <meta id="/dd/Structure/Sites/db-rock0xc15d358">
        152923             <copyNo>1000</copyNo>
        152924             <ModuleName></ModuleName>
        152925           </meta>
        152926         </extra>
        152927       </node>
        152928     </node>
        152929     <extra>
        152930       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
        152931         <matrix coldim="2" name="REFLECTIVITY0xc04f6a8">1.5e-06 0 6.5e-06 0</matrix>
        152932         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04f6a8"/>
        152933         <matrix coldim="2" name="RINDEX0xc33da70">1.5e-06 0 6.5e-06 0</matrix>
        152934         <property name="RINDEX" ref="RINDEX0xc33da70"/>
        152935       </opticalsurface>
        ...
        153188       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
        153189         <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
        153190       </skinsurface> 
        ...
        153290       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
        153291         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
        153292         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
        153293       </bordersurface>
        ...
        153322       <meta>
        153323         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
        153324           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap" ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
        153325           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR" ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
        153326         </bsurf>
        ...
        153359       </meta>
        153360     </extra>
        153361   </library_nodes>




    """
    def __init__(self, opticalsurface=None, skinsurface=None, bordersurface=None, skinmap=None, bordermap=None, xmlnode=None):
        self.opticalsurface = opticalsurface
        self.skinsurface = skinsurface
        self.bordersurface = bordersurface
        self.skinmap = skinmap
        self.bordermap = bordermap

    @staticmethod 
    def load(collada, localscope, xmlnode):
        if 'surfaceproperty' not in localscope:
            localscope['surfaceproperty'] = {} 

        opticalsurface = []
        for elem in xmlnode.findall(tag("opticalsurface")):
            surf = OpticalSurface.load(collada, localscope, elem)
            localscope['surfaceproperty'][surf.name] = surf
            opticalsurface.append(surf)
        #log.debug("loaded %s opticalsurface " % len(opticalsurface))

        skinmap = {}
        skinsurface = []
        for elem in xmlnode.findall(tag("skinsurface")):
            skin = SkinSurface.load(collada, localscope, elem)
            skinsurface.append(skin)

            if skin.volumeref not in skinmap:
                skinmap[skin.volumeref] = []
            pass
            skinmap[skin.volumeref].append(skin)

        log.debug("loaded %s skinsurface " % len(skinsurface))

        bordermap = {}
        bordersurface = []
        for elem in xmlnode.findall(tag("bordersurface")):
            bord = BorderSurface.load(collada, localscope, elem)
            bordersurface.append(bord)

            if bord.physvolref1 not in bordermap:
                bordermap[bord.physvolref1] = []
            bordermap[bord.physvolref1].append(bord)

            if bord.physvolref2 not in bordermap:
                bordermap[bord.physvolref2] = []
            bordermap[bord.physvolref2].append(bord)

        log.debug("loaded %s bordersurface " % len(bordersurface))

        pass
        return DAEExtra(opticalsurface, skinsurface, bordersurface, skinmap, bordermap, xmlnode)

    def __repr__(self):
        return "%s skinsurface %s bordersurface %s opticalsurface %s skinmap %s bordermap %s " % (self.__class__.__name__, 
             len(self.skinsurface),len(self.bordersurface),len(self.opticalsurface), len(self.skinmap),len(self.bordermap)) 
 
    def dump_skinsurface(self):
        print "dump_skinsurface" 
        print "\n".join(map(lambda kv:"[%-0.2d] %s" % (kv[0],str(kv[1])),enumerate(self.skinsurface))) 
    def dump_bordersurface(self):
        print "dump_bordersurface" 
        print "\n".join(map(lambda kv:"[%-0.2d] %s" % (kv[0],str(kv[1])),enumerate(self.bordersurface))) 
    def dump_opticalsurface(self):
        print "\n".join(map(str,self.opticalsurface)) 

    def dump_skinmap(self):
        print "dump_skinmap" 
        for iv,(v,ss) in enumerate(self.skinmap.items()):
            print 
            print iv, len(ss), v
            for j, s in enumerate(ss):
                print "   ", j, s
            pass 
        print self

    def dump_bordermap(self):
        print "dump_bordermap" 
        for iv,(v,ss) in enumerate(self.bordermap.items()):
            print 
            print iv, len(ss), v
            for j, s in enumerate(ss):
                print "   ", j, s
            pass 
        print self

