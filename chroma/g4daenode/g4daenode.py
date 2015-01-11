#!/usr/bin/env python
import numpy as np
import collada
from collada.xmlutil import etree as ET
from collada.xmlutil import writeXML, COLLADA_NS, E
from collada.common import DaeObject
from daeclasses import DAEExtra
from geomclasses import MaterialProperties
from tools import tag

# disable saves, which update the xmlnode, as the preexisting xmlnode for 
# the basis objects are being copied anyhow
collada.geometry.Geometry.save = lambda _:_
collada.material.Material.save = lambda _:_

tostring_ = lambda _:ET.tostring(getattr(_,'xmlnode'))

import sys, os, logging, hashlib, copy, re
log = logging.getLogger(__name__)
from StringIO import StringIO

# globals for debug 
gtop = None
gsub = None


class DAENode(object):
    secs = {}
    registry = []
    lookup = {}
    idlookup = {}
    pvlookup = {}
    lvlookup = {}
    ids = set()
    created = 0
    root = None
    rawcount = 0
    verbosity = 1   # 0:almost no output, 1:one liners, 2:several lines, 3:extreme  
    argptn = re.compile("^(\S*)___(\d*)")   

    @classmethod
    def summary(cls):
        log.info("registry %s " % len(cls.registry) )
        log.info("lookup %s " % len(cls.lookup) )
        log.info("idlookup %s " % len(cls.idlookup) )
        log.info("ids %s " % len(cls.ids) )
        log.info("rawcount %s " % cls.rawcount )
        log.info("created %s " % cls.created )
        log.info("root %s " % cls.root )

    @classmethod
    def dump(cls):
        cls.extra.dump_skinsurface()
        cls.extra.dump_skinmap()
        cls.extra.dump_bordersurface()
        cls.extra.dump_bordermap()
        cls.dump_extra_material()

    @classmethod
    def idmap_parse( cls, path ):
        """
        Read g4_00.idmap file that maps between volume index 
        and sensdet identity, ie the PmtId
        """
        if os.path.exists(path):
            from idmap import IDMap
            idmap = IDMap(path)
            log.info("idmap exists %s entries %s " % (path, len(idmap)))
        else:
            log.warn("no idmap found at %s " % path )
            idmap = None
        pass
        return idmap 

    @classmethod
    def idmaplink(cls, idmap ):
        if idmap is None:
            log.warn("skip idmaplink ")
            return 
        pass
        log.info("linking DAENode with idmap %s identifiers " % len(idmap)) 
        assert len(cls.registry) == len(idmap), ( len(cls.registry), len(idmap))
        for index, node in enumerate(cls.registry):
            node.channel_id = idmap[index]
            node.g4tra = idmap.tra[index]
            node.g4rot = idmap.rot[index]
            #if index % 100 == 0:
            #    print index, node.channel_id, node, node.__class__

    channel = property(lambda self:getattr(self, 'channel_id', 0)) 

    @classmethod
    def add_sensitive_surfaces(cls, matid='__dd__Materials__Bialkali', qeprop='EFFICIENCY'):
        """
        Chroma expects sensitive detectors to have an Optical Surface 
        with channel_id associated.  
        Whereas Geant4 just has sensitive LV.

        This attempts to bridge from Geant4 to Chroma model 
        by creation of "fake" chroma skinsurfaces.  

        Effectively sensitive materials are translated 
        into sensitive surfaces 

        :: 

            In [57]: DAENode.orig.materials['__dd__Materials__Bialkali0xc2f2428'].extra
            Out[57]: <MaterialProperties keys=['RINDEX', 'EFFICIENCY', 'ABSLENGTH'] >


        #. Different efficiency for different cathodes ?

        """
        log.info("add_sensitive_surfaces matid %s qeprop %s " % (matid, qeprop))
        sensitive_material = cls.materialsearch(matid)
        if sensitive_material is None:
            raise ValueError('Did not find sensitive material with id=\"%s\"'%(matid))
        efficiency = sensitive_material.extra.properties[qeprop] 
        if efficiency is None:
            raise ValueError('Did not find QE Efficiency property (%s) in DAE file for material with id=\"%s\"'%(qeprop,matid))

        cls.sensitize(matid=matid)

        # follow convention used in G4DAE exports of using same names for 
        # the SkinSurface and the OpticalSurface it refers too
 
        for node in cls.sensitive_nodes:
            ssid = cls.sensitive_surface_id(node)
            volumeref = node.lv.id

            surf = OpticalSurface.sensitive(name=ssid, properties={qeprop:efficiency})
            cls.add_extra_opticalsurface(surf)

            skin = SkinSurface.sensitive(name=ssid, surfaceproperty=surf, volumeref=volumeref )
            cls.add_extra_skinsurface(skin)
        pass



    @classmethod
    def sensitive_surface_id(cls, node, suffix="Surface"):
        """
        Using name including the channel_id results in too many surfaces
        and causes cuda launch failures::

            ssid = g4dtools.remove_address(node.lv.id) + "0x%7x" % channel_id + suffix  

        """
        channel_id = getattr(node,'channel_id',0)
        if channel_id == 0:
            ssid = None 
        else:
            ssid = remove_address(node.lv.id) + suffix
        return ssid 


    @classmethod
    def sensitize(cls,matid="__dd__Materials__Bialkali"):
        """
        :param matid: material id prefix that confers sensitivity, ie cathode material 

        ::

            In [8]: headon = DAENode.pvsearch('__dd__Geometry__PMT__lvHeadonPmtVacuum--pvHeadonPmtCathode')

            In [9]: hemi = DAENode.pvsearch('__dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode')

            In [10]: len(headon)
            Out[10]: 12

            In [11]: len(hemi)
            Out[11]: 672

            In [12]: map(lambda node:"0x%7x" % node.channel_id, hemi )
            Out[12]: 
            ['0x1010101',
             '0x1010102',
             '0x1010103',
             '0x1010104',
             '0x1010105',
             '0x1010106',
            ...

            In [14]: sorted(DAENode.sensitive_nodes) == sorted(headon+hemi)
            Out[14]: True

            In [16]: c = DAENode.sensitive_nodes[0]

            In [18]: c.lv
            Out[18]: <NodeNode node=__dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0>

            In [19]: c.lv.id
            Out[19]: '__dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0'

            In [20]: c.pv.id
            Out[20]: '__dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380'

            In [21]: c.id
            Out[21]: '__dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.0'

        """
        cls.channel_count = 0 
        cls.channel_ids = set()
        cls.sensitive_nodes = []

        def visit(node):
            channel_id = getattr(node,'channel_id',0)
            if channel_id > 0 and node.matid.startswith(matid): 
                #print "%6d %8d 0x%7x %s " % ( node.index, channel_id, channel_id, node.id )
                cls.channel_ids.add(channel_id)
                cls.channel_count += 1  
                cls.sensitive_nodes.append(node)
            pass
        pass
        cls.vwalk(visit)
        log.info("sensitize %s nodes with matid %s and channel_id > 0, uniques %s " % (cls.channel_count, matid, len(cls.channel_ids)) )


    @classmethod
    def add_extra_skinsurface(cls, skin ):
        """
        Used to add extra surface, not present in the COLLADA/DAE document.
        eg for sensitive detector surfaces needed for matching
        from geant4 to chroma model 

        Attempt to have a different surface per PMT fails, with 


        :param skin: *SkinSurface* instance
        """

        if not skin in cls.extra.skinsurface:
           cls.extra.skinsurface.append(skin)

        ssid = skin.name

        if not ssid in cls.extra.skinmap:
            cls.extra.skinmap[ssid] = []
        pass

        cls.extra.skinmap[ssid].append(skin)
        log.debug("+ssid %s " % (ssid)) 


    @classmethod
    def add_extra_opticalsurface(cls, surf ):
        if not surf in cls.extra.opticalsurface:
           cls.extra.opticalsurface.append(surf)


    @classmethod
    def parse( cls, path, sens_mats=["__dd__Materials__Bialkali0xc2f2428"] ):
        """
        :param path: to collada file

        #. `collada.Collada` parses the .dae 
        #. a list of bound geometry is obtained from `dae.scene.objects`
        #. `DAENode.recurse` traverses the raw pycollada node tree, creating 
           an easier to navigate DAENode heirarchy which has one DAENode per bound geometry  
        #. cross reference between the bound geometry list and the DAENode tree

        """
        path = os.path.expandvars(path)
        log.debug("DAENode.parse pycollada parse %s " % path )

        base, ext = os.path.splitext(path)
        idmap = cls.idmap_parse( base + ".idmap" ) # path with .idmap instead of .dae

        #import IPython
        #IPython.embed()


        dae = collada.Collada(path)
        log.debug("pycollada parse completed ")
        boundgeom = list(dae.scene.objects('geometry'))
        top = dae.scene.nodes[0]
        log.debug("pycollada binding completed, found %s  " % len(boundgeom))
        log.debug("create DAENode heirarchy ")
        cls.orig = dae
        cls.recurse(top)
        #cls.summary()
        cls.indexlink( boundgeom )
        cls.idmaplink( idmap )

        cls.parse_extra_surface( dae )
        cls.parse_extra_material( dae )
        for matid in sens_mats:
            cls.add_sensitive_surfaces(matid=matid)


    @classmethod
    def parse_extra_surface( cls, dae ):
        """
        """
        log.debug("collecting opticalsurface/boundarysurface/skinsurface info from library_nodes/extra/")
        library_nodes = dae.xmlnode.find(".//"+tag("library_nodes"))
        extra = library_nodes.find(tag("extra"))
        assert extra is not None
        cls.extra = DAEExtra.load(collada, {}, extra) 

    @classmethod
    def parse_extra_material( cls, dae ):
        log.debug("collecting extra material properties from library_materials/material/extra ")
        nextra = 0 
        for material in dae.materials:
            extra = material.xmlnode.find(tag("extra"))
            if extra is None:
                material.extra = None
            else:
                nextra += 1
                material.extra = MaterialProperties.load(collada, {}, extra)
            pass 
        log.debug("loaded %s extra elements with MaterialProperties " % nextra )             

    @classmethod
    def dump_extra_material( cls ):
        log.info("dump_extra_material")
        for material in cls.orig.materials:
            print material
            if material.extra is not None:
                print material.extra 



 

    @classmethod
    def recurse(cls, node , ancestors=[], extras=[] ):
        """
        This recursively visits 12230*3 = 36690 Nodes.  
        The below pattern of triplets of node types is followed precisely, due to 
        the node/instance_node/instance_geometry layout adopted for the dae file.

        The triplets are collected into DAENode on every 3rd leaf node.

        ::

            1 0 <MNode top transforms=0, children=1>
            2 1 <NodeNode node=World0xb50dfb8>
            3 2 <MGeometryNode geometry=WorldBox0xb342f60>

            4 2 <MNode __dd__Structure__Sites__db-rock0xb50e0f8 transforms=1, children=1>
            5 3 <NodeNode node=__dd__Geometry__Sites__lvNearSiteRock0xb50de78>
            6 4 <MGeometryNode geometry=near_rock0xb342e30>

            7 4 <MNode __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0 transforms=1, children=1>
            8 5 <NodeNode node=__dd__Geometry__Sites__lvNearHallTop0xb356a70>
            9 6 <MGeometryNode geometry=near_hall_top_dwarf0x92eee48>

            10 6 <MNode __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xb356790 transforms=1, children=1>
            11 7 <NodeNode node=__dd__Geometry__PoolDetails__lvNearTopCover0xb342fe8>
            12 8 <MGeometryNode geometry=near_top_cover_box0x92ecf48>

        """
        cls.rawcount += 1

        children = []
        xtras = []  
        if hasattr(node, 'children'):
            for c in node.children:
                if isinstance(c, collada.scene.ExtraNode):
                    xtras.append(c.xmlnode)
                else:
                    children.append(c) 
                pass
            pass     
        pass
        #log.info("node: %s " % node )
        #log.info("xtras: %s " % xtras )
        #log.info("extras: %s " % extras )

        if len(children) == 0: #leaf formation, gets full ancestry to go on
            cls.make( ancestors + [node], extras + xtras )
        else:
            for child in children:
                cls.recurse(child, ancestors = ancestors + [node] , extras = extras + xtras )

    @classmethod
    def indexget(cls, index):
        return cls.registry[index]

    @classmethod
    def indexgets(cls, indices):
        return [cls.registry[index] for index in indices]

    @classmethod
    def idget(cls, id):
        return cls.idlookup.get(id, None)

    @classmethod
    def get(cls, arg ):
        indices = cls.interpret_ids(arg)
        index = indices[0]
        node = cls.registry[index]
        log.info("arg %s => indices %s => node %s " % ( arg, indices, node ))
        return node

    @classmethod
    def getall(cls, nodespec, path=None):
        if not path is None:
            cls.init(path)
        pass
        indices = cls.interpret_ids(nodespec)
        return [cls.registry[index] for index in indices]

    @classmethod
    def interpret_arg(cls, arg):
        """
        Interpret arguments like:

        #. 0
        #. __dd__some__path.0
        #. __dd__some__path.0___0
        #. 0___0
        #. top.0___0

        Where the triple underscore ___\d* signified the maxdepth to recurse.
        """
        match = cls.argptn.match(arg)
        if match:
            arg, maxdepth = match.groups()
        else:
            maxdepth = 0  # default to zero
        return arg, maxdepth

    @classmethod
    def interpret_ids(cls, arg_, dedupe=True):
        """
        Interprets an "," delimited string like 0:10,400:410,300,40,top.0
        into a list of integer DAENode indices. Where each element
        can use one of the below forms. 

        Listwise::

              3153:3160
              3153:        # means til the end of the registry 

        Shortform treewise, only allows setting mindepth to 0,1::

              3153-     # mindepth 0
              3153+     # mindepth 1 

        Longform treewise, allows mindepth/maxdepth spec::

              3153_1.5  # mindepth 1, maxdepth 5

        Intwise::

              0

        Identifier::

              top.0

        """
        indices = []
        for arg in arg_.split(","):
            prelast, last = arg[:-1], arg[-1]

            listwise = ":" in arg
            treewise_short = last in ("-","+") 
            treewise_long  = arg.count("_") == 1
            intwise = arg.isdigit()

            assert len(filter(None,[listwise,treewise_short,treewise_long,intwise]))<=1, "mixing forms not allowed" 

            if treewise_short or treewise_long:
                pass
                if treewise_short:
                    pass
                    baseindex= prelast
                    mindepth = 0 if last == "-" else 1
                    maxdepth = 100
                    pass
                elif treewise_long:
                    pass
                    elem = arg.split("_")
                    assert len(elem) == 2
                    baseindex = elem[0]
                    mindepth, maxdepth = map(int,elem[1].split(".")) 
                else:
                    assert 0
                treewise_indices = cls.progeny_indices(baseindex, mindepth=mindepth, maxdepth=maxdepth )
                indices.extend(treewise_indices) 

            elif listwise:
                pass
                if last == ":":
                    arg = "%s:%s" % (prelast, len(cls.registry)-1) 
                pass
                listwise_indices=range(*map(int,arg.split(":")))
                indices.extend(listwise_indices)
                pass

            elif intwise:

                indices.append(int(arg))

            else:
                # node lookup by identifier like top.0
                if "___" in arg:
                    arg = arg.split("___")[0]   # get rid of the maxdepth indicator eg "___0"
                node = cls.idlookup.get(arg,None)
                if node:
                    indices.append(node.index)
                else:
                    log.warn("failed to lookup DAENode for arg %s " % arg)
                pass
            pass
        return list(set(indices)) if dedupe else indices


    @classmethod
    def init(cls, path=None ):
        if path is None:
            path = os.environ['DAE_NAME']
         
        if len(cls.registry) == 0:
            cls.parse(path)


    @classmethod
    def find_uid(cls, bid, decodeNCName=False):
        """
        :param bid: basis ID
        :param decodeNCName: more convenient not to decode for easy URL/cmdline  arg passing without escaping 

        Find a unique id for the emerging DAENode
        """
        if decodeNCName:
            bid = bid.replace("__","/").replace("--","#").replace("..",":")
        uid = None
        count = 0 
        while uid is None or uid in cls.ids: 
            uid = "%s.%s" % (bid,count)
            count += 1
        pass
        cls.ids.add(uid)
        return uid 

    @classmethod
    def make(cls, nodepath, extras ):
        """
        Creates `DAENode` instances and positions them within the volume tree
        by setting the `parent` and `children` attributes.

        A digest keyed lookup gives fast access to node parents,
        the digest represents a path through the tree of nodes.
        """
        node = cls(nodepath, extras )
        if node.index == 0:
            cls.root = node

        cls.registry.append(node)
        cls.idlookup[node.id] = node   

        # a list of nodes for each pv.id, need for a list is not so obvious, maybe GDML PV identity bug ?
        pvid = node.pv.id
        if pvid not in cls.pvlookup:
            cls.pvlookup[pvid] = []
        cls.pvlookup[pvid].append(node) 

        # list of nodes for each lv.id, need for a list is obvious
        lvid = node.lv.id
        if lvid not in cls.lvlookup:
            cls.lvlookup[lvid] = []
        cls.lvlookup[lvid].append(node) 

   
        cls.lookup[node.digest] = node   
        cls.created += 1

        parent = cls.lookup.get(node.parent_digest)
        node.parent = parent
        if parent is None:
            if node.id == "top.0":
                pass
            elif node.id == "top.1":
                log.info("root node name %s indicates have parsed twice " % node.id )
            else:
                log.fatal("failed to find parent for %s (failure expected only for root node)" % node )
                assert 0
        else:
            parent.children.append(node)  

        if cls.created % 1000 == 0:
            log.debug("make %s : [%s] %s " % ( cls.created, id(node), node ))
        return node

    @classmethod
    def pvfind(cls, pvid ):
        return cls.pvlookup.get(pvid,[])

    @classmethod
    def lvfind(cls, lvid ):
        return cls.lvlookup.get(lvid,[])

    @classmethod
    def materialsearch(cls, matid ):
        """
        """
        ixpo = matid.find("0x")
        if ixpo > -1:
            mats = filter(lambda mat:mat.id == matid,cls.orig.materials )
        else:
            mats = filter(lambda mat:mat.id[:-9] == matid,cls.orig.materials )
        pass
        if len(mats) > 1:
            log.warn("ambiguous matid %s " % matid )
            return None
        elif len(mats) == 1:
            return mats[0]
        else:
            return None


    @classmethod
    def lvsearch(cls, lvkey ):
        """
        Search for node by key, when the lvkey ends with '0x.....' 
        a precise id match is made, otherwise the match 
        is made excluding the address 0x...
        """
        ixpo = lvkey.find("0x")
        if ixpo > -1:
            keys = filter(lambda k:k == lvkey,cls.lvlookup.keys())
        else:
            keys = filter(lambda k:k[:-9] == lvkey,cls.lvlookup.keys())
        pass
        if len(keys) > 1:
           log.warn("ambiguous lvkey %s " % lvkey )
           return None
        elif len(keys) == 1:
            return cls.lvlookup[keys[0]]
        else:
            return None  

    @classmethod
    def pvsearch(cls, pvkey ):
        """
        Search for node by key, when the pvkey ends with '0x.....' 
        a precise id match is made, otherwise the match 
        is made excluding the address 0x...

        ::

            In [30]: DAENode.pvsearch('__dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode')
            Out[30]: 
            [  __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.0             __dd__Materials__Bialkali0xc2f2428 ,
               __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.1             __dd__Materials__Bialkali0xc2f2428 ,
               __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.2             __dd__Materials__Bialkali0xc2f2428 ,
               __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.3             __dd__Materials__Bialkali0xc2f2428 ,
               __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380.4             __dd__Materials__Bialkali0xc2f2428 ,

        """
        if pvkey[-9:-7] == '0x':
            keys = filter(lambda k:k == pvkey,cls.pvlookup.keys())
        else:
            keys = filter(lambda k:k[:-9] == pvkey,cls.pvlookup.keys())
        pass

        if len(keys) > 1:
           log.warn("ambiguous pvkey %s : %s " % (pvkey, repr(keys)) )
           return None
        elif len(keys) == 1:
            return cls.pvlookup[keys[0]]
        else:
            return None  


    @classmethod
    def walk(cls, node=None, depth=0):
        if node is None:
            cls.wcount = 0
            node=cls.root

        cls.wcount += 1 
        if cls.wcount % 100 == 0:
            log.info("walk %s %s %s " % ( cls.wcount, vdepth, node ))
            if hasattr(node,'boundgeom'):
                print node.boundgeom

        for subnode in node.children:
            cls.walk(subnode, depth+1)


    @classmethod
    def vwalk(cls, visit_=lambda node:None, node=None, depth=0):
        if node is None:
            node=cls.root
        visit_(node) 
        for subnode in node.children:
            cls.vwalk(visit_=visit_, node=subnode, depth=depth+1)

    @classmethod
    def dwalk(cls, visit_=lambda node:None, node=None, depth=0):
        if node is None:
            node=cls.root
        visit_(node, depth) 
        for subnode in node.children:
            cls.dwalk(visit_=visit_, node=subnode, depth=depth+1)


    @classmethod
    def vwalks(cls, visits=[], node=None, depth=0):
        if node is None:
            node=cls.root
        for visit_ in visits:
            visit_(node) 
        for subnode in node.children:
            cls.vwalks(visits=visits, node=subnode, depth=depth+1)

    @classmethod
    def progeny_nodes(cls, baseindex=None, mindepth=0, maxdepth=100):
        """
        :param baseindex: of base node of interest within the tree
        :param mindepth:  0 includes basenode, 1 will start from children of basenode
        :param maxdepth:  0 includes basenode, 1 will start from children of basenode
        :return: all nodes in the tree below and including the base node
        """ 
        nodes = []
        basenode = cls.root if baseindex is None else cls.get(str(baseindex))
        pass
        def visit_(node, depth):
            if mindepth <= depth <= maxdepth:
                nodes.append(node)
            pass
        pass
        cls.dwalk(visit_=visit_, node=basenode )  # recursive walk 
        return nodes

    @classmethod
    def progeny_indices(cls, baseindex=None, mindepth=0, maxdepth=100):
        """
        :param baseindex: of base node of interest within the tree
        :param mindepth:  0 includes basenode, 1 will start from children of basenode
        :param maxdepth:  0 includes basenode, 1 will start from children of basenode
        :return: all nodes in the tree below and including the base node
        """ 
        indices = []
        basenode = cls.root if baseindex is None else cls.get(str(baseindex))
        pass
        def visit_(node, depth):
            if mindepth <= depth <= maxdepth:
                indices.append(node.index)
            pass
        pass
        cls.dwalk(visit_=visit_, node=basenode )  # recursive walk 
        return indices




    @classmethod
    def md5digest(cls, nodepath ):
        """
        Use of id means that will change from run to run. 
        """
        dig = ",".join(map(lambda _:str(id(_)),nodepath))
        dig = hashlib.md5(dig).hexdigest() 
        return dig


    @classmethod
    def indexlink(cls, boundgeom ):
        """
        index linked cross referencing

        For this to be correct the ordering that pycollada comes
        up with for the boundgeom must match the DAENode ordering

        The geometry id comparison performed is a necessary condition, 
        but it does not imply correctness of the cross referencing due
        to a lot of id recycling.
        """
        log.info("index linking DAENode with boundgeom %s volumes " % len(boundgeom)) 
        assert len(cls.registry) == len(boundgeom), ( len(cls.registry), len(boundgeom))
        for vn,bg in zip(cls.registry,boundgeom):
            vn.boundgeom = bg
            vn.matdict = vn.get_matdict()
            bg.daenode = vn
            assert vn.geo.geometry.id == bg.original.id   
        log.debug("index linking completed")    


    @classmethod
    def _metadata(cls, extras):
        """
        :param extras: list of xmlnode of extra elements

        Interpret extra/meta/* text elements, converting into a dict 
        """
        d = {}
        extra = None
        if len(extras)>0:
            extra = extras[-1]
        if not extra is None:
            meta = extra.find("{%s}meta" % COLLADA_NS ) 
            for elem in meta.findall("*"):
                tag = elem.tag[len(COLLADA_NS)+2:]
                d[tag] = elem.text 
        return d



    def ancestors(self,andself=False):
        """
        ::

            In [35]: print VNode.registry[1000]
            VNode(17,19)[1000,__dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xb3445c8.46]

            In [36]: for _ in VNode.registry[1000].ancestors():print _
            VNode(15,17)[994,__dd__Geometry__RPC__lvRPCBarCham23--pvRPCGasgap230xb344918.46]
            VNode(13,15)[993,__dd__Geometry__RPC__lvRPCFoam--pvBarCham23Array--pvBarCham23ArrayOne..1--pvBarCham23Unit0xb344b80.23]
            VNode(11,13)[972,__dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xb344d58.23]
            VNode(9,11)[971,__dd__Geometry__RPC__lvNearRPCRoof--pvNearUnSlopModArray--pvNearUnSlopModOne..4--pvNearUnSlopMod..4--pvNearSlopModUnit0xb346868.0]
            VNode(7,9)[88,__dd__Geometry__Sites__lvNearHallTop--pvNearRPCRoof0xb356ca8.0]
            VNode(5,7)[2,__dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0.0]
            VNode(3,5)[1,__dd__Structure__Sites__db-rock0xb50e0f8.0]
            VNode(1,3)[0,top.0]

        """
        anc = []
        if andself:
            anc.append(self)
        p = self.parent
        while p is not None:
            anc.append(p)
            p = p.parent
        return anc    

    @classmethod
    def format_keys(cls, fmt):
        ptn = re.compile("\%\((\S*)\)s") 
        return ptn.findall(fmt)

    def format(self, fmt, keys=None ):
        if keys is None: 
            keys = self.format_keys(fmt)
        pass     
        nom = self.metadata
        bgm = self.boundgeom_metadata()
        pass
        d = {}
        for key in keys:
            x = key[0:2]
            k = key[2:]
            if x == 'p_':
                d["p_%s" % k] = getattr(self, k, "-")
            elif x == 'n_':
                d["n_%s" % k] = nom.get(k,"-")
            elif x == 'g_':
                d["g_%s" % k] = bgm.get(k,"-")
            else:
                pass
            pass    
        return fmt % d
 
    def __init__(self, nodepath, extras):
        """
        :param nodepath: list of node instances identifying all ancestors and the leaf geometry node
        :param rootdepth: depth 
        :param leafdepth: 

        Currently `rootdepth == leafdepth - 2`,  making each DAENode be constructed out 
        of three raw recursion levels.

        `digest` represents the identity of the specific instances(memory addresses) 
        of the nodes listed in the nodepath allowing rapid ancestor comparison
        """
        assert len(nodepath) >= 3
        leafdepth = len(nodepath)
        rootdepth = len(nodepath) - 2

        pv, lv, geo = nodepath[-3:]
        assert pv.__class__.__name__ in ('Node'), (pv, nodepath)
        assert lv.__class__.__name__ in ('NodeNode'), (lv, nodepath)
        assert geo.__class__.__name__ in ('GeometryNode'), (geo, nodepath)

        self.children = []
        self.metadata = self._metadata(extras)
        self.leafdepth = leafdepth
        self.rootdepth = rootdepth 
        self.digest = self.md5digest( nodepath[0:leafdepth-1] )
        self.parent_digest = self.md5digest( nodepath[0:rootdepth-1] )
        self.matdict = {}  # maybe filled in during index linking 

        # formerly stored ids rather than instances to allow pickling 
        self.pv = pv
        self.lv = lv   
        self.geo = geo
        #self.geo = geo.geometry.id
        pass
        self.id = self.find_uid( pv.id , False)
        self.index = len(self.registry)


    def get_matdict(self):
        assert hasattr(self, 'boundgeom'), "matdict requires associated boundgeom "
        msi = self.boundgeom.materialnodebysymbol.items()
        assert len(msi) == 1 
        symbol, matnode= msi[0]
        return dict(matid=matnode.target.id, symbol=symbol, matnode=matnode)
 
    matnode  = property(lambda self:self.matdict.get('matnode',None))
    matid  = property(lambda self:self.matdict.get('matid',None))
    symbol = property(lambda self:self.matdict.get('symbol',None))

    def primitives(self):
        if not hasattr(self, 'boundgeom'):
            return []
        bg = self.boundgeom
        lprim = list(bg.primitives())
        ret = ["nprim %s " % len(lprim)]
        for bp in lprim:
            ret.append("bp %s nvtx %s " % (str(bp),len(bp.vertex)))
            ret.append("vtxmax %s " % str(bp.vertex.max(axis=0)))
            ret.append("vtxmin %s " % str(bp.vertex.min(axis=0)))
            ret.append("vtxdif %s " % str(bp.vertex.max(axis=0)-bp.vertex.min(axis=0)))
        return ret

    def boundgeom_metadata(self):
        if not hasattr(self, 'boundgeom'):
            return {}
        extras = self.boundgeom.original.xmlnode.findall(".//{%s}extra" % COLLADA_NS )
        return self._metadata(extras)

    def __str__(self):
        lines = []
        if self.verbosity > 0:
            lines.append("  %s             %s " % (self.id, self.matdict.get('matid',"-") ) )
        if self.verbosity > 1:    
            lines.append("DAENode(%s,%s)[%s]    %s             %s " % (self.rootdepth,self.leafdepth,self.index, self.id, self.matdict.get('matid',"-") ) )
            lines.append("    pvid         %s " % self.pv.id )
            lines.append("    lvid         %s " % self.lv.id )
            lines.append("    ggid         %s " % self.geo.geometry.id )
        if self.verbosity > 2:    
            lines.extend(self.primitives())
        return "\n".join(lines)

    __repr__ = __str__

# follow the pycollada pattern for extra nodes


hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
hc_over_MeV = hc_over_GeV*1000.
hc_over_eV  = hc_over_GeV*1.e9


def as_optical_property_vector( s, xunit='MeV', yunit=None ):
    """ 
    Units of the input string first column as assumed to be MeV, 
    (G4MaterialPropertyVector raw numbers photon energies are in units of MeV)
    these are converted to nm and the order is reversed in the returned
    numpy array.
        
    :param s: string with space delimited floats representing a G4MaterialPropertyVector 
    :return: numpy array with nm
    """ 
    # from chroma/demo/optics.py 
    a = np.fromstring(s, dtype=float, sep=' ')
    assert len(a) % 2 == 0
    b = a.reshape((-1,2))[::-1]   ## reverse energy, for ascending wavelength nm

    if yunit is None or yunit in ('','mm'):
        val = b[:,1]
    elif yunit == 'cm':
        val = b[:,1]*10.
    else:   
        assert 0, "unexpected yunit %s " % yunit
        
    energy = b[:,0]
    
    hc_over_x = 0
    if xunit=='MeV':
        hc_over_x  = hc_over_MeV
    elif xunit=='eV':
        hc_over_x  = hc_over_eV
    else:       
        assert 0, "unexpected xunit %s " % xunit


    try:
        e_nm = hc_over_x/energy  
    except RuntimeWarning:
        e_nm = float('inf')      
        log.warn("RuntimeWarning in division for %s " % repr(s)) 

    vv = np.column_stack([e_nm,val])
    return vv
    





class VolMap(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)






def getExtra( top ):
    pass

def getSubCollada(arg, cfg ):
    """
    DAENode kinda merges LV and PV, but this should be a definite place, so regard as PV

    :return: collada XML string for sub geometry
    """
    arg, maxdepth = DAENode.interpret_arg(arg)
    cfg['maxdepth'] = maxdepth
    log.info("getSubCollada arg maxdepth handling %s %s " % (arg, maxdepth))

    indices = DAENode.interpret_ids(arg)
    assert len(indices) == 1, (len(indices), indices ) 
    index = indices[0]
    log.info("geom subcopy arg %s => index %s cfg %s " % (arg, index, cfg) )
    top = DAENode.indexget(index)  


    extra = E.extra()
    meta = E.meta("subcopy arg %s index %s maxdepth %s " % (arg, index, cfg['maxdepth']) )
    extra.append(meta)
    for a in reversed(top.ancestors()):
        ancestor = E.ancestor(str(a.index), id=a.id)
        extra.append(ancestor)
    pass
    extra.append(E.subroot(str(top.index), id=top.id))
    pass
    for c in top.children:
        child = E.child(str(c.index), id=c.id)
        extra.append(child)
    pass
    cfg['extra'] = extra


    vc = DAECopy(top, cfg )
    DAENormalize(vc.dae.xmlnode)

    xmlshebang = cfg.get('xmlshebang',False)
    shebang = '<?xml version="1.0" encoding="ISO-8859-1"?>'
    if xmlshebang:
        svc = "\n".join([shebang,str(vc)])
    else:
        svc = str(vc)
    pass    

    subpath = cfg.get('subpath', None)
    if not subpath is None and cfg.get('daesave',False) == True:
        log.info("daesave to %s " % subpath )
        fp = open(subpath, "w") 
        fp.write(svc)
        fp.close()
    pass 

    debug = cfg.get('debug',False)
    if debug:
        global gsub, gtop
        gsub = vc
        gtop = top
    pass 

    return svc

