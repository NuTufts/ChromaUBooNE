# stores the classes used by DAENode to store Geant4 properties
import collada
from collada.common import DaeObject
from tools import read_properties

class MaterialProperties(DaeObject):
    def __init__(self, properties, xmlnode):
        self.properties = properties
        self.xmlnode = xmlnode
    @staticmethod
    def load(collada, localscope, xmlnode):
        properties = read_properties(xmlnode)
        return MaterialProperties( properties, xmlnode ) 
    def __repr__(self):
        return "<MaterialProperties keys=%s >" % (str(self.properties.keys())) 


class OpticalSurface(DaeObject):
    @classmethod
    def lookup(cls, localscope, surfaceproperty):
        assert surfaceproperty in localscope['surfaceproperty'], localscope
        return localscope['surfaceproperty'][surfaceproperty]

    @classmethod
    def sensitive(cls, name, properties):
        """
        Guessed defaults that are probably not translated anywhere in Chroma model,
        but need values to avoid asserts
        """
        return cls(name=name,finish=0, model=1, type_=0, properties=properties)


    def __init__(self, name=None, finish=None, model=None, type_=None, value=None, properties=None, xmlnode=None):
        """
        Reference

        * `materials/include/G4OpticalSurface.hh`

        """
        self.name = name
        self.finish = finish # 0:polished (smooth perfectly polished surface) 3:ground (rough surface)   (mostly "ground", ESR interfaces "polished")
        self.model = model   # 0:glisur  1:UNIFIED  2:LUT   (all are UNIFIED)
        self.type_ = type_   # 0:dielectric_metal 1:dielectric_dielectric     (all are dielectric_metal)
        self.value = value   # 1. 0. (ESRAir) or 0.2 (Pool Curtain/Liner)
        self.properties = properties
        self.xmlnode = xmlnode

    @staticmethod 
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name'] 
        finish = xmlnode.attrib['finish'] 
        model = xmlnode.attrib['model'] 
        type_ = xmlnode.attrib['type'] 
        value = xmlnode.attrib['value'] 
        properties = read_properties(xmlnode)
        return OpticalSurface(name, finish, model, type_, value, properties, xmlnode )

    def __repr__(self):
        return "<OpticalSurface f%s m%s t%s v%s p%s >" % (self.finish,self.model,self.type_,self.value,str(",".join(["%s:%s" % (k,len(self.properties[k])) for k in self.properties]))) 
        #return "%s" % (str(",".join(self.properties.keys()))) 



class SkinSurface(DaeObject):
    """
    skinsurface/volumeref/@ref are LV names

    ::

        dump_skinsurface
        [00] <SkinSurface PoolDetails__NearPoolSurfaces__NearPoolCoverSurface RINDEX,REFLECTIVITY >
             PoolDetails__lvNearTopCover0xad9a470
        [01] <SkinSurface AdDetails__AdSurfacesAll__RSOilSurface BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT,REFLECTIVITY,SPECULARLOBECONSTANT >
             AdDetails__lvRadialShieldUnit0xaea9f58
        [02] <SkinSurface AdDetails__AdSurfacesAll__AdCableTraySurface RINDEX,REFLECTIVITY >
             AdDetails__lvAdVertiCableTray0xaf28da0
        [03] <SkinSurface PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface RINDEX,REFLECTIVITY >
             PMT__lvPmtTopRing0xaf434c8
        [04] <SkinSurface PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface RINDEX,REFLECTIVITY >
             PMT__lvPmtBaseRing0xaf43520

    """
    def __init__(self, name=None, surfaceproperty=None, volumeref=None, xmlnode=None):
        self.name = name
        self.surfaceproperty = surfaceproperty
        self.volumeref = volumeref
        self.xmlnode = xmlnode
        self.debug = True

    @classmethod 
    def sensitive(cls, name, volumeref, surfaceproperty):
        """
        * __dd__Geometry__PMT__lvPmtHemiCathodeSensitiveSurface 
        * __dd__Geometry__PMT__lvHeadonPmtCathodeSensitiveSurface
        """
        xmlnode = None
        return SkinSurface(name, surfaceproperty, volumeref, xmlnode)


    @staticmethod
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name']
        surfaceproperty = xmlnode.attrib['surfaceproperty']
        surfaceproperty = OpticalSurface.lookup( localscope, surfaceproperty)
        volumeref = xmlnode.find(tag('volumeref'))
        assert volumeref is not None
        volumeref = volumeref.attrib['ref']     
        return SkinSurface(name, surfaceproperty, volumeref, xmlnode)
    def __repr__(self):
        elide = "__dd__Geometry__"
        name = str(self.name)
        if name.startswith(elide):
            name = name[len(elide):]
        smry = "<SkinSurface %s %s >" % (name, str(self.surfaceproperty)) 
        if self.debug:
            lvr = self.volumeref
            if lvr.startswith(elide):
                lvr=lvr[len(elide):]
            lvn = DAENode.lvfind(self.volumeref)  # lvfind lookup forces the parsing order
            smry += "\n" + "     %s %s " % (lvr, len(lvn))
        return smry

class BorderSurface(DaeObject):
    """ 
    """
    def __init__(self, name=None, surfaceproperty=None, physvolref1=None, physvolref2=None, xmlnode=None):
        self.name = name
        self.surfaceproperty = surfaceproperty
        self.physvolref1 = physvolref1
        self.physvolref2 = physvolref2
        self.xmlnode = xmlnode
        self.debug = True
    @staticmethod
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name']
        surfaceproperty = xmlnode.attrib['surfaceproperty']
        surfaceproperty = OpticalSurface.lookup( localscope, surfaceproperty)
        physvolref = xmlnode.findall(tag('physvolref'))
        assert len(physvolref) == 2
        physvolref1 = physvolref[0].attrib['ref']     
        physvolref2 = physvolref[1].attrib['ref']     
        return BorderSurface(name, surfaceproperty, physvolref1, physvolref2, xmlnode)
    def __repr__(self):
        def elide_(s): 
            elide = "__dd__Geometry__"
            if s.startswith(elide):
                return s[len(elide):]
            return s
        nlin = "\n     "
        smry = "<BorderSurface %s %s >" % (elide_(self.name), str(self.surfaceproperty)) 
        if self.debug:
            pvr1 = elide_(self.physvolref1)
            pv1 = DAENode.pvfind(self.physvolref1)  # pvfind lookup forces the parsing order
            hdr1 = "pv1 (%s) %s " % (len(pv1), pvr1)
            smry += nlin + nlin.join([hdr1]+map(str,pv1))

            pvr2 = elide_(self.physvolref2)
            pv2 = DAENode.pvfind(self.physvolref2)
            hdr2 = "pv2 (%s) %s " % (len(pv2), pvr2)
            smry += nlin + nlin.join([hdr2]+map(str,pv2))
        return smry
