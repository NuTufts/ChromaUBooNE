#!/usr/bin/env python
"""

.. warning:: Material codes obtained standalone will not match those in-app 
             if geometry selection is different. Use daegeometry.sh for an easy 
             way to duplicate what the standard app does.

NB the collada to chroma functionality is kept separate 
from DAENode as that needs to operate on machines
where Chroma cannot be installed

Interactive access to geometry via embedded ipython::

    delta:~ blyth$ collada_to_chroma.sh
    ...
    In [1]: g.material1_index
    Out[1]: array([13, 13, 13, ..., 34, 34, 34], dtype=int32)

    In [2]: map(len,[g.material1_index,g.material2_index,g.surface_index,g.unique_materials,g.unique_surfaces])
    Out[2]: [2448160, 2448160, 2448160, 36, 35]

::

    In [33]: map(lambda _:"%s %s" % (_[0],_[1].shape), m.daeprops.items() )
    Out[33]: 
    [
      'SLOWCOMPONENT (275, 2)',
      'FASTCOMPONENT (275, 2)',

      'RAYLEIGH (11, 2)',
      'RINDEX (18, 2)',
      'REEMISSIONPROB (28, 2)',
      'ABSLENGTH (497, 2)',


Making plots comparing GdLS and LS properties (using the collada source properties)::

    cfplt_("RAYLEIGH")  + [plt.show()]        # look same 
    cfplt_("RINDEX")    + [plt.show()]        # look same 
    cfplt_("REEMISSIONPROB")  + [plt.show()]  # look same : whacky top hat 

    cfplt_("SLOWCOMPONENT")  + [plt.show()]   # look same : twin peaks 
    cfplt_("FASTCOMPONENT")  + [plt.show()]   # look same : twin peaks (same as SLOW)

    cfplt_("ABSLENGTH")  + [plt.show()]       # distinct between 400-600 nm 


Using the chroma geometry properties, that are derived from the collada source ones::

    In [3]: plt.plot( gdls.reemission_cdf[:,0], gdls.reemission_cdf[:,1] )
    Out[3]: [<matplotlib.lines.Line2D at 0x118d46950>]

    In [4]: plt.show()

    In [5]: plt.plot( ls.reemission_cdf[:,0], ls.reemission_cdf[:,1] )
    Out[5]: [<matplotlib.lines.Line2D at 0x11b58d210>]

    In [6]: plt.show()


"""
import os, sys, logging, re, json
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 


import numpy as np
from chroma.g4daenode import DAENode
from chroma.geometry import Mesh, Solid, Material, Surface, Geometry
from chroma.detector import Detector
from chroma.loader import load_bvh

import matplotlib.pyplot as plt

def _get_daeprops(self):
    if self.dae.extra is None:
        return {}
    return self.dae.extra.properties    

#
# daeprops must not be present for NPYCacheable to succeed 
# with its cache writing/reading so following switch off 
# may need to --geocacheupdate
#
DEBUG = 0  
if DEBUG:
    Material.daeprops = property(_get_daeprops)


matptn = re.compile("^__dd__Materials__(\S*)0x\S{7}$")
def matshorten(name):
    m = matptn.match(name)
    if m:
        return m.group(1) 
    return name


def construct_cdf( xy ):
    """
    :param xy:

    Creating cumulative density functions needed by chroma, 
    eg for generating a wavelengths of reemitted photons.::

        In [52]: sc
        Out[52]: 
        array([[  79.9898,    0.    ],
               [ 120.0235,    0.    ],
               [ 199.9746,    0.    ],
               ..., 
               [ 599.0011,    0.0017],
               [ 600.0012,    0.0018],
               [ 799.8984,    0.    ]])

        In [53]: cdf = construct_cdf(sc)

        In [54]: cdf
        Out[54]: 
        array([[  79.9898,    0.    ],
               [ 120.0235,    0.    ],
               [ 199.9746,    0.    ],
               ..., 
               [ 599.0011,    1.    ],
               [ 600.0012,    1.    ],
               [ 799.8984,    1.    ]])

        In [55]: plt.plot( cdf[:,0], cdf[:,1] )
        Out[55]: [<matplotlib.lines.Line2D at 0x125115e10>]

        In [56]: plt.show()   # sigmoidal 

        In [57]: plt.plot( cdf[:,0], cdf[:,1] ) + plt.plot( sc[:,0], sc[:,1] )
        Out[57]: 
        [<matplotlib.lines.Line2D at 0x123fa0e90>,
         <matplotlib.lines.Line2D at 0x125146150>]

        In [58]: plt.show()


    

    """
    assert len(xy.shape) == 2 and xy.shape[-1] == 2
    x,y  = xy[:,0], xy[:,1]
    cy = np.cumsum(y)
    cdf_y = cy/cy[-1]   # normalize to 1 at RHS
    return np.vstack([x,cdf_y]).T
         


def construct_cdf_energywise(xy):
    """
    Duplicates DsChromaG4Scintillation::BuildThePhysicsTable     
    """
    assert len(xy.shape) == 2 and xy.shape[-1] == 2

    bcdf = np.empty( xy.shape )

    rxy = xy[::-1]              # reverse order, for ascending energy 

    x = 1/rxy[:,0]              # work in inverse wavelength 1/nm

    y = rxy[:,1]

    ymid = (y[:-1]+y[1:])/2     # looses entry as needs pair

    xdif = np.diff(x)            

    #bcdf[:,0] = rxy[:,0]        # back to wavelength
    bcdf[:,0] = x                # keeping 1/wavelenth

    bcdf[0,1] = 0.

    np.cumsum(ymid*xdif, out=bcdf[1:,1])

    bcdf[1:,1] = bcdf[1:,1]/bcdf[1:,1].max() 

    return bcdf         




class OpticalSurfaceFinish(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4OpticalSurface.hh`::

         61 enum G4OpticalSurfaceFinish
         62 {
         63    polished,                    // smooth perfectly polished surface
         64    polishedfrontpainted,        // smooth top-layer (front) paint
         65    polishedbackpainted,         // same is 'polished' but with a back-paint
         66    ground,                      // rough surface
         67    groundfrontpainted,          // rough top-layer (front) paint
         68    groundbackpainted            // same as 'ground' but with a back-paint
         69 };
         70 

    """
    polished = 0
    ground = 3
       

class OpticalSurfaceModel(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4OpticalSurface.hh`::

         71 enum G4OpticalSurfaceModel
         72 {
         73    glisur,                      // original GEANT3 model
         74    unified                      // UNIFIED model
         75 };

    """
    glisur = 0
    unified = 1


class SurfaceType(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4SurfaceProperty.hh`::

         66 enum G4SurfaceType
         67 {
         68    dielectric_metal,            // dielectric-metal interface
         69    dielectric_dielectric,       // dielectric-dielectric interface
         70    firsov,                      // for Firsov Process
         71    x_ray                        // for x-ray mirror process
         72 };

    """
    dielectric_metal = 0
    dielectric_dielectric = 1




class ColladaToChroma(object):
    secs = {}
    surface_props = "detect absorb reemit reflect_diffuse reflect_specular eta k reemission_cdf".split()

    def __init__(self, nodecls, bvh=False):
        """
        :param nodecls: typically DAENode
        """ 
        log.debug("ColladaToChroma")
        self.nodecls = nodecls
        self.bvh = bvh
        #self.chroma_geometry = Geometry(detector_material=None)    # bialkali ?
        self.chroma_geometry = Detector(detector_material=None)    
        pass
        self.vcount = 0

        self.surfaces = {}
        self.materials = {}   # dict of chroma.geometry.Material 
        self._materialmap = {}  # dict with short name keys 
        self._surfacemap = {}  # dict with short name keys 

        # idmap checking 
        self.channel_count = 0
        self.channel_ids = set()

    def convert_opticalsurfaces(self, debug=False):
        """
        """
        log.info("convert_opticalsurfaces")
        for dsurf in self.nodecls.extra.opticalsurface:
            surface = self.make_opticalsurface( dsurf, debug=debug)
            self.surfaces[surface.name] = surface
        pass 
        #assert len(self.surfaces) == len(self.nodecls.extra.opticalsurface), "opticalsurface with duplicate names ? "
        log.info("convert_opticalsurfaces creates %s from %s  " % (len(self.surfaces),len(self.nodecls.extra.opticalsurface))  )

    def make_opticalsurface(self, dsurf, debug=False):
        """
        :param dsurf: G4DAE surface
        :return: Chroma surface 

        * name
        * model ? defaults to 0

        G4DAE Optical Surface properties

        * REFLECTIVITY (the only property to be widely defined)
        * RINDEX (seems odd for a surface, looks to always be zero) 
        * SPECULARLOBECONSTANT  (set to 0.85 for a few surface)
        * BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT (often present, always zero)


        `chroma/geometry_types.h`::

           enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS };

        Potentially wavelength dependent props all default to zero.
        Having values for these is necessary to get SURFACE_DETECT, SURFACE_ABSORB

        * detect
        * absorb
        * reflect_diffuse
        * reflect_specular

        * reemit
        * eta
        * k
        * reemission_cdf


        `chroma/cuda/photon.h`::

            701 __device__ int
            702 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
            703                      bool use_weights=false)
            704 {
            705     Surface *surface = geometry->surfaces[s.surface_index];
            706 
            707     if (surface->model == SURFACE_COMPLEX)
            708         return propagate_complex(p, s, rng, surface, use_weights);
            709     else if (surface->model == SURFACE_WLS)
            710         return propagate_at_wls(p, s, rng, surface, use_weights);
            711     else
            712     {
            713         // use default surface model: do a combination of specular and
            714         // diffuse reflection, detection, and absorption based on relative
            715         // probabilties
            716 
            717         // since the surface properties are interpolated linearly, we are
            718         // guaranteed that they still sum to 1.0.
            719         float detect = interp_property(surface, p.wavelength, surface->detect);
            720         float absorb = interp_property(surface, p.wavelength, surface->absorb);
            721         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
            722         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
            723 
        """
        if debug:
            print "%-75s %s " % (dsurf.name, dsurf )
        surface = Surface(dsurf.name)

        finish_map = { 
              OpticalSurfaceFinish.polished:'reflect_specular',
              OpticalSurfaceFinish.ground:'reflect_diffuse',
             }

        if 'EFFICIENCY' in dsurf.properties:
            EFFICIENCY = dsurf.properties.get('EFFICIENCY',None) 
            surface.set('detect', EFFICIENCY[:,1], wavelengths=EFFICIENCY[:,0])
            pass
        elif 'REFLECTIVITY' in dsurf.properties:
            REFLECTIVITY = dsurf.properties.get('REFLECTIVITY',None) 
            key = finish_map.get( int(dsurf.finish), None)
            if key is None or REFLECTIVITY is None:
                log.warn("miss REFLECTIVITY key : not setting REFLECTIVITY for %s " % surface.name )
            else: 
                log.debug("setting prop %s for surface %s " % (key, surface.name))
                surface.set(key, REFLECTIVITY[:,1], wavelengths=REFLECTIVITY[:,0])
            pass
        else:
            log.warn(" no REFLECTIVITY/EFFICIENCY in dsurf.properties %s " % repr(dsurf.properties))
        pass
        return surface


    def collada_materials_summary(self, names=['GdDopedLS','LiquidScintillator']):
        collada = self.nodecls.orig 
        find_ = lambda name:filter(lambda m:m.id.find(name) > -1, collada.materials)
        for name in names:
            mats = find_(name)
            assert len(mats) == 1, "name is ambiguous or missing"
            self.dump_collada_material(mats[0])
        pass

    def dump_collada_material(self, mat):
         extra = getattr(mat,'extra',None)
         keys  = extra.properties.keys() if extra else []
         print mat.id
         keys = sorted(keys, key=lambda k:extra.properties[k].shape[0], reverse=True)
         for k in keys:
             xy = extra.properties[k]
             x = xy[:,0]
             y = xy[:,1]
             print "%30s %10s    %10.3f %10.3f   %10.3f %10.3f   " % (k, repr(xy.shape), x.min(), x.max(), y.min(), y.max() ) 


    def convert_materials(self, debug=False):
        """
        #. creates chroma Material instances for each collada material 
        #. fills in properties from the collada extras
        #. records materials in a map keyed by material.name

        Chroma materials default to None, 3 settings:

        * refractive_index
        * absorption_length
        * scattering_length

        And defaults to zero, 2 settings:

        * reemission_prob
        * reemission_cdf

        G4DAE materials have an extra attribute dict that 
        contains keys such as

        * RINDEX
        * ABSLENGTH
        * RAYLEIGH

        Uncertain of key correspondence, especially REEMISSIONPROB
        and what about reemission_cdf ? Possibly the many Scintillator
        keys can provide that ?

        Probably many the scintillator keys are only relevant to photon 
        production rather than photon propagation, so they are irrelevant 
        to Chroma.

        Which materials have each::

             EFFICIENCY                     [1 ] Bialkali 

             -------------- assumed to not apply to optical photons ---------

             FASTTIMECONSTANT               [2 ] GdDopedLS,LiquidScintillator 
             SLOWTIMECONSTANT               [2 ] GdDopedLS,LiquidScintillator 
             YIELDRATIO                     [2 ] GdDopedLS,LiquidScintillator 

             GammaFASTTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             GammaSLOWTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             GammaYIELDRATIO                [2 ] GdDopedLS,LiquidScintillator 

             AlphaFASTTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             AlphaSLOWTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             AlphaYIELDRATIO                [2 ] GdDopedLS,LiquidScintillator 

             NeutronFASTTIMECONSTANT        [2 ] GdDopedLS,LiquidScintillator 
             NeutronSLOWTIMECONSTANT        [2 ] GdDopedLS,LiquidScintillator 
             NeutronYIELDRATIO              [2 ] GdDopedLS,LiquidScintillator 

             SCINTILLATIONYIELD             [2 ] GdDopedLS,LiquidScintillator 
             RESOLUTIONSCALE                [2 ] GdDopedLS,LiquidScintillator 

             ---------------------------------------------------------------------

             ReemissionFASTTIMECONSTANT     [2 ] GdDopedLS,LiquidScintillator      for opticalphoton
             ReemissionSLOWTIMECONSTANT     [2 ] GdDopedLS,LiquidScintillator 
             ReemissionYIELDRATIO           [2 ] GdDopedLS,LiquidScintillator 

             FASTCOMPONENT                  [2 ] GdDopedLS,LiquidScintillator     "Fast_Intensity"
             SLOWCOMPONENT                  [2 ] GdDopedLS,LiquidScintillator     "Slow_Intensity"
             REEMISSIONPROB                 [2 ] GdDopedLS,LiquidScintillator     "Reemission_Prob"

             ------------------------------------------------------------------------

             RAYLEIGH                       [5 ] GdDopedLS,Acrylic,Teflon,LiquidScintillator,MineralOil 
             RINDEX                         [14] Air,GdDopedLS,Acrylic,Teflon,LiquidScintillator,Bialkali,
                                                 Vacuum,Pyrex,MineralOil,Water,NitrogenGas,IwsWater,OwsWater,DeadWater 
             ABSLENGTH                      [20] PPE,Air,GdDopedLS,Acrylic,Teflon,LiquidScintillator,Bialkali,
                                                 Vacuum,Pyrex,UnstStainlessSteel,StainlessSteel,
                                                 ESR,MineralOil,Water,NitrogenGas,IwsWater,ADTableStainlessSteel,Tyvek,OwsWater,DeadWater 

        Observations:
 
        #. no RAYLEIGH for water 


        """
        keymap = {
                          "RINDEX":'refractive_index',
                       "ABSLENGTH":'absorption_length',
                        "RAYLEIGH":'scattering_length',
                  "REEMISSIONPROB":'reemission_prob',
               }  


        keymat = {}
        collada = self.nodecls.orig 
        for dmaterial in collada.materials:
            material = Material(dmaterial.id)   
            if DEBUG:
                material.dae = dmaterial

            # vacuum like defaults ? is that appropriate ? what is the G4 equivalent ?
            material.set('refractive_index', 1.0)  
            material.set('absorption_length',1e6)
            material.set('scattering_length',1e6)
        
            if dmaterial.extra is not None:
                props = dmaterial.extra.properties
                for dkey,dval in props.items():
                    if dkey not in keymat:
                        keymat[dkey] = []
                    keymat[dkey].append(material.name) # record of materials that have each key 

                    if dkey in keymap:
                        key = keymap[dkey]
                        material.set(key, dval[:,1], wavelengths=dval[:,0])
                        log.debug("for material %s set Chroma prop %s from G4DAE prop %s vals %s " % ( material.name, key, dkey, len(dval))) 
                    else:
                        log.debug("for material %s skipping G4DAE prop %s vals %s " % ( material.name, dkey, len(dval)))  
                    pass 
                    self.setup_cdf( material, props )
                pass
            pass 
            self.materials[material.name] = material
        pass
        log.debug("convert_materials G4DAE keys encountered : %s " % len(keymat))
        if debug: 
            for dkey in sorted(keymat,key=lambda _:len(keymat[_])): 
                mats = keymat[dkey]
                print " %-30s [%-2s] %s " % ( dkey, len(mats), ",".join(map(matshorten,mats)) )

    def setup_cdf(self, material, props ):
        """
        Chroma uses "reemission_cdf" cumulative distribution function 
        to generate the wavelength of reemission photons. 

        Currently think that the name "reemission_cdf" is misleading, 
        as it is the RHS normalized CDF obtained from an intensity distribution
        (photon intensity as function of wavelength) 

        NB REEMISSIONPROB->reemission_prob is handled as a 
        normal keymapped property, no need to integrate to construct 
        the cdf for that.
    
        Compare this with the C++

           DsChromaG4Scintillation::BuildThePhysicsTable()  

        """  
        fast = props.get('FASTCOMPONENT', None)
        slow = props.get('SLOWCOMPONENT', None) 
        reem = props.get('REEMISSIONPROB', None) 

        if fast is None or slow is None or reem is None:
            return 

        assert not fast is None 
        assert not slow is None 
        assert not reem is None

        assert np.all( fast == slow )     # CURIOUS, that these are the same

        reemission_cdf = construct_cdf_energywise( fast ) 

        ## yep "fast" : need intensity distribution 
        #
        #   reem_cdf = construct_cdf( reem )
        #   
        #   Nope the CDF are used to generate wavelengths 
        #   following the desired slow/fast intensity distribution 
        #   [ie number of photons in wavelength ranges]
        #   (which happen to be the same)
        # 
        #   conversely the reemission probability gives the 
        #   fraction that reemit at the wavelength
        #   that value can be used directly by random uniform throws
        #   to decide whether to reemit no cdf gymnastics needed
        #   as are just determining whether somethinh happens not
        #   the wavelength distribution  of photons
        # 
        #

        log.debug("setting reemission_cdf for %s to %s " % (material.name, repr(reemission_cdf)))

        #material.set('reemission_cdf', reemission_cdf[:,1], wavelengths=reemission_cdf[:,0])
        material.setraw('reemission_cdf', reemission_cdf)



    def _get_materialmap(self):
        """
        Dict of chroma.geometry.Material instances with short name keys   
        """
        if len(self._materialmap)==0:
            prefix = '__dd__Materials__'
            for name,mat  in self.materials.items():
                if name.startswith(prefix):
                    name = name[len(prefix):]
                if name[-9:-7] == '0x':
                    name = name[:-9]
                pass 
                self._materialmap[name] = mat
            pass
        return self._materialmap
    materialmap = property(_get_materialmap)
       

    def _get_surfacemap(self):
        """
        Dict of chroma.geometry.Surface instances with short name keys   
        """
        postfix = 'Surface'
        if len(self._surfacemap)==0:
            for name,surf  in self.surfaces.items():
                prefix = "__".join(name.split("__")[:-1]) + "__"
                if name.startswith(prefix):
                    nam = name[len(prefix):]
                if nam.endswith(postfix):
                    nam = nam[:-len(postfix)]
                pass 
                self._surfacemap[nam] = surf
            pass
        return self._surfacemap
    surfacemap = property(_get_surfacemap)
 
 
    def property_plot(self, matname , propname ):
        import matplotlib.pyplot as plt
        mat = self.materialmap[matname]
        xy = mat.daeprops[propname]
        #plt.plot( xy[:,0], xy[:,1] )
        plt.plot(*xy.T)

    def convert_geometry_traverse(self, nodes=None):
        log.debug("convert_geometry_traverse")
        if nodes is None:  
            self.nodecls.vwalk(self.visit)
        else:
            for node in nodes:
                self.visit(node)
        pass
        self.dump_channel_info()

    def convert_flatten(self):
        log.debug("ColladaToChroma convert_geometry flattening %s " % len(self.chroma_geometry.solids))
        self.chroma_geometry.flatten()

    def convert_make_maps(self):
        self.cmm = self.make_chroma_material_map( self.chroma_geometry )
        self.csm = self.make_chroma_surface_map( self.chroma_geometry )


    def convert_geometry(self, nodes=None):
        """
        :param nodes: list of DAENode instances or None

        Converts DAENode/pycollada geometry into Chroma geometry.

        When `nodes=None` the entire DAENode tree is visited and converted, 
        otherwise just the listed nodes.
        """ 
        log.debug("convert_geometry")

        self.convert_materials() 
        self.convert_opticalsurfaces() 
        self.convert_geometry_traverse(nodes) 
        self.convert_flatten() 
        self.convert_make_maps() 

        if self.bvh:
            self.add_bvh()
        return self.chroma_geometry

    def make_chroma_material_map(self, chroma_geometry):
        """
        Curiously the order of chroma_geometry.unique_materials on different invokations is 
        "fairly constant" but not precisely so. 
        How is that possible ? Perfect or random would seem more likely outcomes. 
        """
        unique_materials = chroma_geometry.unique_materials
        material_lookup = dict(zip(unique_materials, range(len(unique_materials))))
        cmm = dict([(material_lookup[m],m.name) for m in filter(None,unique_materials)])
        cmm[-1] = "ANY"
        cmm[999] = "UNKNOWN"
        return cmm

    def make_chroma_surface_map(self, chroma_geometry):
        unique_surfaces = chroma_geometry.unique_surfaces
        surface_lookup = dict(zip(unique_surfaces, range(len(unique_surfaces))))
        csm = dict([(surface_lookup[s],s.name) for s in filter(None,unique_surfaces)])
        csm[-1] = "ANY"
        csm[999] = "UNKNOWN"
        return csm


    def add_bvh( self, bvh_name="default", auto_build_bvh=True, read_bvh_cache=True, update_bvh_cache=True, cache_dir=None, cuda_device=None):
        """
        As done by chroma.loader
        """
        log.debug("ColladaToChroma adding BVH")
        self.chroma_geometry.bvh = load_bvh(self.chroma_geometry, 
                                            bvh_name=bvh_name,
                                            auto_build_bvh=auto_build_bvh,
                                            read_bvh_cache=read_bvh_cache,
                                            update_bvh_cache=update_bvh_cache,
                                            cache_dir=cache_dir,
                                            cuda_device=cuda_device)
        log.debug("completed adding BVH")

    def find_outer_inner_materials(self, node ):
        """
        :param node: DAENode instance
        :return: Chroma Material instances for outer and inner materials

        #. Parent node material regarded as outside
        #. Current node material regarded as inside        

        Think about a leaf node to see the sense of that.

        Caveat, the meanings of "inner" and "outer" depend on 
        the orientation of the triangles that make up the surface...  
        So just adopt a convention and try to verify it later.
        """
        assert node.__class__.__name__ == 'DAENode'
        this_material = self.materials[node.matid]
        if node.parent is None:
            parent_material = this_material
            log.warning("setting parent_material to %s as parent is None for node %s " % (parent_material.name, node.id) ) 
        else:
            parent_material = self.materials[node.parent.matid]

        log.debug("find_outer_inner_materials node %s %s %s" % (node, this_material, parent_material))
        return parent_material, this_material

       
    def find_skinsurface(self, node):
        """
        :param node: DAENode instance
        :return: G4DAE Surface instance corresponding to G4LogicalSkinSurface if one is available for the LV of the current node


        * ambiguous skin for lvid __dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0 found 672 
        * if the properties are the same then ambiguity not a problem ?

        """
        assert node.__class__.__name__ == 'DAENode'
        assert self.nodecls.extra.__class__.__name__ == 'DAEExtra'

        ssid = self.nodecls.sensitive_surface_id(node)

        if not ssid is None:
            skin = self.nodecls.extra.skinmap.get(ssid, None)
            log.debug("ssid %s skin %s " % (ssid, repr(skin)))
            if skin is not None:
                if len(skin) > 0:  # expected for sensitives
                    skin = skin[0]
            pass
        else:
            lvid = node.lv.id
            skin = self.nodecls.extra.skinmap.get(lvid, None)
            if skin is not None:
                assert len(skin) == 1, "ambiguous skin for lvid %s found %s  " % (lvid, len(skin)) 
                ##log.warn("ambiguous skin for lvid %s found %s : USING FIRST  " % (lvid, len(skin))) 
                skin = skin[0]
            pass

        return skin
           
    def find_bordersurface(self, node):
        """
        :param node: DAENode instance
        :return: G4DAE Surface instance corresponding to G4LogicalBorderSurface 
                 if one is available for the PVs of the current node and its parent

        Ambiguity bug makes this difficult
        """
        assert node.__class__.__name__ == 'DAENode'
        pass
        #pvid = node.pv.id
        #ppvid = node.parent.pv.id
        #border = self.nodecls.extra.bordermap.get(pvid, None)
        return None


    def find_surface(self, node):
        """
        :param node: DAENode instance
        :return Chroma Surface instance or None:

        G4DAE persists the below surface elements which 
        both reference "opticalsurface" containing the keyed properties
        
        * "skinsurface" (single volumeref, ref by lv.id)
        * "boundarysurface" (physvolref ordered pair, identified by pv1.id,pv2.id) 
          
        The boundary pairs are always parent/child nodes in dyb Near geometry, 
        they could in principal be siblings.
        """
        assert node.__class__.__name__ == 'DAENode'

        skin = self.find_skinsurface( node )
        border = self.find_bordersurface( node )

        dsurf = filter(None,[skin, border])
        assert len(dsurf)<2, "Not expecting both skin %s and border %s surface for the same node %s "  % (skin, border, node)
        if len(dsurf) == 1:
            dsurface = dsurf[0]
            log.debug("found dsurface %s for node %s " % (dsurface, node ))
            surface = self.surfaces.get(dsurface.name, None)
            assert surface is not None, "dsurface %s without corresponding chroma surface of name %s for node %s " % ( dsurface, dsurface.name, node.id) 
        else:
            surface = None  
        pass 
        return surface


    def visit(self, node, debug=False):
        """
        :param node: DAENode instance

        DAENode instances and their pycollada underpinnings meet chroma here

        Chroma needs sensitive detectors to have an associated surface 
        with detect property ...
        """
        #assert node.__class__.__name__ == 'DAENode'
        self.vcount += 1
        if self.vcount < 10:
            log.debug("visit : vcount %s node.index %s node.id %s " % ( self.vcount, node.index, node.id ))

        bps = list(node.boundgeom.primitives())
        
        bpl = bps[0]
        
        assert len(bps) == 1 and bpl.__class__.__name__ == 'BoundPolylist'
        
        tris = bpl.triangleset()

        vertices = tris._vertex

        triangles = tris._vertex_index

        mesh = Mesh( vertices, triangles, remove_duplicate_vertices=False ) 

        material2, material1 = self.find_outer_inner_materials(node)   

        surface = self.find_surface( node )   # lookup Chroma surface corresponding to the node
        if surface==None:
            surfacename = "NOT SPECIFIED"
        else:
            surfacename = surface.name
        print "[NODE %05d:%s]"%(node.index,node.lv.id)," NTriangles=%d OuterMat=%s InnerMat=%s Surface=%s"%(len(mesh.triangles), material2.name, material1.name, surfacename)
        if "volDetEnclosure" in node.lv.id:
            print vertices

        color = 0x33ffffff 

        solid = Solid( mesh, material1, material2, surface, color )
        solid.node = node

        #
        # hmm a PMT is comprised of several volumes all of which 
        # have the same associated channel_id 
        #
        channel_id = getattr(node, 'channel_id', None) 
        if not channel_id is None and channel_id > 0:
            self.channel_count += 1             # nodes with associated non zero channel_id
            self.channel_ids.add(channel_id)
            self.chroma_geometry.add_pmt( solid, channel_id=channel_id)
        else:
            self.chroma_geometry.add_solid( solid )
        pass

        if debug and self.vcount % 1000 == 0:
            print node.id
            print self.vcount, bpl, tris, tris.material
            print mesh
            #print mesh.assemble()
            bounds =  mesh.get_bounds()
            extent = bounds[1] - bounds[0]
            print extent

    def dump_channel_info(self):
        log.info("channel_count (nodes with channel_id > 0) : %s  uniques %s " % ( self.channel_count, len(set(self.channel_ids))))
        log.debug("channel_ids %s " % repr(self.channel_ids))

    def surface_props_table(self):
        """
        ::

            plt.plot(*cc.surfacemap['NearOWSLiner'].reflect_diffuse.T)
            plt.plot(*cc.surfacemap['NearDeadLiner'].reflect_diffuse.T)
            plt.plot(*cc.surfacemap['NearIWSCurtain'].reflect_diffuse.T)  ## all three the same, up to plateau

            plt.plot(*cc.surfacemap['RSOil'].reflect_diffuse.T)    ## falloff 

            plt.plot(*cc.surfacemap['ESRAirSurfaceBot'].reflect_specular.T) 
            plt.plot(*cc.surfacemap['ESRAirSurfaceTop'].reflect_specular.T)  ## Bot and Top the same, cliff

        """
        def smry(spt, suppress="60.0:800.0 =0.0"):
            x,y = spt.T
            xsmr = "%3.1f:%3.1f" % (x.min(),x.max())
            ymin, ymax = y.min(), y.max()
            ysmr = "=%3.1f" % ymin if ymin == ymax else "%3.1f:%3.1f" % (ymin, ymax)
            s = "%s %s" % (xsmr,ysmr)
            return "-" if s == suppress else s
        pass
        lfmt_ = lambda _:"%-23s" % _    
        bfmt_ = lambda _:" ".join(["%-25s" % s for s in _])    
        print lfmt_("surf") + bfmt_(self.surface_props) 
        for nam, surf in self.surfacemap.items():
            sprop = map( lambda prop:smry(getattr(surf,prop)), self.surface_props )     
            print lfmt_(nam) + bfmt_(sprop) 


def daeload(path=None, bvh=False ):
   """
   :param path:
   :return Chroma Geometry instance:

   This is invoked by chroma.loader.load_geometry_from_string when
   the string ends with ".dae".  This allows the standard chroma-cam 
   to be used with COLLADA geometries.

   TODO: add nodespec capabilities here to allow loading partial geometries from chroma-cam 
   """
   DAENode.init(path)
   cc = ColladaToChroma(DAENode, bvh=bvh )
   cc.convert_geometry()
   return cc.chroma_geometry, cc


    
def main():
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(threshold=20, precision=4, suppress=True)

    path = sys.argv[1] if len(sys.argv) > 1 else None

    DAENode.init(path)

    cc = ColladaToChroma(DAENode, bvh=False )  
    cc.collada_materials_summary()
    cc.convert_geometry()
    cg = cc.chroma_geometry 

    ls = cc.materialmap['LiquidScintillator']
    gdls = cc.materialmap['GdDopedLS']  

    ccplt_ = lambda mat,prop,color:plt.plot(*cc.materialmap[mat].daeprops[prop].T, color=color, label="%s %s %s" % (mat,prop,color))
    cfplt_ = lambda _:ccplt_('GdDopedLS',_,'r') + ccplt_('LiquidScintillator',_,'b') + [plt.legend()] 

    rso = cc.surfacemap['RSOil']

    self = cc


    log.info("dropping into IPython.embed() try: cg.<TAB> ")
    import IPython 
    IPython.embed()



if __name__ == '__main__':
    main()





