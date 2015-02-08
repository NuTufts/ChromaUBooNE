#!/usr/bin/env python
import numpy as np


# Photon history bits (see photon.h for source)
NO_HIT           = 0x1 << 0
BULK_ABSORB      = 0x1 << 1
SURFACE_DETECT   = 0x1 << 2
SURFACE_ABSORB   = 0x1 << 3
RAYLEIGH_SCATTER = 0x1 << 4
REFLECT_DIFFUSE  = 0x1 << 5
REFLECT_SPECULAR = 0x1 << 6
SURFACE_REEMIT   = 0x1 << 7
SURFACE_TRANSMIT = 0x1 << 8
BULK_REEMIT      = 0x1 << 9
WIREPLANE_TRANS  = 0x1 << 10
WIREPLANE_ABSORB = 0x1 << 11
NAN_ABORT        = 0x1 << 31

PHOTON_FLAGS =  {
            'NO_HIT':NO_HIT,
       'BULK_ABSORB':BULK_ABSORB,
    'SURFACE_DETECT':SURFACE_DETECT,
    'SURFACE_ABSORB':SURFACE_ABSORB,
  'RAYLEIGH_SCATTER':RAYLEIGH_SCATTER,
   'REFLECT_DIFFUSE':REFLECT_DIFFUSE,
  'REFLECT_SPECULAR':REFLECT_SPECULAR,
    'SURFACE_REEMIT':SURFACE_REEMIT,
  'SURFACE_TRANSMIT':SURFACE_TRANSMIT,
       'BULK_REEMIT':BULK_REEMIT,
   'WIREPLANE_TRANS':WIREPLANE_TRANS,
  'WIREPLANE_ABSORB':WIREPLANE_ABSORB,
         'NAN_ABORT':NAN_ABORT,
            }

def arg2mask_( argl ):
    """
    Convert comma delimited strings like the below into OR mask integer:

    #. "BULK_ABSORB,SURFACE_DETECT" 
    #. "SURFACE_REEMIT" 
    """
    mask = 0
    for arg in argl.split(","):
        if arg in PHOTON_FLAGS:
           mask |= PHOTON_FLAGS[arg]
        pass 
    return mask


def mask2arg_( mask ):
    return ",".join(filter(lambda name:mask & PHOTON_FLAGS[name],PHOTON_FLAGS))
 

def count_unique(vals):
    """ 
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T



class Vertex(object):
    def __init__(self, particle_name, pos, dir, ke, t0=0.0, pol=None):
        '''Create a particle vertex.

           particle_name: string
               Name of particle, following the GEANT4 convention.  
               Examples: e-, e+, gamma, mu-, mu+, pi0

           pos: array-like object, length 3
               Position of particle vertex (mm)

           dir: array-like object, length 3
               Normalized direction vector

           ke: float
               Kinetic energy (MeV)

           t0: float
               Initial time of particle (ns)
               
           pol: array-like object, length 3
               Normalized polarization vector.  By default, set to None,
               and the particle is treated as having a random polarization.
        '''
        self.particle_name = particle_name
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.ke = ke
        self.t0 = t0

class Photons(object):
    """
    A few enhancements: 

    #. from_cpl classmethod 
    #. timesorting
    #. dump
    """

    @classmethod
    def from_obj(cls, obj, extend=True ):
        if isinstance(obj, np.ndarray):
            return cls.from_npl( obj, extend=extend)
        else:
            return cls.from_cpl( obj, extend=extend)
        pass

    @classmethod
    def from_npl(cls, a, extend=True):
        """
        Routing NPL via this Photons class 
        is a stop gap until find more direct route 
        As afforded by it being a numpy array already 
        """
        pos, t           = a[:,0,:3], a[:,0,3]
        dir, wavelengths = a[:,1,:3], a[:,1,3]
        pol, weights     = a[:,2,:3], a[:,2,3]

        flgs = a[:,3,:3].view(np.int32)
        pmtid = a[:,3,3].view(np.int32)

        obj = cls(pos,dir,pol,wavelengths,t)   
  
        if extend:
            order = np.argsort(obj.t)
            obj.sort(order)        
            obj.pmtid = pmtid[order]
            obj.flgs  = flgs[order]
        pass
        return obj





    def as_npl(self, hit=False, directpropagator=False):
        """
        Applying selection based on pmtid, could follow pattern::
    
           a = np.arange(10*4*4).reshape(10,4,4)  
           b = a[np.where( a[:,3,3] > 111 )]      # shape (3,4,4)
           b = a[a[:,3,3] > 111]    # more tersely 

        TODO: pull pmtid and flgs out of the kernel, to allow hit=True
        """
        nall = len(self.pos)
        a = np.zeros( (nall,4,4), dtype=np.float32 )       
 
        a[:,0,:3] = self.pos
        a[:,0, 3] = self.t 

        a[:,1,:3] = self.dir
        a[:,1, 3] = self.wavelengths

        a[:,2,:3] = self.pol
        a[:,2, 3] = np.ones(nall, dtype=a.dtype) #self.weights 

        lht = self.last_hit_triangles
        flg = self.flags
        assert len(lht) == len(flg)
        pmtid = np.zeros( nall, dtype=np.int32 )

        # a kludge setting of pmtid into lht using the map argument of propagate_hit.cu 
        if directpropagator:
            SURFACE_DETECT = 0x1 << 2
            detected = np.where( flg & SURFACE_DETECT  )
            pmtid[detected] = lht[detected]  # sparsely populate, leaving zeros for undetected
        else:
            pass

        a[:,3, 0] = np.arange(nall, dtype=np.int32).view(a.dtype)  # photon_id
        a[:,3, 1] = 0                                              # used in comparison againt vbo prop
        a[:,3, 2] = self.flags.view(a.dtype)                       # history flags 
        a[:,3, 3] = pmtid.view(a.dtype)                            # channel_id ie PmtId

        if hit:
            return a[pmtid > 0]
        else:
            return a  
        pass

    @classmethod
    def from_cpl(cls, cpl, extend=True ):
        """
        :param cpl: ChromaPhotonList instance, as obtained from file or MQ
        :param extend: when True add pmtid attribute and sort by time
        """ 
        xyz_ = lambda x,y,z,dtype:np.column_stack((np.array(x,dtype=dtype),np.array(y,dtype=dtype),np.array(z,dtype=dtype))) 

        pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        wavelengths = np.array(cpl.wavelength, dtype=np.float32)
        t = np.array(cpl.t, dtype=np.float32)
        pass
        obj = cls(pos,dir,pol,wavelengths,t)   
       
        if extend:
            order = np.argsort(obj.t)
            pmtid = np.array(cpl.pmtid, dtype=np.int32)
            obj.sort(order)        
            obj.pmtid = pmtid[order]
        pass
        return obj

    atts = "pos dir pol wavelengths t last_hit_triangles flags weights".split()

    def dump(self):
        for att in self.atts:
            print "%s\n%s" % ( att, getattr(self, att))
        pass
        print "history for all photons" 
        self.dump_history( self.flags )

        inf_wavelengths = np.where( self.wavelengths == np.inf )[0]
        print "history for inf wavelength photons" 
        self.dump_history( self.flags[inf_wavelengths] )

        bulk_reemit = np.where( self.flags & BULK_REEMIT )[0]
        print "history for BULK_REEMIT photons " 
        self.dump_history( self.flags[bulk_reemit] )

        print np.all( bulk_reemit == inf_wavelengths )


    def sort(self, order):
        self.pos = self.pos[order]
        self.dir = self.dir[order] 
        self.pol = self.pol[order] 
        self.wavelengths = self.wavelengths[order]
        self.t = self.t[order]

    def history(self, flags=None):
        if flags is None:
            flags = self.flags
        return len(flags),count_unique(flags)

    def dump_history(self, flags=None):
        if flags is None:
            flags = self.flags
        nflag, history = self.history(flags)
        print "dump_history from %s flags " % nflag
        for mask, count in sorted(history,key=lambda _:_[1], reverse=True):
            print "[0x%x] %d (%5.2f): %s " % (mask,count,float(count)/nflag,mask2arg_(mask))
        pass


    def __init__(self, pos, dir, pol, wavelengths, t=None, last_hit_triangles=None, flags=None, weights=None):
        '''Create a new list of n photons.

            pos: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Position 3-vectors (mm)

            dir: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Direction 3-vectors (normalized)

            pol: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Polarization direction 3-vectors (normalized)

            wavelengths: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon wavelengths (nm)

            t: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon times (ns)

            last_hit_triangles: numpy.ndarray(dtype=numpy.int32, shape=n)
               ID number of last intersected triangle.  -1 if no triangle hit in last step
               If set to None, a default array filled with -1 is created

            flags: numpy.ndarray(dtype=numpy.uint32, shape=n)
               Bit-field indicating the physics interaction history of the photon.  See 
               history bit constants in chroma.event for definition.

            weights: numpy.ndarray(dtype=numpy.float32, shape=n)
               Survival probability for each photon.  Used by 
               photon propagation code when computing likelihood functions.
        '''
        self.pos = np.asarray(pos, dtype=np.float32)
        self.dir = np.asarray(dir, dtype=np.float32)
        self.pol = np.asarray(pol, dtype=np.float32)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)

        if t is None:
            self.t = np.zeros(len(pos), dtype=np.float32)
        else:
            self.t = np.asarray(t, dtype=np.float32)

        if last_hit_triangles is None:
            self.last_hit_triangles = np.empty(len(pos), dtype=np.int32)
            self.last_hit_triangles.fill(-1)
        else:
            self.last_hit_triangles = np.asarray(last_hit_triangles,
                                                 dtype=np.int32)

        if flags is None:
            self.flags = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.flags = np.asarray(flags, dtype=np.uint32)

        if weights is None:
            self.weights = np.ones(len(pos), dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)

    def __add__(self, other):
        '''Concatenate two Photons objects into one list of photons.

           other: chroma.event.Photons
              List of photons to add to self.

           Returns: new instance of chroma.event.Photons containing the photons in self and other.
        '''
        pos = np.concatenate((self.pos, other.pos))
        dir = np.concatenate((self.dir, other.dir))
        pol = np.concatenate((self.pol, other.pol))
        wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        t = np.concatenate((self.t, other.t))
        last_hit_triangles = np.concatenate((self.last_hit_triangles, other.last_hit_triangles))
        flags = np.concatenate((self.flags, other.flags))
        weights = np.concatenate((self.weights, other.weights))
        return Photons(pos, dir, pol, wavelengths, t,
                       last_hit_triangles, flags, weights)

    def __len__(self):
        '''Returns the number of photons in self.'''
        return len(self.pos)

    def __getitem__(self, key):
        return Photons(self.pos[key], self.dir[key], self.pol[key],
                       self.wavelengths[key], self.t[key],
                       self.last_hit_triangles[key], self.flags[key],
                       self.weights[key])

    def reduced(self, reduction_factor=1.0):
        '''Return a new Photons object with approximately
        len(self)*reduction_factor photons.  Photons are selected
        randomly.'''
        n = len(self)
        choice = np.random.permutation(n)[:int(n*reduction_factor)]
        return self[choice]

class Channels(object):
    def __init__(self, hit, t, q, flags=None):
        '''Create a list of n channels.  All channels in the detector must 
        be included, regardless of whether they were hit.

           hit: numpy.ndarray(dtype=bool, shape=n)
             Hit state of each channel.

           t: numpy.ndarray(dtype=numpy.float32, shape=n)
             Hit time of each channel. (ns)

           q: numpy.ndarray(dtype=numpy.float32, shape=n)
             Integrated charge from hit.  (units same as charge 
             distribution in detector definition)
        '''
        self.hit = hit
        self.t = t
        self.q = q
        self.flags = flags

    def hit_channels(self):
        '''Extract a list of hit channels.
        
        Returns: array of hit channel IDs, array of hit times, array of charges on hit channels
        '''
        return self.hit.nonzero(), self.t[self.hit], self.q[self.hit]

class Event(object):
    def __init__(self, id=0, primary_vertex=None, vertices=None, photons_beg=None, photons_end=None, channels=None):
        '''Create an event.

            id: int
              ID number of this event

            primary_vertex: chroma.event.Vertex
              Vertex information for primary generating particle.
              
            vertices: list of chroma.event.Vertex objects
              Starting vertices to propagate in this event.  By default
              this is the primary vertex, but complex interactions
              can be representing by putting vertices for the
              outgoing products in this list.

            photons_beg: chroma.event.Photons
              Set of initial photon vertices in this event

            photons_end: chroma.event.Photons
              Set of final photon vertices in this event

            channels: chroma.event.Channels
              Electronics channel readout information.  Every channel
              should be included, with hit or not hit status indicated
              by the channels.hit flags.
        '''
        self.id = id

        self.nphotons = None

        self.primary_vertex = primary_vertex

        if vertices is not None:
            if np.iterable(vertices):
                self.vertices = vertices
            else:
                self.vertices = [vertices]
        else:
            self.vertices = []

        self.photons_beg = photons_beg
        self.photons_end = photons_end
        self.channels = channels





def check_photon_flags():
    for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
        val = PHOTON_FLAGS[name]
        mask = arg2mask_(name)
        assert mask == val, (mask, val)
        print "%-30s  %12d   0x%0x      " % ( name, mask, mask )

def check_photon_flags_cumulative():
    names = []
    vals = []
    for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
        val = PHOTON_FLAGS[name]
        names.append(name)
        vals.append(val)

        cnames = ",".join(names)
        cmask = arg2mask_(cnames)
        cval  = reduce(lambda a,b:a|b, vals ) # cumulative bitwise OR
        print cnames, cmask, cval
        assert cmask == cval 


if __name__ == '__main__':
    check_photon_flags()
    check_photon_flags_cumulative()








