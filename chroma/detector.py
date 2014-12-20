import os
import logging
log = logging.getLogger(__name__)
import numpy as np

from chroma.geometry import Geometry

class Detector(Geometry):
    '''A Detector is a subclass of Geometry that allows some Solids
    to be marked as photon detectors, which we will suggestively call
    "PMTs."  Each detector is imagined to be connected to an electronics
    channel that records a hit time and charge.

    Each PMT has two integers identifying it: a channel index and a
    channel ID.  When all of the channels in the detector are stored
    in a Numpy array, they will be stored in index order.  Channel
    indices star from zero and have no gaps.  Channel ID numbers are
    arbitrary integers that identify a PMT uniquely in a stable way.
    They are written out to disk when using the Chroma ROOT format,
    and are used when reading events back in to map channels back
    into the correct array index.

    For now, all the PMTs share a single set of time and charge
    distributions.  In the future, this will be generalized to
    allow per-channel distributions.
    '''
    cache_skips = ['solids','solid_displacements','solid_rotations','cache_skips']

    @classmethod
    def get(cls, cachedir):
        if os.path.isdir(cachedir):
            log.info("load chroma_geometry from %s " % cachedir )
            from chroma.loader import load_bvh
            instance = cls.load_instance(cachedir)
            instance.bvh = load_bvh(instance)
        else:
            instance = None
        pass
        return instance

    def __init__(self, detector_material=None):
        Geometry.__init__(self, detector_material=detector_material)

        # Using numpy arrays here to allow for fancy indexing
        self.solid_id_to_channel_index = np.zeros(0, dtype=np.int32)
        self.channel_index_to_solid_id = np.zeros(0, dtype=np.int32)

        self.channel_index_to_channel_id = np.zeros(0, dtype=np.int32)
        self.solid_id_to_channel_id  = np.zeros(0, dtype=np.int32)

        # If the ID numbers are arbitrary, we can't treat them
        # as array indices, so have to use a dictionary
        self.channel_id_to_channel_index = {}

        # zero time and unit charge distributions
        self.time_cdf = (np.array([-0.01, 0.01]), np.array([0.0, 1.0]))
        self.charge_cdf = (np.array([0.99, 1.00]), np.array([0.0, 1.0]))

    def add_solid(self, solid, rotation=None, displacement=None):
        """
        Add the solid `solid` to the geometry. When building the final triangle
        mesh, `solid` will be placed by rotating it with the rotation matrix
        `rotation` and displacing it by the vector `displacement`.
        """
        solid_id = Geometry.add_solid(self, solid=solid, rotation=rotation, 
                                      displacement=displacement)
        self.solid_id_to_channel_index.resize(solid_id+1)
        self.solid_id_to_channel_index[solid_id] = -1 # solid maps to no channel

        self.solid_id_to_channel_id.resize(solid_id+1)
        self.solid_id_to_channel_id[solid_id] = -1 # solid maps to no channel

        return solid_id

    def add_pmt(self, pmt, rotation=None, displacement=None, channel_id=None):
        """Add the PMT `pmt` to the geometry. When building the final triangle
        mesh, `solid` will be placed by rotating it with the rotation matrix
        `rotation` and displacing it by the vector `displacement`, just like
        add_solid().

            `pmt``: instance of chroma.Solid
                Solid representing a PMT.
            `rotation`: numpy.matrix (3x3)
                Rotation to apply to PMT mesh before displacement.  Defaults to
                identity rotation.
            `displacement`: numpy.ndarray (shape=3)
                3-vector displacement to apply to PMT mesh after rotation.
                Defaults to zero vector.
            `channel_id`: int
                Integer identifier for this PMT.  May be any integer, with no
                requirement for consective numbering.  Defaults to None,
                where the ID number will be set to the generated channel index.
                The channel_id must be representable as a 32-bit integer.
        
            Returns: dictionary { 'solid_id' : solid_id, 
                                  'channel_index' : channel_index,
                                  'channel_id' : channel_id }
        """

        solid_id = self.add_solid(solid=pmt, rotation=rotation, 
                                  displacement=displacement)

        channel_index = len(self.channel_index_to_solid_id)
        if channel_id is None:
            channel_id = channel_index

        # add_solid resized this array already
        self.solid_id_to_channel_index[solid_id] = channel_index
        self.solid_id_to_channel_id[solid_id] = channel_id 


        # resize channel_index arrays before filling
        self.channel_index_to_solid_id.resize(channel_index+1)
        self.channel_index_to_solid_id[channel_index] = solid_id
        self.channel_index_to_channel_id.resize(channel_index+1)
        self.channel_index_to_channel_id[channel_index] = channel_id

        # dictionary does not need resizing
        self.channel_id_to_channel_index[channel_id] = channel_index
        
        return { 'solid_id' : solid_id, 
                 'channel_index' : channel_index,
                 'channel_id' : channel_id }
                           
    def _pdf_to_cdf(self, bin_edges, bin_contents):
        '''Returns tuple of arrays `(cdf_x, cdf_y)` corresponding to
        the cumulative distribution function for a binned PDF with the
        given bin_edges and bin_contents.

           `bin_edges`: array
               bin edges of PDF.  length is len(bin_contents) + 1
           `bin_contents`: array
               The contents of each bin.  The array should NOT be
               normalized for bin width if variable binning is present.
        '''
        cdf_x = np.copy(bin_edges)
        cdf_y = np.array([0.0]+bin_contents.cumsum())
        cdf_y /= cdf_y[-1] # normalize
        return (cdf_x, cdf_y)

    def set_time_dist_gaussian(self, rms, lo, hi, nsamples=50):
        pdf_x = np.linspace(lo, hi, nsamples, endpoint=True)
        pdf_y = np.exp(-0.5 * (pdf_x/rms)**2)
        self.time_cdf = self._pdf_to_cdf(pdf_x, pdf_y)

    def set_time_dist(self, bin_edges, bin_contents):
        '''Set the time PDF directly from a histogram.

          `bin_edges`: array
            bin edges of PDF.  length is len(bin_contents) + 1
          `bin_contents`: array
            The contents of each bin in the histogram.  The array
            should not be normalized for bin width if variable bin
            widths are present.
        '''
        self.time_cdf = self._pdf_to_cdf(bin_edges, bin_contents)

    def set_charge_dist_gaussian(self, mean, rms, lo, hi, nsamples=50):
        pdf_x = np.linspace(lo, hi, nsamples, endpoint=True)
        pdf_y = np.exp(-0.5 * ((pdf_x - mean)/rms)**2)
        self.charge_cdf = self._pdf_to_cdf(pdf_x, pdf_y)

    def num_channels(self):
        return len(self.channel_index_to_channel_id)
