from unittest_find import unittest
import os
os.environ["PYOPENCL_CTX"] ='1'
import chroma.api as api
from chroma.gpu.tools import get_module, api_options, chunk_iterator
import chroma.gpu.cltools as cltools
import chroma.gpu.clrandstate as clrand
from  chroma.gpu.gpufuncs import GPUFuncs
import pyopencl as cl
import ROOT as rt
import numpy as np

class TestSampling(unittest.TestCase):
    def setUp(self):
        self.context = cltools.get_last_context()
        self.nthreads_per_block = 256
        self.myoptions = ('-I.',)+api_options
        self.mod = get_module( "test_sample_cdf.cl", self.context, options=self.myoptions, include_source_directory=True)
        self.funcs = GPUFuncs(self.mod)
        self.rng_states = clrand.get_rng_states( self.context, self.nthreads_per_block )
        self.outf = rt.TFile("output_sample_cdf.root", "RECREATE")

    def compare_sampling(self, hist, reps=10):
        queue = cl.CommandQueue(self.context)

        # make cdf histogram
        nbins = hist.GetNbinsX();
        xaxis = hist.GetXaxis()
        intg = hist.GetIntegral()
        cdf_y = np.empty(nbins+1, dtype=float)
        cdf_x = np.empty_like(cdf_y)

        cdf_x[0] = xaxis.GetBinLowEdge(1)
        cdf_y[0] = 0.0
        for i in xrange(1,len(cdf_x)):
            cdf_y[i] = intg[i]
            cdf_x[i] = xaxis.GetBinUpEdge(i)

        cdf_x_gpu = cl.array.to_device(queue, cdf_x.astype(np.float32))
        cdf_y_gpu = cl.array.to_device(queue, cdf_y.astype(np.float32))
        block =(self.nthreads_per_block,1,1)
        grid = (1, 1)
        out_gpu = cl.array.empty(queue, shape=self.nthreads_per_block, dtype=np.float32)

        out_h = rt.TH1D('out_h', '', hist.GetNbinsX(), xaxis.GetXmin(),xaxis.GetXmax())
        out_h.SetLineColor(rt.kGreen)

        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(reps, self.nthreads_per_block, max_blocks=1):
            self.funcs.test_sample_cdf(queue, (elements_this_iter, 1, 1), None,
                                       self.rng_states.data,
                                       np.int32(len(cdf_x_gpu)), 
                                       cdf_x_gpu.data, cdf_y_gpu.data, out_gpu.data)
            out = out_gpu.get()
            for v in out[:elements_this_iter]:
                out_h.Fill(v)

        prob = out_h.KolmogorovTest(hist)
        out_h.Write()
        return prob, out_h 

    def test_sampling(self):
        '''Verify that the CDF-based sampler on the GPU reproduces a binned
        Gaussian distribution'''
        f = rt.TF1('f_gaussian', 'gaus(0)', -5, 5)
        f.SetParameters(1.0/np.sqrt(np.pi * 2), 0.0, 1.0)
        gaussian = rt.TH1D('gaussian', '', 100, -5, 5)
        gaussian.Add(f)

        prob, out_h = self.compare_sampling(gaussian, reps=20000)

        self.outf.cd()
        gaussian.Write("gaussian")
        out_h.Write("out_h")
        assert prob > 0.01

    def tearDown(self):
        self.outf.Close()

if __name__ == "__main__":
    api.use_opencl()
    unittest.main()
