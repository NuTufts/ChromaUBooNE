from unittest_find import unittest
import os
os.environ["PYOPENCL_CTX"] ='1'
import chroma.api as api
import chroma.gpu.cltools as cltools
import chroma.gpu.clrandstate as clrand
import pyopencl as cl

import ROOT as rt

class TestRandomGen( unittest.TestCase ):
    def setUp(self):
        self.context = cltools.get_last_context()
        self.nthreads_per_block = 1028
        self.blocks_per_iter = 1
        self.seed = 1

        self.nthreads = self.nthreads_per_block*self.blocks_per_iter

    def testRNG(self):
        states = clrand.get_rng_states( self.context, 10000, seed=0 )
        array = clrand.fill_array( self.context, states, 10000 )
        
        out = rt.TFile("output_testRNG.root","recreate")
        hout = rt.TH1D("hrand","",1000, -2, 2 )
        for a in array:
            hout.Fill(a)
        out.Write()
        out.Close()

    def tearDown(self):
        pass

if __name__ == "__main__":
    api.use_opencl()
    unittest.main()

