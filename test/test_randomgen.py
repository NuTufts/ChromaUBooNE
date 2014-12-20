from unittest_find import unittest
import chroma.gpu.tools as gputools
import pycuda.driver as cuda
import ROOT as rt

class TestRandomGen( unittest.TestCase ):
    def setUp(self):
        self.context = gputools.create_cuda_context()
        self.nthreads_per_block = 64
        self.blocks_per_iter = 16
        self.seed = 1

        self.nthreads = self.nthreads_per_block*self.blocks_per_iter

    def testRNG(self):
        f = rt.TFile("out_test_randomgen.root","RECREATE")
        hout = rt.TH1D("hout","output of curand_uniform",100,0,1.0)

        rng_states = gputools.get_rng_states( self.nthreads, self.seed )
        for rep in range(0,10):
            outarray = gputools.get_random_array( self.nthreads, rng_states )
            print outarray[:10]
            for i in range(0,self.nthreads):
                hout.Fill(outarray[i])

        hout.Write()
        f.Close()

    def tearDown(self):
        self.context.pop()

if __name__ == "__main__":

    unittest.main()

