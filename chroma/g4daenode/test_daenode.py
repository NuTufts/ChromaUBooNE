from g4daenode import DAENode
import logging


if __name__ == "__main__":
    testfile = "microboone_nowires.dae" # MicroBooNE geometry with no wires
    logging.basicConfig(filename='log.test_daenode',level=logging.DEBUG)
    print "Running test file: ",testfile
    DAENode.parse( testfile, sens_mats=[] )
