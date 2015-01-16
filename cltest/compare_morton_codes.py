import numpy as np

def load_cuda( cutext ):
    fin = open(cutext,'r')
    lines = fin.readlines()
    morton_codes = []
    for n,l in enumerate(lines):
        x = np.uint64( l.strip() )
        morton_codes.append(x)
    return morton_codes

def load_cuda2( cutext ):
    fin = open(cutext,'r')
    lines = fin.readlines()
    morton_codes = []
    qcentroids = []
    for n,l in enumerate(lines):
        if "id " in l and "qcentroid" in l:
            x = np.uint64( l.strip().split()[-1] )
            morton_codes.append(x)
            centroid = l.strip().split("(")[1].split(")")[0].split(",")
            qcentroids.append( [ np.uint32(centroid[0]), np.uint32(centroid[1]), np.uint32(centroid[2]) ] )
    return morton_codes, qcentroids

def load_cl( cltext ):
    fin = open(cltext, 'r')
    lines = fin.readlines()
    morton_codes = []
    qcentroids = []
    for n,l in enumerate(lines):
        if "id: " in l:
            x = np.uint64( l.strip().split()[-1] )
            morton_codes.append(x)
            centroid = l.strip().split("qcent:")[1].split("morton")[0].strip().split(",")
            qcentroids.append( [ np.uint32(centroid[0]), np.uint32(centroid[1]), np.uint32(centroid[2]) ] )
    return morton_codes, qcentroids

if __name__ == "__main__":
    #cucodes = load_cuda( "mortoncodes_cu.txt" )
    cucodes, cuq = load_cuda2( "cuout.txt" )
    clcodes, clq = load_cl( "out" )
    print cucodes[:10]
    print cuq[:10]
    print clcodes[:10]
    print clq[:10]
    raw_input()
    ndiffs = 0
    for i in range(0,len(cucodes)):
        if cucodes[i]!=clcodes[i]:
            print "Mismatch: ID=%d"%(i)," cuda=",cucodes[i]," cl=",clcodes[i]
            print "  CUDA: ",cuq[i], " CL: ",clq[i], " diff=",np.array(cuq[i])-np.array(clq[i])
            ndiffs += 1
    print "Percent different: ",float(ndiffs)/float(len(cucodes))
