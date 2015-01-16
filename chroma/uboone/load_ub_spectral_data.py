import os,sys
import numpy as np

def load_hist_data( datfile, xmin, xmax, sep="\t" ):
    f = open(datfile,'r')
    lines = f.readlines()
    data = []
    avediff = 0
    lastx = 0
    for l in lines:
        point = l.strip().split(sep)
        if len(point)==2:
            wavelength = float(point[0])
            if len(data)>0:
                avediff += (wavelength-lastx)
            pdensity   = float(point[1])
            data.append( (wavelength, pdensity) )
            lastx = wavelength
    avediff /= len(data)-1
    print avediff
    npts = len(data)
    binsize = float( xmax-xmin )/float(npts)
    wavelengths = np.array( zip(np.arange( xmin, xmax+0.001, binsize ).ravel(), np.zeros(npts+1).ravel() ) )
    print npts+1,binsize
    data.append( (xmax,0) )
    data = np.array( data )
    for x in xrange(0,len(data[:-1])):
        #index = np.searchsorted( wavelengths[:,0], data[x,0] )-1
        index = x
        print index, data[x,0], wavelengths[index,0], wavelengths[index+1,0], data[x,1]
        wavelengths[index,1] = data[x,1]*binsize
    wavelengths[:,1] = np.where( wavelengths[:,1]<0, 0.0, wavelengths[:,1] )
    tot = np.sum( wavelengths[:,1] )
    wavelengths[:,1] /= tot
    return wavelengths

if __name__ == "__main__":
    data = load_hist_data( "raw_tpb_emission.dat", 350, 640 )

    try: 
        import ROOT as rt
        from array import array
    except:
        sys.exit(0)
    # plot hist
    print data[:,0].ravel()    
    h = rt.TH1D('hdata','Data', len(data)-1, array('f',data[:,0].ravel()) )
    for pt in data:
        h.SetBinContent( h.FindBin( pt[0]+0.01 ), pt[1] )
    print h.Integral()
    c = rt.TCanvas("c","c",800,600)
    c.Draw()
    h.Draw()
    c.Update()
    raw_input()
    
