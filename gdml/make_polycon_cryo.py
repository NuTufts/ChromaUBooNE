import os,sys
from math import *
import numpy as np

mytubez   = (427.75)*2.54    # cm
mytuber   = 193.0            # cm
myendcapr = 144.5*2.54       # cm

def pos_endcap1():
    return (0,0,427.75*2.54/2- 2.54*sqrt(144.5**2-75.5**2))
def pos_endcap2(tubez, tuber, endcapr, thickness):
    return (0,0, (tubez-2*thickness)/2 -sqrt((endcapr-thickness)**2-(tuber-thickness)**2) )

def get_cryo_r( z, tuber, tubez, endcapr, thickness ):
    if np.fabs(z)<=0.5*(tubez-2*thickness):
        return tuber-thickness
    else:
        dzcenter = fabs(z)-pos_endcap2(tubez, tuber, endcapr, thickness)[2]
        arg = endcapr*endcapr - dzcenter*dzcenter
        if arg<0:
            arg = 0.0
        return sqrt(arg)


def dump_polycone( thickness, tubez, tuber, endcapr, test=False ):
    steps = 10 # on cap
    start = 0
    end = atan2( tuber, 0.5*tubez-pos_endcap2(tubez, tuber, endcapr, thickness )[2] )*180.0/np.pi + 1.0
    zcap = []
    for s in xrange(0,steps):
        deg = 0 + ((end-start)/steps)*s
        z = cos(deg*np.pi/180.0)*(endcapr-thickness) + pos_endcap2(tubez, tuber, endcapr, thickness)[2]
        zcap.append(z)
    # First top cap
    for z in zcap:
        r = max(get_cryo_r(z, tuber, tubez, endcapr, thickness),0)
        rmin = 0.0
        if test==True:
            rmin = r-0.001
        print "  <zplane rmin=\"%.4f\" rmax=\"%.4f\" z=\"%.4f\"/>"%(rmin, r, z)
    r = get_cryo_r(tubez*0.5-thickness, tuber, tubez, endcapr, thickness)
    if test:
        rmin = r-0.001
    print "  <zplane rmin=\"%.4f\" rmax=\"%.4f\" z=\"%.4f\"/>"%( rmin, r, tubez*0.5)
    r = get_cryo_r(0, tuber, tubez, endcapr, thickness)
    if test:
        rmin = r-0.001
    print "  <zplane rmin=\"%.4f\" rmax=\"%.4f\" z=\"%.4f\"/>"%( rmin, r, 0.0)
    r = get_cryo_r(-tubez*0.5+thickness, tuber, tubez, endcapr, thickness)
    if test:
        rmin = r-0.001
    print "  <zplane rmin=\"%.4f\" rmax=\"%.4f\" z=\"%.4f\"/>"%( rmin, r, -tubez*0.5)

    zcap.reverse()
    for z in zcap:
        r = max(get_cryo_r(-z, tuber, tubez, endcapr, thickness),0)
        if test:
            rmin = r-0.001
        print "  <zplane rmin=\"%.4f\" rmax=\"%.4f\" z=\"%.4f\"/>"%( rmin, r, -z)

print " Outer Edge "
dump_polycone(0.0, mytubez, mytuber, myendcapr)
print" Inner Edge "
dump_polycone(2.54, mytubez, mytuber, myendcapr)

print " TEST Outer Edge"
dump_polycone(0.0, mytubez, mytuber, myendcapr, True)
print " TEST Inner Edge"
dump_polycone(2.54, mytubez, mytuber, myendcapr, True)
