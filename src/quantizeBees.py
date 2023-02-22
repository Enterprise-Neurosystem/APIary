#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import re

from Quantizers import Quantizer

def m2s(x):
    return 60.*x

def s2m(x):
    return x/60.

def main(filehead,nbins,tlow,thigh):
    inname = '%s.h5'%filehead
    quants = {}
    qHists = {}
    with h5py.File(inname,'r') as f:
        ckeys = list(f.keys())
        _ = [print(ck) for ck in ckeys]

        for ck in ckeys:
            quants[ck] = Quantizer('bees',nbins=nbins)
            quants[ck].setbins(data=f[ck]['edges'][()])
            qHists[ck] = []
            print('running %i-%i out of %i slices of 1s spectra'%(tlow,thigh,f[ck]['addresses'].shape[0]))
    
            #for frame in range(f[ck]['addresses'].shape[0]):
            for frame in range(tlow,thigh):
                a = f[ck]['addresses'][()][frame]
                n = f[ck]['nedges'][()][frame]
                qHists[ck] += [quants[ck].histogram(data = f[ck]['edges'][()][a:a+n])]
            fig,ax = plt.subplots(2,1,sharex=True,figsize=(12,6))
            im = ax[0].pcolor(np.array(qHists[ck]).T[:,:thigh-tlow],vmin=0,vmax=1)
            plt.colorbar(im,ax=ax[0])
            secax = ax[0].secondary_xaxis('top',functions=(s2m,m2s))
            secax.set_xlabel('time [min]')
            ax[0].set_ylabel('quant bins')
            im = ax[1].pcolor(f[ck]['spect'][()].T[:1024,tlow:thigh],vmin=0,vmax=1<<9)
            ax[1].set_xlabel('time [sec]')
            ax[1].set_ylabel('f [Hz]')
            plt.colorbar(im,ax=ax[1])
            plt.savefig('%s.quantVspect_%i-%i.png'%(filehead,tlow,thigh))
            plt.show()
            #secax = ax.secondary_xaxis('top', functions=(forward, inverse))
            #secax.set_xlabel('period [s]')

    return

if __name__ == '__main__':
    if len(sys.argv)<5:
        print('syntax: src/quantizeBees.py <nqbins> <tlow> <thigh> <h5 filename>')
    else:
        m = re.search('^(.*)\.h5',sys.argv[-1])
        if m:
            filehead = m.group(1)
            main(filehead,nbins=int(sys.argv[1]),tlow=int(sys.argv[2]),thigh=int(sys.argv[3]))

