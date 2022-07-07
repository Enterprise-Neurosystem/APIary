#!/usr/bin/python3

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
import sys

def main():
    fname = sys.argv[1]
    samplerate,data = wavfile.read(fname)
    nsamples = 1<<12 
    nfolds = data.shape[0]>>10
    nfolds = 1<<12
    sz = nfolds*nsamples
    if len(data.shape)>1:
        ldata = data[:sz,0].reshape((nfolds,-1))
        rdata = data[:sz,1].reshape((nfolds,-1))
        RDATA = dct(np.column_stack((rdata,np.flip(rdata,axis=1))),type=2,axis=1)
        LDATA = dct(np.column_stack((ldata,np.flip(ldata,axis=1))),type=2,axis=1)
        np.savetxt('%s.rspect'%fname,np.abs(RDATA[:,:nsamples*2:2]).T)
        np.savetxt('%s.lspect'%fname,np.abs(LDATA[:,:nsamples*2:2]).T)
    else:
        sdata = data[:sz].reshape((nfolds,nsamples)).T
        #sdata = data[:sz].reshape((nsamples,nfolds))
        SDATA = dct(np.row_stack((sdata,np.flip(sdata,axis=0))),type=2,axis=0)
        np.savetxt('%s.sspect'%(fname),np.abs(SDATA[::2,:]),fmt='%.3f')
    print(samplerate)

    return

if __name__ == '__main__':
    main()
