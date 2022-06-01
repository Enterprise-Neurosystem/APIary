#!/usr/bin/python3

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
import sys

def main():
    fname = sys.argv[1]
    samplerate,data = wavfile.read(fname)
    nsamples = 512
    nfolds = data.shape[0]//nsamples
    sz = nfolds*nsamples
    ldata = data[:sz,0].reshape((nfolds,-1))
    rdata = data[:sz,1].reshape((nfolds,-1))
    RDATA = dct(np.column_stack((rdata,np.flip(rdata,axis=1))),type=3,axis=1)
    LDATA = dct(np.column_stack((ldata,np.flip(ldata,axis=1))),type=3,axis=1)
    np.savetxt('%s.rspect'%fname,np.abs(RDATA[:,:nsamples*2:2]).T)
    np.savetxt('%s.lspect'%fname,np.abs(LDATA[:,:nsamples*2:2]).T)
    print(samplerate)

    return

if __name__ == '__main__':
    main()
