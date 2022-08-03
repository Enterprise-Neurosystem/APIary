#!/usr/bin/python3

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
import sys
import wave

def main():
    if len(sys.argv)<3:
        print("syntax:./src/make_spectrogram.py <path/file.WAV> <'bee'|'server'>")
    fname = sys.argv[1]
    nsamples = 1<<12
    nfolds = 1<<14
    scale = 1<<12
    flim = 12
        if sys.argv[2]=='bee':
            nsamples = 1<<14
            nfolds = 1<<12
            scale = 1<<10
        elif sys.argv[2]=='server':
            nsamples = 1<<12
            nfolds = 1<<14
            scale = 1<<12

            
    dt = np.dtype(np.int16).newbyteorder('<')
    data = []
    with wave.open(fname,'r') as f:
        print(f.getparams())
        if f.getnchannels()>1:
            print('Ahhh, processing more than one channel!  Returning!!')
            f.close()
            return
        samplewidth = f.getsampwidth()
        totframes = f.getnframes()
        while f.tell()<min(totframes - nsamples*samplewidth,nsamples*nfolds*samplewidth):
            tmp = np.frombuffer(f.readframes(nsamples*samplewidth),dtype=dt)
            data += [np.abs(dct(np.concatenate((tmp,np.flip(tmp,axis=0))),type=2,axis=0)[:1<<flim:2])//scale]
            #data += [np.abs(dct(np.concatenate((tmp,np.flip(tmp,axis=0))),type=2,axis=0)[::2])//scale]
    np.savetxt('%s.sspect'%(fname),np.array(data).T,fmt='%i',header='remember for %s, highest frequency is %i\% of the 96kHz (only showing 2**%i of 2**%i samples)'%(sys.argv[2],int(100*float(flim)/float(nsamples)),flim,nsamples) )
    return

    '''
    samplerate,data = wavfile.read(fname)
    print(samplerate)
    nsamples = 1<<14 
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
    '''

    return

if __name__ == '__main__':
    main()
