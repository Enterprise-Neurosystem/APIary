#!/usr/bin/python3

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
import sys
import wave

def main():
    if len(sys.argv)<3:
        print("syntax:./src/make_spectrogram.py <path/file.WAV> <'bee'|'server'|'todd'>")
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
    elif sys.argv[2]=='todd':
        nsamples = 1<<14
        nfolds = 1<<14
        scale = 1<<10

    P = (1<<flim)>>2 # setting to 1/8 of frequency limits
    P_offset = 0
    P_filt = np.array([0.5 * (1 - np.cos(2*np.pi * x / P)) for x in range(P)])
            
    dt = np.dtype(np.int16).newbyteorder('<')
    data = {} 
    filtdata = {} 
    nchans = int(1)
    with wave.open(fname,'r') as f:
        print(f.getparams())
        samplewidth = f.getsampwidth()
        totframes = f.getnframes()
        nchans = f.getnchannels()
        for c in range(nchans):
            data['ch%i'%c] = []
            filtdata['ch%i'%c] = []
        while f.tell()<min(totframes - nsamples*samplewidth*nchans,nchans*nsamples*nfolds*samplewidth):
            tmp = np.frombuffer(f.readframes(nsamples*samplewidth*nchans),dtype=dt)
            for c in range(nchans):
                data['ch%i'%c] += [np.abs(dct(np.concatenate((tmp[c::nchans],np.flip(tmp[c::nchans],axis=0))),type=2,axis=0)[:1<<flim:2])//scale]
                cepstrum = dct(np.concatenate((data['ch%i'%c][-1],np.flip(data['ch%i'%c][-1]))),axis=0,type=2)
                cepstrum[P_offset:P_offset+2*P:2] *= P_filt
                cepstrum[P_offset+2*P:] *= 0.0
                cepstrum[:P_offset] *= 0.0
                back = dct(cepstrum,type=3,axis=0).real[:1<<flim]//scale
                back *= (back>0)
                back += 1
                thresh = 14
                width = 1 
                #filtdata['ch%i'%c] += [np.log2(back)]
                filtdata['ch%i'%c] += [(1+np.tanh((np.log2(back)-thresh)/width))/2]
                #filtdata['ch%i'%c] += [np.ones(back.shape)*(np.log2(back)>thresh)]
                #filtdata['ch%i'%c] += [np.tanh((back-thresh)/width)]
                

    for k in data.keys():
        np.savetxt('%s.%s.sspect'%(fname,k),np.array(data[k]).T,fmt='%i',header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(sys.argv[2],int(100*float(flim)/float(nsamples)),flim,np.log2(nsamples)) )
        np.savetxt('%s.%s.sspect_filt'%(fname,k),np.array(filtdata[k]).T,fmt='%.1f',header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(sys.argv[2],int(100*float(flim)/float(nsamples)),flim,np.log2(nsamples)) )
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
