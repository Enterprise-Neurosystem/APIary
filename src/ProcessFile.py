#!/usr/bin/python3

import numpy as np
import wave
from scipy.fft import dct,rfft,irfft
from scipy.io import wavfile
import Params
from utils import randomround
import h5py
import math

def scanedges(params,k):
    d = params.filtdata[k][-1]
    e = []
    s = []
    sz = d.shape[0]
    i:int = int(10)
    while i < sz-10:
        while d[i] < params.logicthresh:
            i += 1
            if i==sz-10: return e,s,len(e)
        while i<sz-10 and d[i]>0:
            i += 1
        stop = i
        ''' dx / (Dy) = dx2/dy2 ; dy2*dx/Dy - dx2 ; x2-dx2 = stop - dy2*1/Dy'''
        x0 = float(stop) - float(d[stop])/float(d[stop]-d[stop-1])
        i += 1
        v = float(params.expand)*float(x0)
        e += [np.uint16(randomround(v,params.rng))]
        s += [d[stop]-d[stop-1]]
    return e,s,np.uint32(len(e))

def chankey(c:int):
    return 'ch%02i'%int(c)

def FFTlogic(params,k):
    cepstrum = rfft(np.concatenate((params.data[k][-1],np.flip(params.data[k][-1]))).astype(float),axis=0)
    derivcep = np.copy(cepstrum)
    cepstrum[:params.P] *= params.Pfilt
    cepstrum[params.P:] = 0
    derivcep[:params.P] *= params.Pfilt
    derivcep[:params.P] *= 1j*np.arange(params.P)
    back = irfft(cepstrum,axis=0).real[:params.flim]
    dback = irfft(derivcep,axis=0).real[:params.flim]
    return back*dback

def processFFT(params):
    params.setTid()
    with wave.open(params.fname,'r') as f:
        params.samplerate = f.getframerate()
        params.nsamples = params.samplerate
        params.samplewidth = f.getsampwidth()
        params.totframes = f.getnframes()
        params.nchans = f.getnchannels()
        params.nfolds = np.uint16(min(params.nfolds,params.totframes//params.samplerate))
        print(params.nfolds)
        for c in range(params.nchans):
            params.data[chankey(c)] = []
            params.filtdata[chankey(c)] = []
            params.edges[chankey(c)] = []
            params.slopes[chankey(c)] = []
            params.addresses[chankey(c)] = []
            params.nedges[chankey(c)] = []
            params.initstate[chankey(c)] = True

        params.sizeofframe = params.samplewidth * params.nsamples * params.nchans # in Bytes, not bits
        while ( f.tell() < min(params.totframes - params.sizeofframe , params.nfolds*params.sizeofframe) ) :
            chunk = np.frombuffer( f.readframes(params.nsamples*params.nchans), dtype=params.dtype )
            for c in range(params.nchans):
                S = rfft(np.concatenate( (chunk[c::params.nchans],np.flip(chunk[c::params.nchans],axis=0)) ).astype(int), axis=0 ) 
                params.freqrange = params.samplerate//2
                params.tstep = 1./float(params.samplerate)
                params.data[chankey(c)] += [ np.array([ v>>14 for v in np.abs(S[:params.flim]).astype(int) ]) ]
                #params.data[chankey(c)] += [ np.log2(np.abs(S[:params.flim])+1.).astype(float) ]
                params.filtdata[chankey(c)] += [ FFTlogic(params,chankey(c)) ]
                edges,slopes,nedges = scanedges(params,chankey(c))

                if params.initstate[chankey(c)]:
                    params.addresses[chankey(c)] = [np.uint64(0)]
                    params.nedges[chankey(c)] = [np.uint64(nedges)]
                    params.initstate[chankey(c)] = False
                else:
                    params.addresses[chankey(c)] += [np.uint64(len(params.edges[chankey(c)]))]
                    params.nedges[chankey(c)] += [np.uint64(nedges)]

                if nedges>0:
                    params.edges[chankey(c)] += edges
                    params.slopes[chankey(c)] += slopes 

    h5name = '%s.h5'%(params.fname)
    with h5py.File(h5name,'w') as h:
        for k in params.data.keys():
            chgrp = h.create_group(k)
            chgrp.attrs.create('expand',data=params.expand)
            chgrp.attrs.create('freqrange',data=params.freqrange)
            chgrp.attrs.create('subject',data=params.subject)
            chgrp.attrs.create('tstep',data=1./float(params.samplerate))
            chgrp.attrs.create('samplerate',data=params.samplerate)
            chgrp.attrs.create('freqstep',data='1Hz')
            chgrp.create_dataset('spect',data=np.array(params.data[chankey(c)]))
            chgrp.create_dataset('filt',data=np.array(params.filtdata[chankey(c)]))
            chgrp.create_dataset('edges',data=np.array(params.edges[chankey(c)]))
            chgrp.create_dataset('slopes',data=np.array(params.slopes[chankey(c)]))
            chgrp.create_dataset('nedges',data=np.array(params.nedges[chankey(c)]))
            chgrp.create_dataset('addresses',data=np.array(params.addresses[chankey(c)]))
    return params
    '''
                params.bindata[chankey(c)] += [np.zeros(params.flim,dtype=int)]
                for i,e in enumerate(edges):
                    e >>= params.expand
                    if e< params.flim:
                        params.bindata[chankey(c)][-1][e] = int(np.log2(-slopes[i])-11.) #slopes are negative, want most significant bit. (e.g. log scale)
    for k in params.data.keys():
        oname = '%s.%s.log2spect'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.data[k]).T,
                fmt='%i',
                header='remember for %s, highest frequency is %.3f kHz'%(params.subject,1e-3*params.freqrange) 
                )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
        
        oname = '%s.%s.log2spect_filt'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.filtdata[k]).T,
                fmt='%.1f'
                )
        oname = '%s.%s.log2spect_bin'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.bindata[k]).T,
                fmt='%.1f'
                )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
    return params
    '''

def processDCT(params):
    params.setTid()
    with wave.open(params.fname,'r') as f:
        print(f.getparams())
        params.samplerate = f.getframerate()
        params.samplewidth = f.getsampwidth()
        params.totframes = f.getnframes()
        params.nchans = f.getnchannels()
        for c in range(params.nchans):
            params.data[chankey(c)] = []
            params.filtdata[chankey(c)] = []
        params.sizeofframe = params.samplewidth * params.nsamples * params.nchans
        while f.tell()<min(params.totframes - params.sizeofframe , params.nfolds*params.sizeofframe):
            tmp = np.frombuffer(f.readframes(params.nsamples*params.nchans),dtype=params.dtype)
            for c in range(params.nchans):
                params.data[chankey(c)] += [np.log2(np.abs(dct(np.concatenate((tmp[c::params.nchans],np.flip(tmp[c::params.nchans],axis=0))),type=2,axis=0)[:1<<params.flim:2])/params.scale)]
                cepstrum = dct(np.concatenate((params.data[chankey(c)][-1],np.flip(params.data[chankey(c)][-1]))),axis=0,type=2)
                cepstrum[params.Poffset:params.Poffset+2*params.P:2] *= params.Pfilt
                cepstrum[params.Poffset+2*params.P:] *= 0.0
                cepstrum[:params.Poffset] *= 0.0
                back = dct(cepstrum,type=3,axis=0).real[:1<<(params.flim-1)]/params.scale
                back *= (back>0)
                params.filtdata[chankey(c)] += [back]
                #params.filtdata[chankey(c)] += [(1+np.tanh((back-params.thresh)/params.width))/2]
    for k in params.data.keys():
        oname = '%s.%s.sspect'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.data[k]).T,
                fmt='%i',
                header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(params.subject,int(100*float(1<<params.flim)/float(params.nsamples)),1<<params.flim,np.log2(params.nsamples)) )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
        
        oname = '%s.%s.sspect_filt'%(params.fname,k)
        np.savetxt(oname,
                np.array(params.filtdata[k]).T,
                fmt='%.1f',
                header='remember for %s, highest frequency is %i percent of the 96kHz (only showing 2**%i of 2**%i samples)'%(params.subject,int(100*float(1<<params.flim)/float(params.nsamples)),1<<params.flim,np.log2(params.nsamples)) )
        print('finished writing:\t%s\tprocName:\t%s'%(oname,params.tid))
    return params
