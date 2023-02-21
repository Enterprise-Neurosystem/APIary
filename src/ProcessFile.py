#!/usr/bin/python3

import numpy as np
import wave
from scipy.fft import dct,rfft,irfft
from scipy.io import wavfile
import Params
from utils import randomround

def scanedges(params,d):
    fs = []
    slopes = []
    sz = d.shape[0]
    i:int = int(10)
    while i < sz-10:
        while d[i] < params.logicthresh:
            i += 1
            if i==sz-10: return fs,slopes,len(fs)
        while i<sz-10 and d[i]>0:
            i += 1
        stop = i
        ''' dx / (Dy) = dx2/dy2 ; dy2*dx/Dy - dx2 ; x2-dx2 = stop - dy2*1/Dy'''
        x0 = float(stop) - float(d[stop])/float(d[stop]-d[stop-1])
        i += 1
        v = float(params.expand)*float(x0)
        fs += [np.uint16(randomround(v,params.rng))]
        slopes += [d[stop]-d[stop-1]]
    return fs,slopes,np.uint32(len(fs))



def processFFT(params):
    params.setTid()
    with wave.open(params.fname,'r') as f:
        params.samplerate = f.getframerate()
        params.nsamples = params.samplerate
        params.samplewidth = f.getsampwidth()
        params.totframes = f.getnframes()
        params.nchans = f.getnchannels()
        for c in range(params.nchans):
            params.data['ch%02i'%c] = []
            params.filtdata['ch%02i'%c] = []
            params.bindata['ch%02i'%c] = []
        params.sizeofframe = params.samplewidth * params.nsamples * params.nchans # in Bytes, not bits
        while ( f.tell() < min(params.totframes - params.sizeofframe , params.nfolds*params.sizeofframe) ) :
            chunk = np.frombuffer( f.readframes(params.nsamples*params.nchans), dtype=params.dtype )
            for c in range(params.nchans):
                S = rfft(np.concatenate( (chunk[c::params.nchans],np.flip(chunk[c::params.nchans],axis=0)) ).astype(int), axis=0 ) 
                params.freqrange = params.samplerate//2
                params.tstep = 1./float(params.samplerate)
                params.data['ch%02i'%c] += [ np.log2(np.abs(S[:params.flim])+1.).astype(float) ]
                cepstrum = rfft(np.concatenate((params.data['ch%02i'%c][-1],np.flip(params.data['ch%02i'%c][-1]))),axis=0)
                cepstrum[:params.P] *= params.Pfilt
                cepstrum[params.P:] = 0
                derivcep = np.copy(cepstrum)
                derivcep[:params.P] *= 1j*np.arange(params.P)
                back = irfft(cepstrum,axis=0).real[:params.flim]
                dback = irfft(derivcep,axis=0).real[:params.flim]
                params.filtdata['ch%02i'%c] += [dback*back]
                edges,slopes,nedges = scanedges(params,params.filtdata['ch%02i'%c][-1])
                params.bindata['ch%02i'%c] += [np.zeros(params.noutbins,dtype=int)]
                for e in edges:
                    e >>= params.expand
                    if e< params.noutbins:
                        params.bindata['ch%02i'%c][-1][e] += 1 
                #params.bindata['ch%02i'%c] += [np.histogram(edges,bins=params.outbins)[0]]
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

def processDCT(params):
    params.setTid()
    with wave.open(params.fname,'r') as f:
        print(f.getparams())
        params.samplerate = f.getframerate()
        params.samplewidth = f.getsampwidth()
        params.totframes = f.getnframes()
        params.nchans = f.getnchannels()
        for c in range(params.nchans):
            params.data['ch%02i'%c] = []
            params.filtdata['ch%02i'%c] = []
        params.sizeofframe = params.samplewidth * params.nsamples * params.nchans
        while f.tell()<min(params.totframes - params.sizeofframe , params.nfolds*params.sizeofframe):
            tmp = np.frombuffer(f.readframes(params.nsamples*params.nchans),dtype=params.dtype)
            for c in range(params.nchans):
                params.data['ch%02i'%c] += [np.log2(np.abs(dct(np.concatenate((tmp[c::params.nchans],np.flip(tmp[c::params.nchans],axis=0))),type=2,axis=0)[:1<<params.flim:2])/params.scale)]
                cepstrum = dct(np.concatenate((params.data['ch%02i'%c][-1],np.flip(params.data['ch%02i'%c][-1]))),axis=0,type=2)
                cepstrum[params.Poffset:params.Poffset+2*params.P:2] *= params.Pfilt
                cepstrum[params.Poffset+2*params.P:] *= 0.0
                cepstrum[:params.Poffset] *= 0.0
                back = dct(cepstrum,type=3,axis=0).real[:1<<(params.flim-1)]/params.scale
                back *= (back>0)
                params.filtdata['ch%02i'%c] += [back]
                #params.filtdata['ch%02i'%c] += [(1+np.tanh((back-params.thresh)/params.width))/2]
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
