#!/usr/bin/python3

import numpy as np
import multiprocessing as mp
import time


class Params:
    def __init__(self,fname,s):
        self.rng = np.random.default_rng( time.time_ns()%(1<<8) )
        self.logicthresh = (1<<4) 
        self.fname = fname
        self.setsubject(s)
        self.flim = 1<<10
        self.noutbins = 1<<10
        self.tid = 'initState' 
        self.nchans = 1
        self.data = {}
        self.filtdata = {}
        self.bindata = {}
        self.dtype = np.dtype(np.int16).newbyteorder('<')
        self.Poffset = 0
        self.thresh = 14
        self.width =1
        self.samplerate = 192000
        self.sizeofframe=((int(1<<16)<<1)<<8) # 16 bits deep, 2 channels, 8 timesamples per step
        self.tstep = 0.000005208333333333333
        self.freqrange = self.samplerate//2

    def initforsubject(self):
        if self.subject == 'bee':
            self.nsamples = 1<<14
            self.nfolds = 1<<12
            self.scale = 1<<10
        elif self.subject == 'todd':
            self.nsamples = 1<<12
            self.nfolds = 1<<14
            self.scale = 1<<10
        elif self.subject == 'server':
            self.nsamples = 1<<12
            self.nfolds = 1<<14
            self.scale = 1<<12
        return self

    def setsubject(self,s):
        self.subject = s
        self.initforsubject()
        return self

    def setthresh(self,n):
        self.logicthresh = n
        return self

    def getsubject(self):
        return self.subject

    def setnsamples(self,n):
        if n>self.nsamples:
            return self
        self.nsamples = n
        return self

    def setnfolds(self,n):
        if n>self.nfolds:
            reutrn
        self.nfolds = n
        return self

    def setFreqLim(self,f): #set this only for output limiting 
        self.flim = f
        return self

    def setexpand(self,n):
        self.expand = n
        return self

    def setoutbins(self,n):
        self.noutbins = n
        self.outbins = np.arange(0,self.flim+1,float(self.flim)/float(n))
        return self

    def setPfilt(self,n):
        self.P = n
        self.Pfilt = np.zeros(self.P,dtype=float)
        self.Pfilt = 0.5 * ( 1. + np.cos(np.pi * np.arange(self.P,dtype=float) / (self.P)) )
        return self

    def setPoffset(self,p):
        self.Poffset = p
        return self

    def setTid(self):
        self.tid = '%s'%mp.current_process().name
        return self

