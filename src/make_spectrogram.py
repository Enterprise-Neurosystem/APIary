#!/usr/bin/python3

import sys
import multiprocessing as mp
from Params import Params
from ProcessFile import processFFT




def main(subject,fnames):
    paramslist = [Params(fname,subject) for fname in fnames]
    _ = [p.setnfolds(1<<8).setFreqLim(1<<12).setPfilt(1<<8) for p in paramslist]
    _ = [p.setthresh(1<<10).setexpand(1<<2).setoutbins(1<<10) for p in paramslist]
        

    print('CPU cores:\t%i'%mp.cpu_count())
    _ = [print(p.fname) for p in paramslist]

    with mp.Pool(processes=len(paramslist)) as pool:
        pool.map(processFFT,paramslist)
    return

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("syntax:./src/make_spectrogram.py <'bee'|'server'|'todd'> <path/fnames>")
    else:
        subject = sys.argv[1]
        fnames = sys.argv[2:]
        main(subject,fnames)



#### old method ######
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
