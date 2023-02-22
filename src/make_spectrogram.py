#!/usr/bin/python3

import sys
import multiprocessing as mp
from Params import Params
from ProcessFile import processFFT

def main(subject,fnames):
    paramslist = [Params(fname,subject) for fname in fnames]
    _ = [p.setnfolds(1<<12).setFreqLim(1<<12).setPfilt(1<<8) for p in paramslist]
    _ = [p.setthresh(1<<14).setexpand(2).setoutbins(1<<10) for p in paramslist]
        

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

