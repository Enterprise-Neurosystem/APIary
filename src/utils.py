import numpy as np
import math

def randomround(x:float,rng):
    return (np.int64(x) + np.int64(x%1>rng.random()))

def mypoly(x,order=4):
    result = np.ones((x.shape[0],order+1),dtype=float)
    result[:,1] = x.copy()
    if order < 2:
        return result
    for p in range(2,order+1):
        result[:,p] = np.power(result[:,1],int(p))
    return result

def fitpoly(x,y,order=4):
    assert len(x)==len(y)
    assert len(x)>order
    x0 = np.mean(np.array(x))
    theta = np.linalg.pinv( mypoly(np.array(x-x0).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
    return x0,theta

def fitval(x,theta):
    y = float(0.)
    for p,th in enumerate(theta):
        y += float(th)*math.pow(x,int(p))
    return y

def fitcurve(x,theta):
    y = np.zeros(x.shape,dtype=float)
    for p in range(theta.shape[0]):
        y += theta[p]*np.power(x,int(p))
    return y
