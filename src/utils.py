import numpy as np

def randomround(x:float,rng):
    return (np.int64(x) + np.int64(x%1>rng.random()))

