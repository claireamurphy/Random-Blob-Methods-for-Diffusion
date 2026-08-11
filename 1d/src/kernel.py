import numpy as np

# A mollifier dependent on epsilon
def varphi(x, moll_eps):
    y = x / moll_eps
    denom = np.sqrt(2 * np.pi) * moll_eps
    arg = - (y**2) / 2
    return np.exp(arg) / denom
    
# Calculated by hand - easy to do in the case d = 1
def grad_varphi(x, moll_eps):
    y = x / moll_eps
    denom = np.sqrt(2 * np.pi) * moll_eps**2
    return - y * np.exp(- y**2 / 2) / denom