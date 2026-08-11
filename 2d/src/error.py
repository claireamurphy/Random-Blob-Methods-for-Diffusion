import numpy as np
from scipy import integrate
import ot

def wass_2_approx(particles, m_vals, func, R, h_1):
    if func(R, 0) > 10**(-3):
        print("Warning: Radius of support of discretization of exact continuum solution perhaps too small.")

    # get x_vals and y_vals
    vec = np.arange(-R, R + h_1, h_1)
    n = vec.size
    ones = np.ones((n, n))
    x_vals = (vec * ones).flatten()
    y_vals = np.repeat(vec, n)
              
    sig_vals = np.zeros((n**2))
    for i in range(n**2):
        sig_vals[i] = integrate.dblquad(func, y_vals[i] - h_1 / 2, y_vals[i] + h_1 / 2, x_vals[i] - h_1 / 2, x_vals[i] + h_1 / 2)[0]
    
    mask = sig_vals != 0
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    sig_vals = sig_vals[mask]

    sig_vals = sig_vals / sum(sig_vals)
    m_vals = m_vals / sum(m_vals)
    
    N = int(particles.size / 2)
    C = (particles[:N, None] - x_vals[None, :])**2 + (particles[N:2*N, None] - y_vals[None, :])**2

    return ot.emd2(m_vals, sig_vals, C, numItermax = 10**9)**(1/2)