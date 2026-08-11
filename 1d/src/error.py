import numpy as np
from scipy.integrate import quad
import ot

def wass_2_approx(x_vals, m_vals, func, R, dy):
    if func(R) >= 10**(-3):
        print("Warning: Radius of support of discretization of exact continuum solution perhaps too small.")

    y_vals = np.arange(-R, R, dy)
    M = y_vals.size

    sig_vals = np.zeros((M))
    for i in range(M):
        sig_vals[i] = quad(func, y_vals[i] - dy / 2, y_vals[i] + dy / 2)[0]

    mask = sig_vals != 0
    sig_vals = sig_vals[mask]
    y_vals = y_vals[mask]

    sig_vals = sig_vals / sum(sig_vals)
    m_vals = m_vals / sum(m_vals)

    C = (x_vals[:, None] - y_vals[None, :])**2

    return ot.emd2(m_vals, sig_vals, C)**(1/2)