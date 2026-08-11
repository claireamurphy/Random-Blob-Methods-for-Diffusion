import numpy as np
from scipy.integrate import quad

def initial_data(rho_0, R, h):
    x0 = np.arange(-R, R, h)
    if rho_0(R) >= 10**(-3):
        print("Warning: Support radius not large enough.")
        print(rho_0(R))
    
    if 0 < rho_0(R) and rho_0(R) <= 10**(-4):
        print("Warning: Support radius perhaps too large.")
        print(rho_0(R))

    m_vals = np.zeros(x0.size)
    for i in range(x0.size):
        m_vals[i] = quad(rho_0, x0[i] - h / 2, x0[i] + h / 2)[0]
    
    m_vals = m_vals / np.sum(m_vals)
    mask = m_vals != 0
    x0 = x0[mask]
    m_vals = m_vals[mask]
    print("N = " + str(m_vals.size) + ".")

    return x0, m_vals