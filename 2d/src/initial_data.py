import numpy as np
from scipy import integrate

def initial_data(rho_0, R, h):
   points_x = np.array([])
   points_y = np.array([])
   m_vals = np.array([])

   if rho_0(R, 0) > 10**(-3):
        print("Support radius not large enough.")
        print(rho_0(R, 0))
            
   N = int(2 * R / h) + 1
   vec = np.linspace(-R, R, N)
   ones = np.ones((N, N))
   points_x = (vec * ones).flatten()
   points_y = np.repeat(vec, N)

   m_vals = np.zeros((N**2))
   for i in range(N**2):
    m_vals[i] = integrate.dblquad(rho_0, points_y[i] - h / 2, points_y[i] + h / 2, points_x[i] - h / 2, points_x[i] + h / 2)[0]

   mask = m_vals != 0
   points_x = points_x[mask]
   points_y = points_y[mask]
   m_vals = m_vals[mask]
   m_vals = m_vals / sum(m_vals)

   return points_x, points_y, m_vals