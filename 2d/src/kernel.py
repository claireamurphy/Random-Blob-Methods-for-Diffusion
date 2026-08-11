import numpy as np

def varphi_and_grad(x, y, moll_eps):
    denom = 2 * np.pi * moll_eps**2
    arg = - (x**2 + y**2) / (2 * moll_eps**2)
    my_varphi = np.exp(arg) / denom

    my_grad_varphi = my_varphi / (- moll_eps**2) * np.array([x, y])
    return my_varphi, my_grad_varphi