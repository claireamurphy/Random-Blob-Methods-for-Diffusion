import numpy as np
import time

from src.kernel import varphi, grad_varphi

def x_dot(x_values, m_values, batch_out, moll_eps, f_double_prime_epsilon, grad_V):
    first_factor_arg = np.dot(m_values, varphi(x_values[batch_out] - x_values[:, None], moll_eps))
    first_factor = f_double_prime_epsilon(first_factor_arg)
    second_factor = np.dot(m_values, grad_varphi(x_values[batch_out] - x_values[:, None], moll_eps))
    return - np.multiply(first_factor, second_factor) - grad_V(x_values[batch_out])
    
def solve_ODE(x0, m_values, alg, num_batches, ratio, moll_eps, f_double_prime_epsilon, grad_V, t_step, t_max, bounds = None):
    def f(x, batch):
        if alg == 1:
            m_hat = m_values[batch] / np.sum(m_values[batch])
            return x_dot( x, m_hat, np.arange(x.size), moll_eps, f_double_prime_epsilon, grad_V )
        if alg == 2:
            return x_dot( x, m_values, batch, moll_eps, f_double_prime_epsilon, grad_V )

    return advance(f, x0, alg, num_batches, t_step, t_max, ratio, bounds)

def advance(f, x0, alg, num_batches, t_step, t_max, ratio, bounds):
    N = len(x0)
    M = int(t_max / t_step)

    x = np.zeros([M + 1, N])
    x[0, :] = x0

    rng = np.random.default_rng(seed=1)
    batch_size = int(N / num_batches)
    rm_state = None

    start_time = time.time()
    for i in range(0, M):
        if i % 1000 == 999:
            print("Time step " + str(i + 1) + ".")

        xo = x[i, :]
        if alg == 1:
            xp = _rb_step(f, xo, num_batches, t_step, rng)
        if alg == 2:
            xp, rm_state = _rm_step(f, xo, i, ratio, t_step, batch_size, N, rng, rm_state)

        if bounds is not None:
            xp = np.clip(xp, - bounds, bounds)

        x[i + 1] = xp

    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    return x, runtime

def _rb_step(f, xo, num_batches, t_step, rng):
    N = xo.size
    xp = xo.copy()
    indices = rng.permutation(np.arange(N))
    for batch in np.array_split(indices, num_batches):
        xp[batch] = xo[batch] + t_step * f(xo[batch], batch)
    return xp

def _rm_step(f, xo, i, ratio, t_step, batch_size, N, rng, state):
    xp = xo.copy()

    if i % ratio == 0:
        indices = np.arange(N)
        slow_ind = rng.choice(indices, size=batch_size, replace=False)
        mask = np.ones(N, dtype=bool)
        mask[slow_ind] = False
        fast_ind = np.nonzero(mask)[0]

        fast_start = xo.copy()
        xp[fast_ind] = xo[fast_ind] + ratio * t_step * f(xo, fast_ind)
        fast_final = xp.copy()
        state = (fast_ind, slow_ind, fast_start, fast_final, 0)
    else:
        fast_ind, slow_ind, fast_start, fast_final, j = state
        xp[fast_ind] = xo[fast_ind]

    fast_ind, slow_ind, fast_start, fast_final, j = state
    interp_x = xp.copy()
    theta = (j + 1) / ratio
    interp_x[fast_ind] = (1 - theta) * fast_start[fast_ind] + theta * fast_final[fast_ind]
    xp[slow_ind] = xo[slow_ind] + t_step * f(interp_x, slow_ind)

    return xp, (fast_ind, slow_ind, fast_start, fast_final, j + 1)