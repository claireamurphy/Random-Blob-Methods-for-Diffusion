import numpy as np
import time
from src.kernel import varphi_and_grad

def xy_dot_batch(x_vals, y_vals, m_vals, batch_in, batch_out, moll_eps, f_double_prime_epsilon, grad_V, nav_stokes, g):
    N = batch_out.size
    res = np.zeros((2 * N))

    my_m_vals = m_vals[batch_in] / np.sum(m_vals[batch_in])

    # x_diff[i, j] = x[i] - x[j]
    x_diff = x_vals[batch_out] - x_vals[batch_in, None]
    y_diff = y_vals[batch_out] - y_vals[batch_in, None]

    temp = varphi_and_grad(x_diff, y_diff, moll_eps)
    varph = temp[0]
    grad = temp[1]

    first_factor_arg = np.sum(my_m_vals[:, None] * varph, axis = 0)
    first_factor = f_double_prime_epsilon(first_factor_arg)

    df_dx = np.sum(my_m_vals[:, None] * grad[0, :, :], axis = 0)
    df_dy = np.sum(my_m_vals[:, None] * grad[1, :, :], axis = 0)

    res[:N] = np.multiply(first_factor, df_dx)
    res[N:] = np.multiply(first_factor, df_dy)

    if not nav_stokes:
      return - res - grad_V(x_vals[batch_out], y_vals[batch_out])

    # Introduce EXTERNAL Navier-Stokes velocity field
    norms_squared = x_diff**2 + y_diff**2
    zero_mask = (norms_squared == 0)
    norms_squared[zero_mask] = 1
    factors = 1 / (2 * np.pi * norms_squared)
    factors[zero_mask] = 0

    g_vals = g(np.sqrt(norms_squared))
    K_2_x_vals = np.sum(my_m_vals[:, None] * -y_diff * factors * g_vals, axis = 0)
    K_2_y_vals = np.sum(my_m_vals[:, None] * x_diff * factors * g_vals, axis = 0)

    res[:N] = - res[:N] + K_2_x_vals
    res[N:] = - res[N:] + K_2_y_vals
    return res

def solve_ODE(x0, y0, m_vals, alg, num_batches, ratio, moll_eps, f_double_prime_epsilon, grad_V, t_step, t_max, nav_stokes = False, g = None, bounds = None):
  def f(x, y, batch_in, batch_out):
      return xy_dot_batch(x, y, m_vals, batch_in, batch_out, moll_eps, f_double_prime_epsilon, grad_V, nav_stokes, g)
  result = advance (f, x0, y0, alg, num_batches, ratio, t_step, t_max, bounds)
  return result

def advance(f, x0, y0, alg, num_batches, ratio, t_step, t_max, bounds=None):
    N = len(x0)
    M = int(t_max / t_step)

    x = np.zeros([M + 1, N])
    y = np.zeros([M + 1, N])
    x[0, :] = x0
    y[0, :] = y0

    rng = np.random.default_rng(seed=1)
    batch_size = int(N / num_batches)
    rm_state = None

    start_time = time.time()
    for i in range(0, M):
        print("Time step " + str(i + 1) + ".")

        xo = x[i, :]
        yo = y[i, :]

        if alg == 1:
            xp, yp = _rb_step(f, xo, yo, num_batches, batch_size, t_step, rng)
        if alg == 2:
            xp, yp, rm_state = _rm_step(f, xo, yo, i, ratio, t_step, batch_size, N, rng, rm_state)

        if bounds is not None:
            xp = np.clip(xp, -bounds, bounds)
            yp = np.clip(yp, -bounds, bounds)

        x[i + 1, :] = xp
        y[i + 1, :] = yp

    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)

    return np.concatenate((x, y), axis=1), runtime

def _rb_step(f, xo, yo, num_batches, batch_size, t_step, rng):
    N = xo.size
    xp = xo.copy()
    yp = yo.copy()
    indices = rng.permutation(np.arange(N))
    for this_batch in np.array_split(indices, num_batches):
        trajectories = f(xo, yo, this_batch, this_batch)
        xp[this_batch] = xo[this_batch] + t_step * trajectories[:batch_size]
        yp[this_batch] = yo[this_batch] + t_step * trajectories[batch_size:]
    return xp, yp

def _rm_step(f, xo, yo, i, ratio, t_step, batch_size, N, rng, state):
    xp = xo.copy()
    yp = yo.copy()
    indices = np.arange(N)

    if i % ratio == 0:
        slow_ind = np.sort(rng.choice(indices, size=batch_size, replace=False))
        mask = np.ones(N, dtype=bool)
        mask[slow_ind] = False
        fast_ind = np.nonzero(mask)[0]

        fast_start_x = xo.copy()
        fast_start_y = yo.copy()

        trajectories = f(xo, yo, indices, fast_ind)
        xp[fast_ind] = xo[fast_ind] + ratio * t_step * trajectories[:(N - batch_size)]
        yp[fast_ind] = yo[fast_ind] + ratio * t_step * trajectories[(N - batch_size):]

        fast_final_x = xp.copy()
        fast_final_y = yp.copy()
        state = (fast_ind, slow_ind, fast_start_x, fast_start_y, fast_final_x, fast_final_y, 0)
    else:
        fast_ind, slow_ind, fast_start_x, fast_start_y, fast_final_x, fast_final_y, j = state
        xp[fast_ind] = xo[fast_ind]
        yp[fast_ind] = yo[fast_ind]

    fast_ind, slow_ind, fast_start_x, fast_start_y, fast_final_x, fast_final_y, j = state

    interp_x = xp.copy()
    interp_y = yp.copy()
    theta = (j + 1) / ratio
    interp_x[fast_ind] = (1 - theta) * fast_start_x[fast_ind] + theta * fast_final_x[fast_ind]
    interp_y[fast_ind] = (1 - theta) * fast_start_y[fast_ind] + theta * fast_final_y[fast_ind]

    trajectories = f(interp_x, interp_y, indices, slow_ind)
    xp[slow_ind] = xo[slow_ind] + t_step * trajectories[:batch_size]
    yp[slow_ind] = yo[slow_ind] + t_step * trajectories[batch_size:]

    return xp, yp, (fast_ind, slow_ind, fast_start_x, fast_start_y, fast_final_x, fast_final_y, j + 1)

