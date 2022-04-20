# %% SETUP

import numpy as np

eps = np.finfo(np.float32).eps
np.seterr(divide="ignore", invalid="ignore")

# %% BATCH EFFECT


def batch_effect(x, theta):
    if x.ndim == 1:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        x = np.broadcast_to(x, (n_batch, n_points))  # .flatten()
    return x.flatten()


# %% FUNCTIONS


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]

        # r = np.broadcast_to(np.arange(n_batch), [n_points, n_batch]).T
        # NOTE: here we suppose batch effect has been already executed
        r = np.arange(n_batch).repeat(n_points / n_batch)

        A = params.B.dot(theta.T).T.reshape(n_batch, -1, 2)

        return A, r


def precompute_affine(x, theta, params):
    params = params.copy()
    params.precomputed = False
    params.A, params.r = get_affine(x, theta, params)
    params.precomputed = True
    return params


def right_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + (c + 1) * (xmax - xmin) / nc + eps


def left_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + c * (xmax - xmin) / nc - eps


def get_cell(x, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc

    c = np.floor((x - xmin) / (xmax - xmin) * nc)
    c = np.clip(c, 0, nc - 1).astype(np.int32)
    return c


def get_velocity(x, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return a * x + b


def get_psi(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = a == 0
    x1 = x + t * b
    eta = np.exp(t * a)
    x2 = eta * x + (b / a) * (eta - 1.0)
    # x2 = np.exp(t * a) * (x + (b / a)) - (b / a)
    psi = np.where(cond, x1, x2)
    return psi


def get_hit_time(x, theta, params):
    A, r = get_affine(x, theta, params)

    c = get_cell(x, params)
    v = get_velocity(x, theta, params)
    xc = np.where(v >= 0, right_boundary(c, params), left_boundary(c, params))

    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = a == 0
    x1 = (xc - x) / b
    x2 = np.log((xc + b / a) / (x + b / a)) / a
    thit = np.where(cond, x1, x2)
    return thit


def get_phi_numeric(x, t, theta, params):
    nSteps2 = params.nSteps2

    yn = x
    deltaT = t / nSteps2
    for j in range(nSteps2):
        c = get_cell(yn, params)
        midpoint = yn + deltaT / 2 * get_velocity(yn, theta, params)
        c = get_cell(midpoint, params)
        yn = yn + deltaT * get_velocity(midpoint, theta, params)
    return yn


# %% INTEGRATION


def integrate_numeric(x, theta, params, time=1.0):
    # setup
    x = batch_effect(x, theta)
    n_batch = theta.shape[0]
    t = time
    params = precompute_affine(x, theta, params)

    # computation
    xPrev = x
    nSteps1 = params.nSteps1
    deltaT = t / nSteps1
    c = get_cell(xPrev, params)
    for j in range(nSteps1):
        xTemp = get_psi(xPrev, deltaT, theta, params)
        cTemp = get_cell(xTemp, params)
        xNum = get_phi_numeric(xPrev, deltaT, theta, params)
        xPrev = np.where(c == cTemp, xTemp, xNum)
        c = get_cell(xPrev, params)
    return xPrev.reshape((n_batch, -1))


def integrate_closed_form(x, theta, params, time=1.0):
    # setup
    x = batch_effect(x, theta)
    t = np.ones_like(x) * time
    params = precompute_affine(x, theta, params)
    n_batch = theta.shape[0]

    # computation
    phi = np.empty_like(x)
    done = np.full_like(x, False, dtype=bool)
    c = get_cell(x, params)

    cont = 0
    while True:
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        v = get_velocity(x, theta, params)
        psi = get_psi(x, t, theta, params)

        cond1 = np.logical_and(left <= psi, psi <= right)
        cond2 = np.logical_and(v >= 0, c == params.nc - 1)
        cond3 = np.logical_and(v <= 0, c == 0)
        valid = np.any((cond1, cond2, cond3), axis=0)

        phi[~done] = psi
        done[~done] = valid
        if np.alltrue(valid):
            return phi.reshape((n_batch, -1))

        x, t, params.r = x[~valid], t[~valid], params.r[~valid]
        t -= get_hit_time(x, theta, params)
        x = np.clip(psi, left, right)[~valid]
        c = np.where(v >= 0, c + 1, c - 1)[~valid]

        cont += 1
        if cont > params.nc:
            raise BaseException
    return None


def integrate_closed_form_trace(x, theta, params, time=1.0):

    x = batch_effect(x, theta)
    t = np.ones_like(x) * time
    params = precompute_affine(x, theta, params)

    result = np.empty((*x.shape, 3))
    done = np.full_like(x, False, dtype=bool)

    c = get_cell(x, params)
    cont = 0
    while True:
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        v = get_velocity(x, theta, params)
        psi = get_psi(x, t, theta, params)

        cond1 = np.logical_and(left <= psi, psi <= right)
        cond2 = np.logical_and(v >= 0, c == params.nc - 1)
        cond3 = np.logical_and(v <= 0, c == 0)
        valid = np.any((cond1, cond2, cond3), axis=0)

        result[~done] = np.array([psi, t, c]).T
        done[~done] = valid
        if np.alltrue(valid):
            return result

        x, t, params.r = x[~valid], t[~valid], params.r[~valid]
        t -= get_hit_time(x, theta, params)
        x = np.clip(psi, left, right)[~valid]
        c = np.where(v >= 0, c + 1, c - 1)[~valid]

        cont += 1
        if cont > params.nc:
            raise BaseException
    return None


# %% DERIVATIVE


def derivative_numeric(x, theta, params, time=1.0, h=1e-3):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    der = np.empty((n_batch, n_points, d))

    phi_1 = integrate_numeric(x, theta, params, time)
    for k in range(d):
        theta2 = theta.copy()
        theta2[:, k] += h
        phi_2 = integrate_numeric(x, theta2, params, time)

        der[:, :, k] = (phi_2 - phi_1) / h

    return phi_1, der


def derivative_closed_form(x, theta, params, time=1.0):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    result = integrate_closed_form_trace(x, theta, params, time)
    phi = result[:, 0].reshape((n_batch, -1))
    tm = result[:, 1]
    cm = result[:, 2]

    # setup
    x = batch_effect(x, theta)
    params = precompute_affine(x, theta, params)

    der = np.empty((n_batch, n_points, d))
    for k in range(d):
        dthit_dtheta_cum = np.zeros_like(x)

        xm = x.copy()
        c = get_cell(x, params)
        while True:
            valid = c == cm
            if np.alltrue(valid):
                break
            step = np.sign(cm - c)
            dthit_dtheta_cum[~valid] -= derivative_thit_theta(xm, theta, k, params)[~valid]
            xm[~valid] = np.where(step == 1, right_boundary(c, params), left_boundary(c, params))[~valid]
            c = c + step

        dpsi_dtheta = derivative_psi_theta(xm, tm, theta, k, params)
        dpsi_dtime = derivative_phi_time(xm, tm, theta, k, params)
        dphi_dtheta = dpsi_dtheta + dpsi_dtime * dthit_dtheta_cum
        der[:, :, k] = dphi_dtheta.reshape(n_batch, n_points)

    return phi, der


def derivative_psi_theta(x, t, theta, k, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, k]
    bk = params.B[2 * c + 1, k]

    cond = a == 0
    d1 = t * (x * ak + bk)
    d2 = ak * t * np.exp(a * t) * (x + b / a) + (np.exp(t * a) - 1) * (bk * a - ak * b) / a ** 2
    dpsi_dtheta = np.where(cond, d1, d2)
    return dpsi_dtheta


def derivative_phi_time(x, t, theta, k, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = a == 0
    d1 = b
    d2 = np.exp(t * a) * (a * x + b)
    dpsi_dtime = np.where(cond, d1, d2)
    return dpsi_dtime


def derivative_thit_theta(x, theta, k, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, k]
    bk = params.B[2 * c + 1, k]

    v = get_velocity(x, theta, params)

    xc = np.where(v >= 0, right_boundary(c, params), left_boundary(c, params))

    cond = a == 0
    d1 = (x - xc) * bk / b ** 2
    d2 = -ak * np.log((a * xc + b) / (a * x + b)) / a ** 2
    d3 = (x - xc) * (bk * a - ak * b) / (a * (a * x + b) * (a * xc + b))
    dthit_dtheta = np.where(cond, d1, d2 + d3)
    return dthit_dtheta


# %% DERIVATIVE SPACE

def derivative_space_numeric(x, theta, params, time=1.0, h=1e-3):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]

    # computation
    xe = np.concatenate([x, x+h])

    phi = integrate_numeric(xe, theta, params, time)
    phi_1, phi_2 = np.split(phi, 2, axis=1)
    der = (phi_2 - phi_1) / h

    return phi_1, der

def derivative_space_closed_form(x, theta, params, time=1.0):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    t = np.ones_like(x) * time
    result = integrate_closed_form_trace(x, theta, params, time)
    phi = result[:, 0].reshape((n_batch, -1))
    tm = result[:, 1]
    cm = result[:, 2]

    # setup
    x = batch_effect(x, theta)
    params = precompute_affine(x, theta, params)

    dthit_dx = np.zeros_like(x)
    dpsi_dx = np.zeros_like(x)
    
    c = get_cell(x, params)
    valid = c == cm
    # dpsi_dx only on first valid cell
    dpsi_dx[valid] = derivative_psi_x(x, t, theta, params)[valid]
    # dthit_dx only on first non valid cell
    dthit_dx[~valid] = derivative_thit_x(x, t, theta, params)[~valid]

    xm = x.copy()
    while True:
        valid = c == cm
        if np.alltrue(valid):
            break
        step = np.sign(cm - c)
        xm[~valid] = np.where(step == 1, right_boundary(c, params), left_boundary(c, params))[~valid]
        c = c + step
    
    # dpsi_dtime at last cell
    dpsi_dtime = derivative_psi_t(xm, tm, theta, params)
    dphi_dx = dpsi_dx + dpsi_dtime * dthit_dx

    dphi_dx = dphi_dx.reshape(n_batch, n_points)

    return phi, dphi_dx


def derivative_thit_x(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    return 1.0 / (a*x + b)

def derivative_psi_x(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    return np.exp(t * a)

def derivative_psi_t(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return np.exp(t * a) * (a * x + b)