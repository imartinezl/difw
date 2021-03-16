import numpy as np


eps = np.finfo(float).eps
eps = 1e-12
np.seterr(divide="ignore", invalid="ignore")

def batch_effect(x, theta):
    n_batch = theta.shape[0]
    n_points = x.shape[-1]
    x = np.broadcast_to(x, (n_batch, n_points)).flatten()
    return x

def get_cell(x, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc

    c = np.floor((x - xmin) / (xmax - xmin) * nc)
    c = np.clip(c, 0, nc - 1).astype(np.int32)
    return c


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]

        # r = np.broadcast_to(np.arange(n_batch), [n_points, n_batch]).T
        # TODO: here we suppose batch effect has been already executed
        r = np.arange(n_batch).repeat(n_points / n_batch)

        A = params.B.dot(theta.T).T.reshape(n_batch, -1, 2)

        return A, r


def precompute_affine(x, theta, params):
    params = params.copy()
    params.precomputed = False
    params.A, params.r = get_affine(x, theta, params)
    params.precomputed = True
    return params


# TODO: theta is not used if precomputed A is used
def get_velocity(x, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return a * x + b


# TODO: theta is not used if precomputed A is used
def get_psi(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    # TODO: review np.where
    cond = a == 0
    x1 = x + t * b
    x2 = np.exp(t * a) * (x + (b / a)) - (b / a)
    psi = np.where(cond, x1, x2)
    # psi[cond] = x[cond] + t * b[cond]
    # psi[~cond] = np.exp(t * a[~cond]) * (x[~cond] + (b[cond] / a[cond])) - (b[cond] / a[cond])
    # psi = np.where(a == 0, x + t * b, np.exp(t * a) * (x + (b / a)) - (b / a))
    return psi


# TODO: theta is not used if precomputed A is used
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


def right_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + (c + 1) * (xmax - xmin) / nc + eps


def left_boundary(c, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc
    return xmin + c * (xmax - xmin) / nc - eps


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


def integrate_numeric(x, t, theta, params):
    nSteps1 = params.nSteps1

    xPrev = x
    deltaT = t / nSteps1
    c = get_cell(xPrev, params)
    for j in range(nSteps1):
        xTemp = get_psi(xPrev, deltaT, theta, params)
        cTemp = get_cell(xTemp, params)
        xNum = get_phi_numeric(xPrev, deltaT, theta, params)
        xPrev = np.where(c == cTemp, xTemp, xNum)
        c = get_cell(xPrev, params)
    return xPrev

def integrate_closed_form(x, t, theta, params):
    n_batch = theta.shape[0]
    cont = 0

    phi = np.empty_like(x)
    done = np.full_like(x, False, dtype=bool)
    
    c = get_cell(x, params)

    while True:
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        v = get_velocity(x, theta, params)
        psi = get_psi(x, t, theta, params)

        cond1 = np.logical_and(left <= psi, psi <= right)
        cond2 = np.logical_and(v >= 0, c == params.nc-1)
        cond3 = np.logical_and(v <= 0, c == 0)
        valid = np.any((cond1, cond2, cond3), axis=0)

        phi[~done] = psi
        done[~done] = valid
        if np.alltrue(valid):
            return phi.reshape((n_batch, -1))

        x, t, params.r = x[~valid], t[~valid], params.r[~valid]
        t -= get_hit_time(x, theta, params)
        x = np.clip(psi, left, right)[~valid]
        c = np.where(v >= 0, c+1, c-1)[~valid]

        cont += 1
        if cont > params.nc:
            raise BaseException
    return None



def integrate_closed_form_ext(x, t, theta, params, derivative=False):
    cont = 0

    n_batch = theta.shape[0]
    n_points = x.shape[-1]
    # phi = np.empty((n_batch, n_points))
    # done = np.full((n_batch, n_points), False)
    phi = np.empty(n_batch * n_points)
    done = np.full(n_batch * n_points, False)

    x, t, params.r = x.flatten(), t.flatten(), params.r.flatten()

    if derivative:
        result = np.vstack([x, t, params.r]).T[np.newaxis]
        b = np.empty((n_batch * n_points, 3), dtype=object)
        print(result.shape, b.shape)

    while True:

        c = get_cell(x, params)
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        psi = get_psi(x, t, theta, params)

        valid = np.logical_and(left <= psi, psi <= right)
        phi[~done] = psi
        done[~done] = valid
        if np.alltrue(valid):
            if derivative:
                return phi, result
            return phi.reshape((n_batch, n_points))

        x, t, params.r = x[~valid], t[~valid], params.r[~valid]
        t -= get_hit_time(x, theta, params)
        x = np.clip(psi[~valid], left[~valid], right[~valid])

        if derivative:
            b[~done] = np.vstack([x, t, params.r]).T
            result = np.append(result, b[np.newaxis], axis=0)
            # b = np.empty_like(done, dtype=object)
            b = np.empty((n_batch * n_points, 3), dtype=object)

        cont += 1
        if cont > params.nc:
            raise BaseException
    return None


# TODO: move methods to interpolation and transformer

# TODO: include derivative methods? hard to do in numpy

compiled = False
# TODO: change name with integrate or integration
def CPAB_transformer(points, theta, params):
    # params = params.copy()
    # n_batch = theta.shape[0]
    # n_points = points.shape[-1]
    # x = np.broadcast_to(points, (n_batch, n_points)).flatten()
    x = batch_effect(points, theta)
    t = np.ones_like(x)
    params = precompute_affine(x, theta, params)
    return integrate_closed_form(x, t, theta, params)
    # newpoints = integrate_analytic(x, t, theta, params, derivative=False)
    # return newpoints

    t = 1
    return integrate_numeric(x, t, theta, params)

    if compiled:
        return CPAB_transformer_fast(points, theta, params)
    else:
        return CPAB_transformer_slow(points, theta, params)


def CPAB_transformer_slow(points, theta, params):
    pass


def CPAB_transformer_fast(points, theta, params):
    pass
    # TODO: jit compile cpp code into callable python code


def derivative(points, theta, params):
    params = params.copy()
    n_batch, n_points = theta.shape[0], points.shape[-1]
    points = np.broadcast_to(points, (n_batch, n_points))
    params.n_batch, params.n_points = n_batch, n_points

    params = precompute_affine(points, theta, params)

    t = np.ones((n_batch, n_points))
    # return integrate_analytic(points, t, theta, params)
    newpoints, trace = integrate_analytic(points, t, theta, params, derivative=True)

    # for j in range(len(theta))
    j = 0
    dthit_dtheta_cum = 0.0
    for i in range(len(trace) - 1):
        print(i)
        x = trace[i, :, 0]
        t = trace[i, :, 1]
        params.r = trace[i, :, 2].astype(int)
        print(x, t, params.r)
        dthit_dtheta = derivative_thit_theta(x, t, theta, j, params)
        dthit_dtheta_cum -= dthit_dtheta

    x, t, params.r = trace[-1, :, 0], trace[-1, :, 1], trace[-1, :, 2]
    dpsi_dtheta = derivative_psi_theta(x, t, theta, j, params)
    dpsi_dtime = derivative_phi_time(x, t, theta, j, params)
    dphi_dtheta = dpsi_dtheta + dpsi_dtime * dthit_dtheta_cum
    return dphi_dtheta


def derivative_psi_theta(x, t, theta, j, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, j]
    bk = params.B[2 * c + 1, j]

    cond = a == 0
    d1 = t * (x * ak + bk)
    d2 = (
        ak * t * np.exp(a * t) * (x + b / a)
        + (np.exp(t * a) - 1) * (bk * a - ak * b) / a ** 2
    )
    dpsi_dtheta = np.where(cond, d1, d2)
    return dpsi_dtheta


def derivative_phi_time(x, t, theta, j, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = a == 0
    d1 = b
    d2 = np.exp(t * a) * (a * x + b)
    dpsi_dtime = np.where(cond, d1, d2)
    return dpsi_dtime


def derivative_thit_theta(x, t, theta, j, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, j]
    bk = params.B[2 * c + 1, j]

    v = get_velocity(x, theta, params)

    xc = np.where(v >= 0, right_boundary(c, params), left_boundary(c, params))

    cond = a == 0
    d1 = (x - xc) * bk / b ** 2
    d2 = -ak * np.log((a * xc + b) / (a * x + b)) / a ** 2
    d3 = (x - xc) * (bk * a - ak * b) / (a * (a * x + b) * (a * xc + b))
    dthit_dtheta = np.where(cond, d1, d2 + d3)
    return dthit_dtheta
