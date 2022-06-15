# %% SETUP

import torch

# %% COMPARE EQUAL TO ZERO
eps = torch.finfo(torch.float32).eps


def cmpf(x, y):
    return torch.abs(x - y) < eps


def cmpf0(x):
    return torch.abs(x) < eps


# %% BATCH EFFECT


def batch_effect(x, theta):
    if x.ndim == 1:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        x = torch.broadcast_to(x, (n_batch, n_points))  # .flatten()
    return x.flatten()


# %% FUNCTIONS


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        repeat = int(n_points / n_batch)

        # r = np.broadcast_to(np.arange(n_batch), [n_points, n_batch]).T
        # NOTE: here we suppose batch effect has been already executed
        r = torch.arange(n_batch).repeat_interleave(repeat).long().to(x.device)

        A = params.B.mm(theta.T).T.reshape(n_batch, -1, 2).to(x.device)
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

    c = torch.floor((x - xmin) / (xmax - xmin) * nc)
    c = torch.clamp(c, 0, nc - 1).long()
    return c


def get_velocity(x, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return a * x + b


def calc_velocity(grid, theta, params):
    grid = batch_effect(grid, theta)
    v = get_velocity(grid, theta, params)
    return v.reshape(theta.shape[0], -1)


def get_psi(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = cmpf0(a)
    x1 = x + t * b
    eta = torch.exp(t * a)
    x2 = eta * x + (b / a) * (eta - 1.0)
    # x2 = torch.exp(t * a) * (x + (b / a)) - (b / a)
    psi = torch.where(cond, x1, x2)
    return psi


def get_hit_time(x, theta, params):

    thit = torch.empty_like(x)
    valid = torch.full_like(x, True, dtype=bool)

    c = get_cell(x, params)
    A, r = get_affine(x, theta, params)

    a = A[r, c, 0]
    b = A[r, c, 1]

    v = a * x + b
    cond1 = cmpf0(v)

    cc = c + torch.sign(v)
    cond2 = torch.logical_or(cc < 0, cc >= params.nc)
    xc = torch.where(v > 0, right_boundary(c, params), left_boundary(c, params))

    vc = a * xc + b
    cond3 = cmpf0(vc)
    cond4 = torch.sign(v) != torch.sign(vc)
    cond5 = torch.logical_or(xc == params.xmin, xc == params.xmax)

    cond = cond1 | cond2 | cond3 | cond4 | cond5
    thit[~cond] = torch.where(
        cmpf0(a[~cond]), (xc[~cond] - x[~cond]) / b[~cond], torch.log(vc[~cond] / v[~cond]) / a[~cond],
    )
    thit[cond] = float("inf")
    return thit, xc, cc


def get_phi_numeric(x, t, theta, params):
    nSteps2 = params.nSteps2

    yn = x
    deltaT = t / nSteps2
    for j in range(nSteps2):
        midpoint = yn + deltaT / 2 * get_velocity(yn, theta, params)
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
    for j in range(nSteps1):
        c = get_cell(xPrev, params)
        xTemp = get_psi(xPrev, deltaT, theta, params)
        cTemp = get_cell(xTemp, params)

        xNum = get_phi_numeric(xPrev, deltaT, theta, params)
        xPrev = torch.where(c == cTemp, xTemp, xNum)
    return xPrev.reshape((n_batch, -1))


def integrate_closed_form(x, theta, params, time=1.0):
    # setup
    x = batch_effect(x, theta)
    t = torch.ones_like(x) * time
    params = precompute_affine(x, theta, params)
    n_batch = theta.shape[0]

    # computation
    phi = torch.empty_like(x)
    done = torch.full_like(x, False, dtype=bool)

    c = get_cell(x, params)
    cont = 0
    while True:
        thit, xc, cc = get_hit_time(x, theta, params)
        psi = get_psi(x, t, theta, params)

        valid = thit > t
        phi[~done] = psi
        done[~done] = valid

        if torch.all(valid):
            return phi.reshape((n_batch, -1))

        params.r = params.r[~valid]
        x = xc[~valid]
        c = cc[~valid]
        t = (t - thit)[~valid]

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
    der = torch.empty((n_batch, n_points, d), device=x.device)

    phi_1 = integrate_numeric(x, theta, params, time)
    for k in range(d):
        theta2 = theta.clone()
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

    der = torch.empty((n_batch, n_points, d), device=x.device)
    for k in range(d):
        dthit_dtheta_cum = torch.zeros_like(x)

        xm = x.clone()
        c = get_cell(x, params)
        while True:
            valid = c == cm
            if torch.all(valid):
                break
            step = torch.sign(cm - c)
            dthit_dtheta_cum[~valid] -= derivative_thit_theta(xm, theta, k, params)[~valid]
            xm[~valid] = torch.where(step == 1, right_boundary(c, params), left_boundary(c, params))[~valid]
            c = c + step

        dpsi_dtheta = derivative_psi_theta(xm, tm, theta, k, params)
        dpsi_dtime = derivative_phi_time(xm, tm, theta, k, params)
        dphi_dtheta = dpsi_dtheta + dpsi_dtime * dthit_dtheta_cum
        der[:, :, k] = dphi_dtheta.reshape(n_batch, n_points)

    return phi, der


def derivative_psi_theta(x, t, theta, k, params):
    A, r = get_affine(x, theta, params)
    A = A.double()  # NOTE: double precision is necessary

    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, k]
    bk = params.B[2 * c + 1, k]

    cond = cmpf0(a)
    d1 = t * (x * ak + bk)
    d2 = ak * t * torch.exp(t * a) * (x + b / a) + (torch.exp(t * a) - 1) * (bk * a - ak * b) / a ** 2
    d1 = d1.double()
    d2 = d2.double()
    dpsi_dtheta = torch.where(cond, d1, d2)
    return dpsi_dtheta


def derivative_phi_time(x, t, theta, k, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = cmpf0(a)
    d1 = b
    d2 = torch.exp(t * a) * (a * x + b)
    dpsi_dtime = torch.where(cond, d1, d2)
    return dpsi_dtime


def derivative_thit_theta(x, theta, k, params):
    A, r = get_affine(x, theta, params)
    A = A.double()  # NOTE: double precision is necessary

    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]

    ak = params.B[2 * c, k]
    bk = params.B[2 * c + 1, k]

    v = get_velocity(x, theta, params)

    xc = torch.where(v >= 0, right_boundary(c, params), left_boundary(c, params))

    cond = cmpf0(a)
    d1 = (x - xc) * bk / b ** 2
    d2 = -ak * torch.log((a * xc + b) / (a * x + b)) / a ** 2
    d3 = (x - xc) * (bk * a - ak * b) / (a * (a * x + b) * (a * xc + b))
    dthit_dtheta = torch.where(cond, d1, d2 + d3)
    return dthit_dtheta


# %% TRANSFORMATION


def integrate_closed_form_trace(x, theta, params, time=1.0):
    # setup
    x = batch_effect(x, theta)
    t = torch.ones_like(x) * time
    params = precompute_affine(x, theta, params)
    n_batch = theta.shape[0]

    # computation
    result = torch.empty((*x.shape, 3), device=x.device)
    done = torch.full_like(x, False, dtype=bool)

    c = get_cell(x, params)
    cont = 0
    while True:
        thit, xc, cc = get_hit_time(x, theta, params)
        psi = get_psi(x, t, theta, params)

        valid = thit > t
        result[~done] = torch.stack((psi, t, c)).T
        done[~done] = valid

        if torch.all(valid):
            return result

        params.r = params.r[~valid]
        x = xc[~valid]
        c = cc[~valid]
        t = (t - thit)[~valid]

        cont += 1
        if cont > params.nc:
            raise BaseException
    return None


def derivative_closed_form_trace(result, x, theta, params):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    # result = integrate_closed_form_trace(x, theta, params, time)
    phi = result[:, 0].reshape((n_batch, -1))
    tm = result[:, 1]
    cm = result[:, 2]

    # setup
    x = batch_effect(x, theta)
    params = precompute_affine(x, theta, params)

    der = torch.empty((n_batch, n_points, d), device=x.device)
    for k in range(d):
        dthit_dtheta_cum = torch.zeros_like(x)

        xm = x.clone()
        c = get_cell(x, params)
        while True:
            valid = c == cm
            if torch.all(valid):
                break
            step = torch.sign(cm - c)
            dthit_dtheta_cum[~valid] -= derivative_thit_theta(xm, theta, k, params)[~valid]
            xm[~valid] = torch.where(step == 1, right_boundary(c, params), left_boundary(c, params))[~valid]
            c = c + step

        dpsi_dtheta = derivative_psi_theta(xm, tm, theta, k, params)
        dpsi_dtime = derivative_phi_time(xm, tm, theta, k, params)
        dphi_dtheta = dpsi_dtheta + dpsi_dtime * dthit_dtheta_cum
        der[:, :, k] = dphi_dtheta.reshape(n_batch, n_points)

    return der


def derivative_numeric_trace(phi_1, x, theta, params, time=1.0, h=1e-3):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    der = torch.empty((n_batch, n_points, d), device=x.device)

    # phi_1 = integrate_numeric(x, theta, params)
    for k in range(d):
        theta[:, k] += h
        phi_2 = integrate_numeric(x, theta, params, time)
        theta[:, k] -= h

        der[:, :, k] = (phi_2 - phi_1) / h
    return der


# %% GRADIENT SPACE

def derivative_space_numeric(x, theta, params, time=1.0, h=1e-3):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # phi_1 = integrate_numeric(x, theta, params, time)
    # phi_2 = integrate_numeric(x+h, theta, params, time)
    # der = (phi_2 - phi_1) / h
    # return phi_1, der

    # computation
    xe = torch.cat([x, x+h])

    phi = integrate_numeric(xe, theta, params, time)
    phi_1, phi_2 = torch.split(phi, n_points, dim=1)
    der = (phi_2 - phi_1) / h

    return phi_1, der

def derivative_space_closed_form(x, theta, params, time=1.0):
    # setup
    n_points = x.shape[-1]
    n_batch = theta.shape[0]
    d = theta.shape[1]

    # computation
    t = torch.ones_like(x) * time
    result = integrate_closed_form_trace(x, theta, params, time)
    phi = result[:, 0].reshape((n_batch, -1))
    tm = result[:, 1]
    cm = result[:, 2]

    # setup
    x = batch_effect(x, theta)
    params = precompute_affine(x, theta, params)

    dthit_dx = torch.zeros_like(x)
    dpsi_dx = torch.zeros_like(x)

    c = get_cell(x, params)
    valid = c == cm
    # dpsi_dx only on first valid cell
    dpsi_dx[valid] = derivative_psi_x(x, t, theta, params)[valid]
    # dthit_dx only on first non valid cell
    dthit_dx[~valid] = derivative_thit_x(x, t, theta, params)[~valid]

    xm = x.clone()
    while True:
        valid = c == cm
        if torch.all(valid):
            break
        step = torch.sign(cm - c)
        xm[~valid] = torch.where(step == 1, right_boundary(c, params), left_boundary(c, params))[~valid]
        c = c + step

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

    return torch.exp(t * a)

def derivative_psi_t(x, t, theta, params):
    A, r = get_affine(x, theta, params)
    c = get_cell(x, params)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return torch.exp(t * a) * (a * x + b)