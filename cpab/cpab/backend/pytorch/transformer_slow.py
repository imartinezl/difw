# %%

import torch

# %%
def batch_effect(x, theta):
    n_batch = theta.shape[0]
    n_points = x.shape[-1]
    x = torch.broadcast_to(x, (n_batch, n_points)).flatten()
    return x


# FUNCTIONS

def get_cell(x, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc

    c = torch.floor((x - xmin) / (xmax - xmin) * nc)
    c = torch.clamp(c, 0, nc - 1).long()
    return c


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_batch = theta.shape[0]
        n_points = x.shape[-1]
        repeat = int(n_points / n_batch)

        # r = np.broadcast_to(np.arange(n_batch), [n_points, n_batch]).T
        # TODO: here we suppose batch effect has been already executed
        r = torch.arange(n_batch).repeat_interleave(repeat).long().to(x.device)

        A = params.B.mm(theta.T).T.reshape(n_batch, -1, 2).to(x.device)

        return A, r


def precompute_affine(x, theta, params):
    params = params.copy()
    params.precomputed = False
    params.A, params.r = get_affine(x, theta, params)
    params.precomputed = True
    return params


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
    x2 = torch.exp(t * a) * (x + (b / a)) - (b / a)
    psi = torch.where(cond, x1, x2)
    return psi



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


def integrate_numeric(x, theta, params):
    # setup
    x = batch_effect(x, theta)
    n_batch = theta.shape[0]
    t = 1
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
        xPrev = torch.where(c == cTemp, xTemp, xNum)
        c = get_cell(xPrev, params)
    return xPrev.reshape((n_batch, -1))

