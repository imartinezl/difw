# %%
import numpy as np


# TODO: review channels dimension and overall data dimension
# for sure, we need two dimensions:
#   1. One for batch_size
#   2. Another for time series length
# not sure why we need a third one


def interpolate(ndim, data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    width = data.shape[1]
    n_channels = data.shape[2]
    out_width = outsize[0]

    # Extract points
    x = grid[:, 0].flatten()

    # Scale to domain
    x = x * (width - 1)

    # Do sampling
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1

    # Clip values
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)

    # Batch effect
    batch_size = out_width
    batch_idx = np.arange(n_batch).repeat(batch_size)

    # Index
    c0 = data[batch_idx, x0, :]
    c1 = data[batch_idx, x1, :]

    # Interpolation weights
    xd = (x - x0.astype(np.float32)).reshape((-1, 1))

    # Do interpolation
    c = c0 * (1 - xd) + c1 * xd

    # Reshape
    new_data = np.reshape(c, (n_batch, out_width, n_channels))
    return new_data


def assert_version():
    numbers = np.__version__.split(".")
    version = float(numbers[0] + "." + numbers[1])
    assert (
        version >= 1.14
    ), """ You are using a older installation of numpy, please install 1.15.
            or newer """


def to(x, dtype=np.float32, device=None):
    return np.array(x)


def tonumpy(x):
    return x


def check_device(x, device_name):
    return True  # always return true, because device can only be cpu


def backend_type():
    return np.ndarray


def sample_transformation(d, n_sample=1, mean=None, cov=None, device="cpu"):
    mean = np.zeros(d, dtype=np.float32) if mean is None else mean
    cov = np.eye(d, dtype=np.float32) if cov is None else cov
    samples = np.random.multivariate_normal(mean, cov, size=n_sample)
    return samples


def identity(d, n_sample=1, epsilon=0, device="cpu"):
    assert epsilon >= 0, "epsilon need to be larger than or 0"
    return np.zeros((n_sample, d), dtype=np.float32) + epsilon


def uniform_meshgrid(xmin, xmax, n_points, device="cpu"):
    return np.linspace(xmin, xmax, n_points)


def get_cell(x, params):
    xmin, xmax, nc = params.xmin, params.xmax, params.nc

    c = np.floor((x - xmin) / (xmax - xmin) * nc)
    c = np.clip(c, 0, nc - 1).astype(np.int32)
    return c


# TODO: remove
def get_velocity_works_old(x, theta, params):
    B, xmin, xmax, nc = params.B, params.xmin, params.xmax, params.nc

    n_theta = theta.shape[0]
    n_points = x.shape[-1]

    batch_idx = nc * np.arange(n_theta)
    batch_idx = np.broadcast_to(batch_idx, [n_points, n_theta])
    batch_idx = batch_idx.flatten().T.astype(np.int32)

    batch_size, d = theta.shape
    A = B.dot(theta.T).T.reshape(-1, 2)
    c = get_cell(x, params)
    a = A[c, 0].reshape(n_theta, -1)
    b = A[c, 1].reshape(n_theta, -1)
    return a * x + b


def get_velocity_works(x, theta, params):
    B, xmin, xmax, nc = params.B, params.xmin, params.xmax, params.nc

    n_theta = theta.shape[0]
    n_points = x.shape[-1]

    r = np.broadcast_to(np.arange(n_theta), [n_points, n_theta]).T
    c = get_cell(x, params)

    A = B.dot(theta.T).T.reshape(n_theta, -1, 2)
    a = A[r, c, 0]
    b = A[r, c, 1]
    return a * x + b


def get_affine(x, theta, params):
    if params.precomputed:
        return params.A, params.r
    else:
        n_theta = theta.shape[0]
        n_points = x.shape[-1]

        A = params.B.dot(theta.T).T.reshape(n_theta, -1, 2)
        r = np.broadcast_to(np.arange(n_theta), [n_points, n_theta]).T

        return A, r


def precompute_affine(x, theta, params):
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


def get_psi_works(x, t, theta, params):
    B, xmin, xmax, nc = params.B, params.xmin, params.xmax, params.nc

    n_theta = theta.shape[0]
    n_points = x.shape[-1]

    r = np.broadcast_to(np.arange(n_theta), [n_points, n_theta]).T
    c = get_cell(x, params)

    A = B.dot(theta.T).T.reshape(n_theta, -1, 2)
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


def get_hit_time_works(x, theta, params):
    B = params.B

    c = get_cell(x, params)
    v = get_velocity(x, theta, params)
    xc = np.where(v >= 0, right_boundary(c, params), left_boundary(c, params))

    n_theta = theta.shape[0]
    n_points = x.shape[-1]

    r = np.broadcast_to(np.arange(n_theta), [n_points, n_theta]).T
    c = get_cell(x, params)

    A = B.dot(theta.T).T.reshape(n_theta, -1, 2)
    a = A[r, c, 0]
    b = A[r, c, 1]

    cond = a == 0
    x1 = (xc - x) / b
    x2 = np.log((xc + b / a) / (x + b / a)) / a
    thit = np.where(cond, x1, x2)
    return thit


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


eps = np.finfo(float).eps
# eps = 1e-12
# np.seterr(divide='ignore', invalid='ignore')


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


def integrate_analytic_works(x, t, theta, params):
    nc = params.nc
    cont = 0
    # result = [[x, t]]
    # print("########### NEW POINT")
    n_theta = theta.shape[0]
    n_points = x.shape[-1]
    valid = np.full((n_theta, n_points), False)

    while True:
        c = get_cell(x, params)
        v = get_velocity(x, theta, params)
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        psi = get_psi(x, t, theta, params)

        cond1 = left <= psi
        cond2 = psi <= right
        valid = np.logical_and(cond1, cond2)
        done = np.alltrue(valid)
        if done:
            # print(cont)
            return phi
        return psi.reshape(params.n_theta, -1)

        thit = get_hit_time(x, theta, params)
        t = np.where(valid, t, t - thit)
        x = np.where(psi < left, left, x)
        x = np.where(psi > right, right, x)
        # x[psi < left] = left[psi < left]
        # x[psi > right] = left[psi > right]
        # print(thit, thit.shape)
        # print(thit[~valid], thit[~valid].shape)
        cont += 1
        if cont > nc:
            raise BaseException
    return None


def integrate_analytic(x, t, theta, params):
    n_theta, n_points, nc = params.n_theta, params.n_points, params.nc
    cont = 0
    phi = np.empty((n_theta, n_points))
    done = np.full((n_theta, n_points), False)

    x = x[~done]
    t = t[~done]
    params.r = params.r[~done]

    # result = [[x, t]]

    while True:
        c = get_cell(x, params)
        left = left_boundary(c, params)
        right = right_boundary(c, params)
        psi = get_psi(x, t, theta, params)

        phi[~done] = psi
        valid = np.logical_and(left <= psi, psi <= right)
        if np.alltrue(valid):
            return phi

        x, t, params.r = x[~valid], t[~valid], params.r[~valid]
        t -= get_hit_time(x, theta, params)
        x = np.clip(psi[~valid], left[~valid], right[~valid])

        # result.append([x, t])
        done[~done] = valid

        cont += 1
        if cont > nc:
            raise BaseException
    return None


compiled = False
# TODO: change name with integrate or integration
def CPAB_transformer(points, theta, params):
    params = params.copy()
    n_theta = theta.shape[0]
    n_points = points.shape[-1]
    points = np.broadcast_to(points, (n_theta, n_points))
    params.n_theta, params.n_points = n_theta, n_points

    params = precompute_affine(points, theta, params)

    t = np.ones((n_theta, n_points))
    return integrate_analytic(points, t, theta, params)

    t = 1
    return integrate_numeric(points, t, theta, params)

    if compiled:
        return CPAB_transformer_fast(points, theta, params)
    else:
        return CPAB_transformer_slow(points, theta, params)


def CPAB_transformer_slow(points, theta, params):
    pass


def CPAB_transformer_fast(points, theta, params):
    pass
    # TODO: jit compile cpp code into callable python code


transformer = CPAB_transformer


class Solver:
    def __init__(self, tess):
        self.tess = tess

        self.xr = self.tess.xr
        self.nc = self.tess.nc
        # self.cs = self.xr / self.nc

        self.theta = np.zeros((self.tess.d))
        self.set_theta(self.theta)

    def get_cell(self, x):
        # c = np.floor(x / self.xr * self.nc)
        c = np.floor((x - self.tess.xmin) / self.xr * self.nc)
        return np.clip(c, 0, self.nc - 1).astype(int)

    def get_velocity(self, x):
        c = self.get_cell(x)
        a = self.A[c, 0]
        b = self.A[c, 1]
        return a * x + b

    def get_psi(self, x, t):
        c = self.get_cell(x)
        a = self.A[c, 0]
        b = self.A[c, 1]
        psi = np.where(a == 0, x + t * b, np.exp(t * a) * (x + (b / a)) - (b / a))
        return psi

    def get_hit_time(self, x):
        c = self.get_cell(x)
        v = self.get_velocity(x)
        xc = np.where(v >= 0, self.right_boundary(c), self.left_boundary(c))
        a = self.A[c, 0]
        b = self.A[c, 1]
        tcross = np.where(a == 0, (xc - x) / b, np.log((xc + b / a) / (x + b / a)) / a)
        return tcross

    def right_boundary(self, c):
        return (
            self.tess.xmin + (c + 1) * self.xr / self.nc + 1e-12
        )  # TODO: replace with machine epsilon
        return (c + 1) * self.xr / self.nc + 1e-12

    def left_boundary(self, c):
        return self.tess.xmin + c * self.xr / self.nc - 1e-12
        return c * self.xr / self.nc - 1e-12

    def integrate(self, x, t):
        cont = 0
        result = [[x, t]]
        # print("########### NEW POINT")
        while True:
            c = self.get_cell(x)
            v = self.get_velocity(x)
            left = self.left_boundary(c)
            right = self.right_boundary(c)

            psi = self.get_psi(x, t)
            # print(c, left, psi, right, x, v, t)

            if left <= psi <= right:
                return psi, result
            else:
                tcross = self.get_crossing(x)
                t -= tcross

            if psi < left:
                x = left
            elif psi > right:
                x = right

            result.append([x, t])

            cont += 1
            if cont > self.nc:  # TODO: replace with nc-c1+1
                return psi, result
                raise BaseException("Reached MAX ITER")
                break

    def derivative(self, result, j):
        dthit_dtheta_cum = 0.0
        for x, t in result[:-1]:
            dthit_dtheta = self.derivative_thit_theta(x, j)
            dthit_dtheta_cum -= dthit_dtheta

        x, t = result[-1]
        dpsi_dtheta = self.derivative_psi_theta(x, t, j)
        dpsi_dtime = self.derivative_phi_time(x, t, j)
        dphi_dtheta = dpsi_dtheta + dpsi_dtime * dthit_dtheta_cum
        return dphi_dtheta

    def derivative_psi_theta(self, x, t, j):
        c = self.get_cell(x)
        a = self.A[c, 0]
        b = self.A[c, 1]

        ak = self.tess.B[2 * c, j]
        bk = self.tess.B[2 * c + 1, j]

        if a == 0:
            dpsi_dtheta = t * (x * ak + bk)
        else:
            dpsi_dtheta = (
                ak * t * np.exp(a * t) * (x + b / a)
                + (np.exp(t * a) - 1) * (bk * a - ak * b) / a ** 2
            )
        return dpsi_dtheta

    def derivative_phi_time(self, x, t, j):
        c = self.get_cell(x)
        a = self.A[c, 0]
        b = self.A[c, 1]

        if a == 0:
            dpsi_dtime = b
        else:
            dpsi_dtime = np.exp(t * a) * (a * x + b)
        return dpsi_dtime

    def derivative_thit_theta(self, x, j):
        c = self.get_cell(x)
        a = self.A[c, 0]
        b = self.A[c, 1]

        ak = self.tess.B[2 * c, j]
        bk = self.tess.B[2 * c + 1, j]

        v = self.get_velocity(x)

        xc = np.where(v >= 0, self.right_boundary(c), self.left_boundary(c))

        if a == 0:
            dthit_dtheta = (x - xc) * bk / b ** 2
        else:
            d1 = -ak * np.log((a * xc + b) / (a * x + b)) / a ** 2
            d2 = (x - xc) * (bk * a - ak * b) / (a * (a * x + b) * (a * xc + b))
            dthit_dtheta = d1 + d2
        return dthit_dtheta

    def get_numerical_phi(self, x, t, nSteps2):
        yn = x
        deltaT = t / nSteps2
        for j in range(nSteps2):
            c = self.get_cell(yn)
            midpoint = yn + deltaT / 2 * self.get_velocity(yn)
            c = self.get_cell(midpoint)
            yn = yn + deltaT * self.get_velocity(midpoint)
        return yn

    def integrate_numerical(self, x, t, nSteps1=10, nSteps2=100):
        xPrev = x
        deltaT = t / nSteps1
        c = self.get_cell(xPrev)
        for j in range(nSteps1):
            xTemp = self.get_psi(xPrev, deltaT)
            cTemp = self.get_cell(xTemp)
            xNum = self.get_numerical_phi(xPrev, deltaT, nSteps2)
            if c == cTemp:
                xPrev = xTemp
            else:
                xPrev = xNum
            c = self.get_cell(xPrev)
        return xPrev
