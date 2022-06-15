# %%
import numpy as np
from .interpolation import interpolate, interpolate_grid as interpolate_grid_wrapper
from .transformer import (
    integrate_numeric,
    integrate_closed_form,
    derivative_numeric,
    derivative_closed_form,
    derivative_space_numeric,
    derivative_space_closed_form
)
from .transformer import get_cell, get_velocity, batch_effect
from ...core.utility import methods

# %%
def assert_version():
    numbers = np.__version__.split(".")
    version = float(numbers[0] + "." + numbers[1])
    assert (
        version >= 1.14
    ), """ You are using a older installation of numpy, please install 1.15.
            or newer """


# %%
half = np.float16
single = np.float32
double = np.float64


def to(x, dtype=np.float32, device=None):
    return np.array(x)


# %%
def tonumpy(x):
    return x


# %%
def check_device(x, device_name):
    return True  # always return true, because device can only be cpu


# %%
def backend_type():
    return np.ndarray


# %%
def sample_transformation(d, n_sample=1, mean=None, cov=None, device="cpu"):
    mean = np.zeros(d, dtype=np.float32) if mean is None else mean
    cov = np.eye(d, dtype=np.float32) if cov is None else cov
    samples = np.random.multivariate_normal(mean, cov, size=n_sample)
    return samples


# %%
def identity(d, n_sample=1, epsilon=0, device="cpu"):
    assert epsilon >= 0, "epsilon need to be larger than or 0"
    return np.zeros((n_sample, d), dtype=np.float32) + epsilon


# %%
def uniform_meshgrid(xmin, xmax, n_points, device="cpu"):
    return np.linspace(xmin, xmax, n_points)


# %%
def calc_velocity(grid, theta, params):
    grid = batch_effect(grid, theta)
    v = get_velocity(grid, theta, params)
    return v.reshape(theta.shape[0], -1)


# %%
def exp(*args, **kwargs):
    return np.exp(*args, **kwargs)


def linspace(*args, **kwargs):
    return np.linspace(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return np.meshgrid(*args, **kwargs)


def matmul(*args, **kwargs):
    return np.matmul(*args, **kwargs)


def max(*args, **kwargs):
    return np.max(*args, **kwargs)


def ones(*args, **kwargs):
    return np.ones(*args, **kwargs)


def pdist(c):
    x, y = np.meshgrid(c, c)
    return np.abs(x - y)


# %%


def transformer(grid, theta, params, method=None, time=1.0):
    methods.check(method)
    method = methods.default(method)

    if method == methods.closed_form:
        return integrate_closed_form(grid, theta, params, time)
    elif method == methods.numeric:
        return integrate_numeric(grid, theta, params, time)


# %%
def gradient(grid, theta, params, method=None, time=1.0):
    methods.check(method)
    method = methods.default(method)

    if method == methods.closed_form:
        phi, der = derivative_closed_form(grid, theta, params, time)
        return der
    elif method == methods.numeric:
        h = 1e-2
        phi, der = derivative_numeric(grid, theta, params, time, h)
        return der

# %%
def gradient_space(grid, theta, params, method=None, time=1.0):
    methods.check(method)
    method = methods.default(method)

    if method == methods.closed_form:
        phi, der = derivative_space_closed_form(grid, theta, params, time)
        return der
    elif method == methods.numeric:
        h = 1e-2
        phi, der = derivative_space_numeric(grid, theta, params, time, h)
        return der

# %%
def interpolate_grid(transformed_grid, params):
    return interpolate_grid_wrapper(transformed_grid)