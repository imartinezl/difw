# %%
import numpy as np
from .interpolation import interpolate
from .transformer import integrate_numeric, integrate_closed_form, derivative_numeric, derivative_closed_form
from .transformer import get_cell, get_velocity, batch_effect


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

def calc_velocity(grid, theta, params):
    grid = batch_effect(grid, theta)
    v = get_velocity(grid, theta, params)
    return v.reshape(theta.shape[0], -1)

def transformer(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return integrate_closed_form(grid, theta, params)
    elif mode == "numeric":
        return integrate_numeric(grid, theta, params)

def gradient(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return derivative_closed_form(grid, theta, params)
    elif mode == "numeric":
        h = 1e-3
        return derivative_numeric(grid, theta, params, h)
