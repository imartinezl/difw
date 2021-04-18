import torch

from .interpolation import interpolate
from .transformer import transformer, gradient
from .transformer import get_cell, calc_velocity

# %%
def assert_version():
    numbers = torch.__version__.split(".")
    version = float(numbers[0] + "." + numbers[1])
    assert (
        version >= 1.0
    ), """ You are using a older installation of pytorch, please install 1.0.0
            or newer """


# %%
def to(x, dtype=torch.float32, device=None):
    if type(device) == str:
        device = torch.device("cuda") if device == "gpu" else torch.device("cpu")
    if torch.is_tensor(x):
        return x.detach().clone().type(dtype).to(device)
    return torch.tensor(x, dtype=dtype, device=device)


# %%
def tonumpy(x):
    return x.cpu().numpy()


# %%
def check_device(x, device_name):
    return (x.is_cuda) == (device_name == "gpu")


# %%
def backend_type():
    return torch.Tensor


# %%
def sample_transformation(d, n_sample=1, mean=None, cov=None, device="cpu"):
    device = torch.device("cpu") if device == "cpu" else torch.device("cuda")
    mean = torch.zeros(d, dtype=torch.float32, device=device) if mean is None else mean
    cov = torch.eye(d, dtype=torch.float32, device=device) if cov is None else cov
    distribution = torch.distributions.MultivariateNormal(mean, cov)
    return distribution.sample((n_sample,)).to(device)


# %%
def identity(d, n_sample=1, epsilon=0, device="cpu"):
    assert epsilon >= 0, "epsilon need to be larger than 0"
    device = torch.device("cpu") if device == "cpu" else torch.device("cuda")
    return torch.zeros(n_sample, d, dtype=torch.float32, device=device) + epsilon


# %%
def uniform_meshgrid(xmin, xmax, n_points, device="cpu"):
    device = torch.device("cpu") if device == "cpu" else torch.device("cuda")
    return torch.linspace(xmin, xmax, n_points, dtype=torch.float32, device=device)

# %%
def exp(*args, **kwargs):
    return torch.exp(*args, **kwargs)

def linspace(*args, **kwargs):
    return torch.linspace(*args, **kwargs)

def meshgrid(*args, **kwargs):
    return torch.meshgrid(*args, **kwargs)

def matmul(*args, **kwargs):
    return torch.matmul(*args, **kwargs)