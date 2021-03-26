# %%
import torch
from torch.utils.cpp_extension import load
from ...core.utility import get_dir

#%%
class _notcompiled:
    # Small class, with structure similar to the compiled modules we can default
    # to. The class will never be called but the program can compile at run time
    def __init__(self):
        def f(*args):
            return None

        self.forward = f
        self.backward = f


#%%
_dir = get_dir(__file__)
_verbose = False

try:
    cpab_cpu = load(
        name="cpab_cpu",
        sources=[_dir + "/transformer.cpp",
                _dir + '/../../core/cpab.cpp'],
        extra_cflags=["-O0", "-g"],
        verbose=_verbose,
    )
    _cpu_success = True
    if _verbose:
        print(70 * "=")
        print("succesfully compiled cpu source")
        print(70 * "=")
except Exception as e:
    cpab_cpu = _notcompiled()
    _cpu_success = False
    if _verbose:
        print(70 * "=")
        print("Unsuccesfully compiled cpu source")
        print("Error was: ")
        print(e)
        print(70 * "=")

# %%

def get_cell(grid, params):
    return cpab_cpu.get_cell(grid, params.xmin, params.xmax, params.nc)
    

def calc_velocity(grid, theta, params):
    return cpab_cpu.get_velocity(grid, theta, params.B, params.xmin, params.xmax, params.nc)


def transformer(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return cpab_cpu.integrate_closed_form(grid, theta, params.B, params.xmin, params.xmax, params.nc)
    elif mode == "numeric":
        return cpab_cpu.integrate_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2)

def gradient(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return cpab_cpu.derivative_closed_form(grid, theta, params.B, params.xmin, params.xmax, params.nc)
    elif mode == "numeric":
        h = 1e-3
        return cpab_cpu.derivative_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)

# %%
def CPAB_transformer(grid, theta, params, mode=None):
    if grid.is_cuda and theta.is_cuda:
        pass
    else:
        if not params.use_slow and _cpu_success:
            if _verbose: print('using fast cpu implementation')
            return CPAB_transformer_fast(grid, theta, params, mode)
        else:
            if _verbose: print('using slow cpu implementation')
            return CPAB_transformer_slow(grid, theta, params, mode)

# %%
def CPAB_transformer_slow(grid, theta, params, mode=None):
    pass

# %%
def CPAB_transformer_fast(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return _CPABFunction_AnalyticGrad.apply(grid, theta, params)
    elif mode == "numeric":
        return _CPABFunction_NumericGrad.apply(grid, theta, params)

#%%
class _CPABFunction_AnalyticGrad(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        newpoints = cpab_cpu.forward(grid, theta, params.B, params.xmin, params.xmax, params.nc)
        ctx.save_for_backward(newpoints, grid, theta)
        return newpoints[:,:,0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad_output [n_batch, n_points]
        # Grap input
        newpoints, grid, theta = ctx.saved_tensors
        params = ctx.params
        gradient = cpab_cpu.backward(newpoints, grid, theta, params.B, params.xmin, params.xmax, params.nc) # [n_batch, n_points, d]

        # print(grad.shape, gradient.shape)
        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad_theta = grad_output.mul(gradient.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad_theta, None # [n_batch, d]