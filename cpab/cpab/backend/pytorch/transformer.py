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

# Jit compile cpu source
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


# Jit compile gpu source
_gpu_success = False
# try:
#     cpab_gpu = load(name = 'cpab_gpu',
#                     sources = [_dir + '/transformer_cuda.cpp',
#                                _dir + '/transformer_cuda.cu',
#                                _dir + '/../core/cpab_ops.cu'],
#                     verbose=_verbose,
#                     with_cuda=True)
#     _gpu_success = True
#     if _verbose:
#         print(70*'=')
#         print('succesfully compiled gpu source')
#         print(70*'=')
# except Exception as e:
#     cpab_gpu = _notcompiled()
#     _gpu_success = False
#     if _verbose:
#         print(70*'=')
#         print('Unsuccesfully compiled gpu source')
#         print('Error was: ')
#         print(e)

# %%

def get_cell(grid, params):
    return cpab_cpu.get_cell(grid, params.xmin, params.xmax, params.nc)
    

def calc_velocity(grid, theta, params):
    return cpab_cpu.get_velocity(grid, theta, params.B, params.xmin, params.xmax, params.nc)


def transformer_rename(grid, theta, params, mode=None):
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
def transformer(grid, theta, params, mode=None):
    if grid.is_cuda and theta.is_cuda:
        if not params.use_slow and _gpu_success:
            if _verbose: print('using fast gpu implementation')
            return transformer_fast_gpu(grid, theta, params, mode)
        else:
            if _verbose: print('using slow gpu implementation')
            return transformer_slow(grid, theta, params, mode)
    else:
        if not params.use_slow and _cpu_success:
            if _verbose: print('using fast cpu implementation')
            return transformer_fast_cpu(grid, theta, params, mode)
        else:
            if _verbose: print('using slow cpu implementation')
            return transformer_slow(grid, theta, params, mode)

# %%
def transformer_slow(grid, theta, params, mode=None):
    pass

# %%
def transformer_fast_cpu(grid, theta, params, mode=None):
    if mode is None:
        mode = "closed_form"
    if mode == "closed_form":
        return _CPABFunction_ClosedForm_CPU.apply(grid, theta, params)
    elif mode == "numeric":
        return _CPABFunction_Numeric_CPU.apply(grid, theta, params)

#%%
class _CPABFunction_ClosedForm_CPU(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        output = cpab_cpu.integrate_closed_form_trace(grid, theta, params.B, params.xmin, params.xmax, params.nc)
        ctx.save_for_backward(output, grid, theta)
        grid_t = output[:,:,0]
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad [n_batch, n_points]
        # Grap input
        output, grid, theta = ctx.saved_tensors
        params = ctx.params
        grad_theta = cpab_cpu.derivative_closed_form_trace(output, grid, theta, params.B, params.xmin, params.xmax, params.nc) # [n_batch, n_points, d]

        # print(grad_output.shape, gradient.shape)
        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

#%%
class _CPABFunction_Numeric_CPU(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        grid_t = cpab_cpu.integrate_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2)
        ctx.save_for_backward(grid_t, grid, theta)
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad_output [n_batch, n_points]
        # Grap input
        params = ctx.params
        grid_t, grid, theta = ctx.saved_tensors

        h = 1e-2

        grad_theta = cpab_cpu.derivative_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)
        # grad_theta = cpab_cpu.derivative_numeric2(grid_t, grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

        gradient = []
        
        n_theta, d = theta.shape
        for k in range(d):
            # Permute theta
            thetap = theta.clone()
            thetap[:,k] += h

            grid_tp = cpab_cpu.integrate_numeric(grid, thetap, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2)

            diff = (grid_tp - grid_t) / h

            # Do finite gradient
            gradient.append((grad_output * diff).sum(dim=[1]))

        # Reshaping
        grad = torch.stack(gradient, dim = 1) # [n_theta, d]
        return None, grad, None
        gradient = cpab_cpu.derivative_closed_form_trace(newpoints, grid, theta, params.B, params.xmin, params.xmax, params.nc) # [n_batch, n_points, d]

        # print(grad.shape, gradient.shape)
        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad_theta = grad_output.mul(gradient.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]


