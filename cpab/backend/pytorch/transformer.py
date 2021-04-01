# %%
import torch
from torch.utils.cpp_extension import load
from ...core.utility import get_dir, modes

#%% NOT COMPILED HELPER
class _notcompiled:
    # Small class, with structure similar to the compiled modules we can default
    # to. The class will never be called but the program can compile at run time
    def __init__(self):
        def f(*args):
            return None

        self.forward = f
        self.backward = f


#%% COMPILE AND LOAD

# TODO: change slow and fast for c++ / python or compiled / not compiled
_dir = get_dir(__file__)
_verbose = True

# Jit compile cpu source
_cpu_success = False
cpab_cpu = _notcompiled()
try:
    cpab_cpu = load(
        name="cpab_cpu",
        sources=[_dir + "/transformer.cpp",
                _dir + '/../../core/cpab.cpp'],
        extra_cflags=["-O0", "-g"], # TODO: change compilation flags to -O3 or -Ofast
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
cpab_gpu = _notcompiled()
try:
    cpab_gpu = load(
        name = 'cpab_gpu',
        sources = [_dir + '/transformer_cuda.cpp',
                _dir + '/transformer_cuda.cu',
                _dir + '/../../core/cpab_ops.cu'],
        extra_cflags=["-O0", "-g"], # TODO: change compilation flags to -O3 or -Ofast
        verbose=_verbose,
        with_cuda=True)
    _gpu_success = True
    if _verbose:
        print(70*'=')
        print('succesfully compiled gpu source')
        print(70*'=')
except Exception as e:
    cpab_gpu = _notcompiled()
    _gpu_success = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled gpu source')
        print('Error was: ')
        print(e)

# %% GET CELL

from .transformer_slow import get_cell as get_cell_slow
# TODO: review call to compiled cpu
def get_cell(grid, params):
    if grid.is_cuda:
        if not params.use_slow and _gpu_success:
            if _verbose: print('using fast gpu implementation')
            raise "NOT IMPLEMENTED" # TODO
            # return cpab_gpu.get_cell(grid, params.xmin, params.xmax, params.nc)
        else:
            if _verbose: print('using slow gpu implementation')
            return get_cell_slow(grid, params)
    else:
        if not params.use_slow and _cpu_success:
            if _verbose: print('using fast cpu implementation')
            return cpab_cpu.get_cell(grid, params.xmin, params.xmax, params.nc)
        else:
            if _verbose: print('using slow cpu implementation')
            return get_cell_slow(grid, params)

# %% CALC VELOCITY

from .transformer_slow import calc_velocity as calc_velocity_slow 
def calc_velocity(grid, theta, params):
    if grid.is_cuda and theta.is_cuda:
        if not params.use_slow and _gpu_success:
            if _verbose: print('using fast gpu implementation')
            raise "NOT IMPLEMENTED" # TODO
            # return cpab_gpu.get_velocity(grid, theta, params.B, params.xmin, params.xmax, params.nc)
        else:
            if _verbose: print('using slow gpu implementation')
            return calc_velocity_slow(grid, theta, params)
    else:
        if not params.use_slow and _cpu_success:
            if _verbose: print('using fast cpu implementation')
            return cpab_cpu.get_velocity(grid, theta, params.B, params.xmin, params.xmax, params.nc)
        else:
            if _verbose: print('using slow cpu implementation')
            return calc_velocity_slow(grid, theta, params)


# %% GRADIENT

# TODO: maybe also remove gradient? if we can use the backward function to return the gradient
def gradient(grid, theta, params, mode=None):
    if grid.is_cuda and theta.is_cuda:
        if not params.use_slow and _gpu_success:
            if _verbose: print('using fast gpu implementation')
            raise "NOT IMPLEMENTED" # TODO
            return gradient_fast_gpu(grid, theta, params, mode)
        else:
            if _verbose: print('using slow gpu implementation')
            return gradient_slow(grid, theta, params, mode)
    else:
        if not params.use_slow and _cpu_success:
            if _verbose: print('using fast cpu implementation')
            return gradient_fast_cpu(grid, theta, params, mode)
        else:
            if _verbose: print('using slow cpu implementation')
            return gradient_slow(grid, theta, params, mode)

# %% GRADIENT: SLOW / CPU + GPU
from .transformer_slow import derivative_numeric, derivative_closed_form

def gradient_slow(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)

    if mode == modes.closed_form:
        phi, der = derivative_closed_form(grid, theta, params)
        return der
    elif mode == modes.numeric:
        h = 1e-3
        phi, der = derivative_numeric(grid, theta, params, h)
        return der

# %% GRADIENT: FAST / CPU
def gradient_fast_cpu(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)
    
    if mode == modes.closed_form:
        return cpab_cpu.derivative_closed_form(grid, theta, params.B, params.xmin, params.xmax, params.nc)
    elif mode == modes.numeric:
        h = 1e-3
        return cpab_cpu.derivative_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)


# %% GRADIENT: FAST / GPU
def gradient_fast_gpu(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)

    if mode == modes.closed_form:
        pass # TODO
        return cpab_gpu.derivative_closed_form(grid, theta, params.B, params.xmin, params.xmax, params.nc)
    elif mode == modes.numeric:
        h = 1e-3
        pass # TODO
        return cpab_gpu.derivative_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)

# %% TRANSFORMER

def transformer(grid, theta, params, mode=None):
    if grid.is_cuda and theta.is_cuda:
        if not params.use_slow and _gpu_success:
            if _verbose: print('using fast gpu implementation')
            pass # TODO
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

# %% TRANSFORMER: SLOW / CPU + GPU

def transformer_slow(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)

    if mode == modes.closed_form:
        return Transformer_slow_closed_form.apply(grid, theta, params)
    elif mode == modes.numeric:
        return Transformer_slow_numeric.apply(grid, theta, params)

#%% TRANSFORMER: SLOW / CPU + GPU / CLOSED-FORM 
from .transformer_slow import integrate_closed_form_trace, derivative_closed_form_trace

class Transformer_slow_closed_form(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        output = integrate_closed_form_trace(grid, theta, params)
        ctx.save_for_backward(output, grid, theta)
        n_batch = theta.shape[0]
        grid_t = output[:,0].reshape((n_batch, -1))
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad [n_batch, n_points]
        output, grid, theta = ctx.saved_tensors
        params = ctx.params
        grad_theta = derivative_closed_form_trace(output, grid, theta, params) # [n_batch, n_points, d]

        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

#%% TRANSFORMER: SLOW / CPU + GPU / NUMERIC 
from .transformer_slow import integrate_numeric, derivative_numeric_trace

class Transformer_slow_numeric(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        grid_t = integrate_numeric(grid, theta, params)
        ctx.save_for_backward(grid_t, grid, theta)
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad_output [n_batch, n_points]
        params = ctx.params
        grid_t, grid, theta = ctx.saved_tensors

        h = 1e-2

        grad_theta = derivative_numeric_trace(grid_t, grid, theta, params, h)
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]
# %% #%% TRANSFORMER: FAST / CPU 
def transformer_fast_cpu(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)

    if mode == modes.closed_form:
        return Transformer_fast_cpu_closed_form.apply(grid, theta, params)
    elif mode == modes.numeric:
        return Transformer_fast_cpu_numeric.apply(grid, theta, params)

# %% #%% TRANSFORMER: FAST / CPU / CLOSED-FORM
class Transformer_fast_cpu_closed_form(torch.autograd.Function):
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
        output, grid, theta = ctx.saved_tensors
        params = ctx.params
        grad_theta = cpab_cpu.derivative_closed_form_trace(output, grid, theta, params.B, params.xmin, params.xmax, params.nc) # [n_batch, n_points, d]

        # print(grad_output.shape, gradient.shape)
        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

# %% #%% TRANSFORMER: FAST / CPU / NUMERIC
class Transformer_fast_cpu_numeric(torch.autograd.Function):
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
        params = ctx.params
        grid_t, grid, theta = ctx.saved_tensors

        h = 1e-2

        grad_theta = cpab_cpu.derivative_numeric_trace(grid_t, grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

# %% #%% TRANSFORMER: FAST / GPU
def transformer_fast_gpu(grid, theta, params, mode=None):
    modes.check_mode(mode)
    mode = modes.default(mode)

    if mode == modes.closed_form:
        return Transformer_fast_gpu_closed_form.apply(grid, theta, params)
    elif mode == modes.numeric:
        return Transformer_fast_gpu_numeric.apply(grid, theta, params)

# %% #%% TRANSFORMER: FAST / GPU / CLOSED-FORM
class Transformer_fast_gpu_closed_form(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        output = cpab_gpu.integrate_closed_form_trace(grid, theta, params.B, params.xmin, params.xmax, params.nc)
        ctx.save_for_backward(output, grid, theta)
        grid_t = output[:,:,0]
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad [n_batch, n_points]
        output, grid, theta = ctx.saved_tensors
        params = ctx.params
        grad_theta = cpab_gpu.derivative_closed_form_trace(output, grid, theta, params.B, params.xmin, params.xmax, params.nc) # [n_batch, n_points, d]

        # print(grad_output.shape, gradient.shape)
        # NOTE: we have to permute the gradient in order to do the element-wise product
        # Then, the gradient must be summarized for all grid points
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]

# %% #%% TRANSFORMER: FAST / GPU / NUMERIC
class Transformer_fast_gpu_numeric(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, grid, theta, params):
        ctx.params = params
        grid_t = cpab_gpu.integrate_numeric(grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2)
        ctx.save_for_backward(grid_t, grid, theta)
        return grid_t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output): # grad_output [n_batch, n_points]
        params = ctx.params
        grid_t, grid, theta = ctx.saved_tensors

        h = 1e-2

        grad_theta = cpab_gpu.derivative_numeric_trace(grid_t, grid, theta, params.B, params.xmin, params.xmax, params.nc, params.nSteps1, params.nSteps2, h)
        grad = grad_output.mul(grad_theta.permute(2,0,1)).sum(dim=(2)).t()
        return None, grad, None # [n_batch, d]


