# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import difw
from tqdm import tqdm 

# %% TEST 

tess_size = 10
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 100
batch_size = 2
method = "closed_form"
basis = "svd"
basis = "sparse"
basis = "rref"

T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=1)
theta = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, theta, method=method)
grid_t2 = T.transform_grid(grid_t, theta, method=method)

v = T.calc_velocity(grid, theta)
plt.plot(grid, v.T)
plt.figure()
plt.plot(grid, grid_t.T)
plt.plot(grid, grid_t2.T)

# %%

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=0.1)
theta = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, theta)
grid_t2 = T.transform_grid(grid_t, theta)
grid_t3 = T.transform_grid(grid_t2, theta)

plt.plot(grid)
plt.plot(grid_t.T)
plt.plot(grid_t2.T)
plt.plot(grid_t3.T)

# %%
channels = 3
width = 100
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = 0.5 + np.sin(x-noise)
data = torch.tensor(data)

N = 0
data_t = T.transform_data_ss(data, theta / 2**N, width, N=N)
print(data.shape, data_t.shape)
# data_t2 = T.transform_data(data_t, theta, width)
# data_t3 = T.transform_data(data_t2, theta, width)
# data_t4 = T.interpolate(data, grid_t3, width)

plt.figure()
plt.plot(data[:,:,1].T)
plt.plot(data_t[:,:,1].T)
# plt.plot(data_t2[:,:,0].T)
# plt.plot(data_t3[:,:,0].T)
# plt.plot(data_t4[:,:,0].T)

# T.visualize_velocity(theta)
# T.visualize_deformgrid(theta)
# T.visualize_gradient(theta)

T.visualize_deformdata(data, 2**N * theta)

# %%

batch_size = 3
channels = 4
outsize = 100

a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, outsize, channels))
x = np.linspace(a, b, outsize, axis=1)
dataA = np.sin(x)
dataB = np.sin(x + 0.3)

dataA = torch.tensor(dataA)
dataB = torch.tensor(dataB)

tess_size = 10
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = True
method = "closed_form"
basis = "svd"
basis = "sparse"
basis = "rref"

T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=0)
theta = torch.autograd.Variable(theta, requires_grad=True)

lr = 1e-3
optimizer = torch.optim.Adam([theta], lr=lr)
# optimizer = torch.optim.SGD([theta_2], lr=1e1)

# torch.set_num_threads(1)
loss_values = []
maxiter = 50
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        N = 4
        dataT = T.transform_data_ss(dataA, theta, outsize, N=N)
        loss = torch.norm(dataT - dataB, dim=1).mean()
        loss.backward()
        optimizer.step()

        if torch.any(torch.isnan(theta)):
            print("AHSDASD")
            break

        loss_values.append(loss.item())
        pb.update()
        pb.set_postfix({'loss': loss.item()})
    pb.close()

plt.figure()
plt.plot(loss_values)

plt.figure()

plt.plot(dataA[:,:,0].T)
plt.plot(dataB[:,:,0].T)
plt.plot(dataT.detach()[:,:,0].T)

T.visualize_deformgrid(2**N*theta.detach())

# %%
%%timeit -r 20 -n 1

N = 3
t = 1.0 / 2**N
method = "numeric"
method = "closed_form"
# grid_t1 = T.transform_grid(grid, theta, method, time=t)
# grid_t1_num = T.transform_grid(grid, theta, "numeric", time=t)
# grid_t2 = T.transform_grid_ss(grid, theta, method, time=t, N=0)
grad_t = T.gradient_grid(grid, theta, method, time=t)
# plt.plot(grad_t[0,:,:])

# %%
%%timeit -r 200 -n 1

N = 2
t = 1.0 / 2**N 
method = "closed_form"
grid_t2 = T.transform_grid(grid, theta, method, time=t)
for j in range(N):
    grid_t2 = T.backend.interpolate_grid(grid_t2)
    # grid_t3 = np.interp(grid_t2[0], grid_t2[0], grid_t2[0])[np.newaxis, :]
# grad_t = T.gradient_grid(grid, theta, method, time=t)
# error = np.linalg.norm(grid_t - grid_t2)

# %%
t = 1.0
# T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta, time=t)
T.visualize_deformgrid(theta, method="numeric", time=t)
T.visualize_gradient(theta, time=t)
T.visualize_gradient(theta, "numeric", time=t)

plt.figure()
# plt.plot(T.transform_grid(grid, theta, method="numeric", time=0.0)[0]-grid)
plt.plot(T.transform_grid(grid, theta, method="closed_form", time=0.0)[0]-grid)

# %% OPTIMIZATION BY GRADIENT

tess_size = 500
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 100
batch_size = 1
method = "closed_form"
basis = "sparse"
basis = "svd"
basis = "rref"

from scipy.interpolate import interp1d

results= {}
for basis in ["sparse", "svd", "rref", "qr"]:
    start_time = time.time()

    T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow
    # T.params.B = torch.tensor(B2, dtype=torch.float)

    grid = T.uniform_meshgrid(outsize)

    torch.manual_seed(0)
    theta_1 = T.sample_transformation(batch_size)*10
    theta_1 = T.identity(batch_size, epsilon=1.0)
    grid_t1 = T.transform_grid(grid, theta_1, method)
    x = [0, .2, .4, .6, .8, 1]
    y = [0, .5, .6, .7, .75, 1]
    f = interp1d(x, y, kind="linear", fill_value="extrapolate")
    xv = np.linspace(0,1,outsize)
    yv = f(xv)
    # grid_t1 = torch.tile(torch.linspace(0,1,outsize)**2, (batch_size,1))
    grid_t1 = torch.tile(torch.tensor(yv, dtype=torch.float), (batch_size, 1))

    torch.manual_seed(1)
    theta_2 = torch.autograd.Variable(T.sample_transformation(batch_size), requires_grad=True)
    theta_2 = torch.autograd.Variable(T.identity(batch_size, epsilon=0.0), requires_grad=True)

    lr = 1e-2
    optimizer = torch.optim.Adam([theta_2], lr=lr)
    # optimizer = torch.optim.SGD([theta_2], lr=1e1)

    # torch.set_num_threads(1)
    loss_values = []
    maxiter = 250
    with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
        for i in range(maxiter):
            optimizer.zero_grad()
            
            # output = T.backend.cpab_cpu.integrate_closed_form_trace(grid, theta_2, T.params.B, T.params.xmin, T.params.xmax, T.params.nc)
            # grad_theta = T.backend.cpab_cpu.derivative_closed_form_trace(output, grid, theta_2, T.params.B, T.params.xmin, T.params.xmax, T.params.nc) # [n_batch, n_points, d]
            # theta_backup = theta_2.clone()
            
            grid_t2 = T.transform_grid(grid, theta_2, method=method)
            loss = torch.norm(grid_t2 - grid_t1)
            loss = torch.norm(grid_t2 - grid_t1, dim=1).mean()
            loss.backward()
            optimizer.step()

            if torch.any(torch.isnan(theta_2)):
                print("AHSDASD")
                break

            loss_values.append(loss.item())
            pb.update()
            pb.set_postfix({'loss': loss.item()})
        pb.close()

    end_time = time.time()

    if False:
        plt.figure()
        plt.plot(loss_values)
        plt.axhline(color="black", ls="dashed")
        # plt.yscale('log')

        plt.figure()
        plt.plot(grid, grid_t1.t())
        plt.plot(grid, grid_t2.detach().t())

        plt.figure()
        plt.plot(grid_t1.t() - grid_t2.detach().t())
        plt.axhline(color="black", ls="dashed")

    results[basis] = {}
    results[basis]["loss"] = loss_values
    results[basis]["B"] = T.params.B
    results[basis]["grid_t1"] = grid_t1.detach().numpy()
    results[basis]["grid_t2"] = grid_t2.detach().numpy()
    results[basis]["theta"] = theta_2.detach().numpy()
    results[basis]["time"] = end_time - start_time

# %%

plt.figure()
for k, v in results.items():
    plt.plot(v["loss"])
plt.title("Loss")
plt.legend(results.keys())
plt.axhline(0, ls="dashed", c="black")

plt.figure()
plt.plot(v["grid_t1"].T, c="black")
for k, v in results.items():
    plt.plot(v["grid_t2"].T)
plt.title("Grid Deform")
plt.legend(["Original"] + list(results.keys()))

print("Loss")
for k, v in results.items():
    print(k, ": ", np.mean(v["loss"]))

print("Time")
for k, v in results.items():
    print(k, ": ", v["time"])

print("Theta")
for k, v in results.items():
    print(k, ": ", np.linalg.norm(v["theta"]))

# %% OPTIMIZATION BY MCMC SAMPLING

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1

T = difw.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

theta_ref = T.sample_transformation(batch_size)
grid_ref = T.transform_grid(grid, theta_ref)

current_sample = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, current_sample)

current_error = np.linalg.norm(grid_t - grid_ref)
accept_ratio = 0

samples = []
maxiter = 1500
pb = tqdm(desc='Alignment of samples', unit='samples', total=maxiter)
for i in range(maxiter):
    # Sample and transform 
    theta = T.sample_transformation(batch_size, mean=current_sample.flatten())
    grid_t = T.transform_grid(grid, theta, method="closed_form")

    # Calculate new error
    new_error = np.linalg.norm(grid_t - grid_ref)

    samples.append( T.backend.tonumpy(theta[0]))
    
    if new_error < current_error:
        current_sample = theta
        current_error = new_error
        accept_ratio += 1
    pb.update()
print('Acceptence ratio: ', accept_ratio / maxiter * 100, '%')
pb.close()

# samples = np.array(samples)
# for i in range(len(theta[0])):
#     plt.figure()
#     plt.hist(samples[:,i])
#     plt.axvline(theta_ref[0][i], c="red", ls="dashed")


theta = np.mean(samples, axis=0)[np.newaxis,:]
grid_t = T.transform_grid(grid, T.backend.to(theta), method="closed_form")

plt.figure()
plt.plot(grid, grid_ref[0])
plt.plot(grid, grid_t[0])




# %% TRANSFORM DATA

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 100
batch_size = 2

T = difw.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

theta = T.sample_transformation(batch_size)
grid = T.uniform_meshgrid(outsize)
grid_t = T.transform_grid(grid, theta)

T.visualize_deformgrid(theta)

width = 50
channels = 2

# data generation
# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)

data_t = T.transform_data(torch.tensor(data), theta, outsize)

# plot data
batch_size, width, channels = data.shape

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
    ax[i, 0].plot(data[:, :, i].T, color="blue", alpha=0.1)

# plot transformed data per batch
fig, ax = plt.subplots(nrows=channels, ncols=batch_size, sharex=True, squeeze=False)
for i in range(channels):
    for j in range(batch_size):
        ax[i, j].plot(np.linspace(0,1,width), data[j, :, i], color="blue")
        ax[i, j].plot(np.linspace(0,1,outsize), data_t[j, :, i], color="red")

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
    ax[i, 0].plot(np.linspace(0,1,width), data[:, :, i].T, color="blue")
    ax[i, 0].plot(np.linspace(0,1,outsize), data_t[:, :, i].T, color="red")


# %% OPTIMIZATION BY GRADIENT (DATA)

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1
method = "closed_form"

basis = "qr"
basis = "svd"
basis = "sparse"
basis = "rref"

width = 100
channels = 1

# %%
import scipy
from scipy.interpolate import UnivariateSpline
def gradient_spline(time, f, smooth=False):
    """
    This function takes the gradient of f using b-spline smoothing
    :param time: vector of size N describing the sample points
    :param f: numpy array of shape (N) with N samples
    :param smooth: smooth data (default = F)
    :rtype: tuple of numpy ndarray
    :return f0: smoothed functions functions
    :return g: first derivative of each function
    :return g2: second derivative of each function
    """
    
    if smooth:
        spar = len(time) * (.025 * np.fabs(f).max()) ** 2
    else:
        spar = 0
    spline = UnivariateSpline(time, f, s=spar)
    f0 = spline(time)
    g = spline(time, 1)
    g2 = spline(time, 2)

    return f0, g, g2

def f_to_srsf(f, time, smooth=False):
    """
    converts f to a square-root slope function (SRSF)
    :param f: vector of size N samples
    :param time: vector of size N describing the sample points
    :rtype: vector
    :return q: srsf of f
    """
    eps = np.finfo(np.double).eps
    f0, g, g2 = gradient_spline(time, f, smooth)
    q = g / np.sqrt(np.fabs(g) + eps)
    return q

from scipy.integrate import cumtrapz
def srsf_to_f(q, time, f0=0.0):
    """
    converts q (srsf) to a function
    :param q: vector of size N samples of srsf
    :param time: vector of size N describing time sample points
    :param f0: initial value
    :rtype: vector
    :return f: function
    """
    integrand = q*np.fabs(q)
    f = f0 + cumtrapz(integrand, time, initial=0)
    return f


def gradient_custom(f, axis=1, edge_order = 1):
    N = f.ndim 
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(None, -2)
    slice3[axis] = slice(1, -1)
    slice4[axis] = slice(2, None)
    
    backend = np if isinstance(f, np.ndarray) else torch
    g = backend.empty_like(f, dtype=backend.float64)

    g[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / 2.

    if edge_order == 1:
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0

        g[tuple(slice1)] = f[tuple(slice2)] - f[tuple(slice3)]

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        g[tuple(slice1)] = f[tuple(slice2)] - f[tuple(slice3)]

    else:
        slice1[axis] = 0
        slice2[axis] = 0
        slice3[axis] = 1
        slice4[axis] = 2

        a = -1.5
        b = 2. 
        c = -0.5
        g[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        slice1[axis] = -1
        slice2[axis] = -3
        slice3[axis] = -2
        slice4[axis] = -1

        a = 0.5
        b = -2.
        c = 1.5
        g[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

    return g

def curve2srvf(f):
    eps = np.finfo(np.double).eps
    g = np.gradient(f, axis=1, edge_order=2)
    # g = gradient_custom(f, axis=1, edge_order=2)
    q = g / np.sqrt(np.fabs(g) + eps)
    return q

    i = 2*f[:,0,:] - f[:,1,:]
    i = np.expand_dims(i, 1)
    fp = np.diff(f, axis=1, prepend=i)
    tmp =  np.sign(fp) * np.sqrt(np.abs(fp))

    q = fp / np.sqrt(np.fabs(fp) + eps)

    return tmp

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def cumtrapz(y, axis=1, initial=0):
    d = 1
    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))

    backend = np if isinstance(y, np.ndarray) else torch
    cat = np.concatenate if isinstance(y, np.ndarray) else torch.cat
    res = backend.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        shape = list(res.shape)
        shape[axis] = 1
        res = cat([backend.ones(shape) * initial, res], axis)

    return res

def srvf2curve(q, f0=0.0):
    integrand = q*np.abs(q)
    return f0 + cumtrapz(integrand, axis=1, initial=0)
    # return f0 + np.cumsum(integrand, axis=1)

# def curve2srvf(f):
#     i = 2*f[:,0,:] - f[:,1,:]
#     i = torch.unsqueeze(i, 1)
#     fp = torch.diff(f, axis=1, prepend=i)
#     return torch.sign(fp) * torch.sqrt(torch.abs(fp))

# def srvf2curve(q, f0=0.0):
#     integrand = q*torch.abs(q)
#     return f0 + torch.cumsum(integrand, dim=1)

# Generation
batch_size = 1
channels = 1
width = 100
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)
data = torch.tensor(data)
srvf = False
if srvf:
    data = curve2srvf(data)

# newdata = np.empty_like(data)
# olddata = np.empty_like(data)
# for i in range(batch_size):
#     for j in range(channels):
#         newdata[i,:,j] = f_to_srsf(data[i,:,j], x[i,:,j], smooth=False)

#         olddata[i,:,j] = srsf_to_f(newdata[i,:,j], x[i,:,j], f0=data[i,0,j])

newdata2 = curve2srvf(data)
olddata2 = srvf2curve(newdata2, f0=data[:,0,:])

plt.figure()
plt.plot(data[0,:,0])
# plt.plot(olddata[0,:,0])
plt.plot(olddata2[0,:,0])

plt.figure()
# plt.plot(newdata[0,:,0])
plt.plot(newdata2[0,:,0])

# %%
T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

# custom deformation
alpha = 1
n = outsize
np.random.seed(0)
grid_t = np.cumsum(np.random.dirichlet(alpha=[alpha] * n))
grid_t = np.linspace(0,1,n)**3
data_t1 = T.interpolate(data + noise*0, torch.tensor(grid_t), outsize)
# theta_1 = T.identity(batch_size, epsilon=1.0)
# data_t1 = data

# torch.manual_seed(0)
# theta_1 = T.sample_transformation(batch_size)*1
# data_t1 = T.transform_data(data, theta_1, outsize, method)

theta_2 = torch.autograd.Variable(T.sample_transformation(batch_size), requires_grad=True)
# theta_2 = torch.autograd.Variable(T.identity(batch_size), requires_grad=True)
data_t2 = T.transform_data(data, theta_2, outsize, method)

lr = 1e-2
optimizer = torch.optim.Adam([theta_2], lr=lr)
# optimizer = torch.optim.SGD([theta_2], lr=lr, momentum=0.9)

# torch.set_num_threads(1)
loss_values = []
maxiter = 500
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        data_t2 = T.transform_data(data, theta_2, outsize, method)
        loss = torch.norm(data_t2 - data_t1)
        # loss = torch.norm(data_t2 - data_t1, dim=1).mean()
        loss.backward()
        optimizer.step()

        if torch.any(torch.isnan(theta_2)):
            print("AHSDASD")
            break

        loss_values.append(loss.item())
        pb.update()
        pb.set_postfix({'loss': loss.item()})
    pb.close()

plt.figure()
plt.plot(loss_values)
# plt.axhline(color="black", ls="dashed")
plt.yscale('log')

plt.figure()
plt.plot(data_t1[:,:,0].t())
plt.plot(data_t2[:,:,0].detach().t())

plt.figure()
plt.plot(data_t1[:,:,0].t() - data_t2[:,:,0].detach().t())
plt.axhline(color="black", ls="dashed")
# theta_1, theta_2

if srvf:
    plt.figure()
    plt.plot(srvf2curve(data_t1)[:,:,0].t())
    plt.plot(srvf2curve(data_t2)[:,:,0].detach().t())



# %%

x = np.linspace(0,1, 100)
y = np.sin(2*np.pi*x) + np.random.normal(0,0.1, (10, 100))
plt.figure()
for i in range(10):
    y[i,:] = y[i,:] * np.random.uniform(0.5,1)
    plt.plot(y[i,:])

c = np.cov(y, rowvar=False)
u, s, vh = np.linalg.svd(c)

plt.figure()
for i in range(10):
    plt.plot(u[:,i]*s[i])