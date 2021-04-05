# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab
from tqdm import tqdm 

# %% TEST 

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = True
outsize = 10
batch_size = 2

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, theta)

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, mode='numeric')
T.visualize_gradient(theta)
T.visualize_gradient(theta, mode="numeric")


# %% OPTIMIZATION BY GRADIENT

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 20

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

theta_1 = T.sample_transformation(batch_size)
grid_t1 = T.transform_grid(grid, theta_1)

theta_2 = torch.autograd.Variable(T.sample_transformation(batch_size), requires_grad=True)

lr = 1e-1
optimizer = torch.optim.Adam([theta_2], lr=lr)

# torch.set_num_threads(1)
loss_values = []
maxiter = 500
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        grid_t2 = T.transform_grid(grid, theta_2, mode="closed_form")
        # grid_t2 = T.gradient_grid(grid, theta_2, mode="numeric")
        loss = torch.norm(grid_t2 - grid_t1)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        pb.update()
        pb.set_postfix({'loss': loss.item()})
    pb.close()

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
# theta_1, theta_2

# %% OPTIMIZATION BY MCMC SAMPLING

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
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
    grid_t = T.transform_grid(grid, theta, mode="closed_form")

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
grid_t = T.transform_grid(grid, T.backend.to(theta), mode="closed_form")

plt.figure()
plt.plot(grid, grid_ref[0])
plt.plot(grid, grid_t[0])




# %% TRANSFORM DATA

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 2

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

theta = T.sample_transformation(batch_size)
grid = T.uniform_meshgrid(outsize)
grid_t = T.transform_grid(grid, theta)

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

# TODO: define data visualize functions on cpab
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
