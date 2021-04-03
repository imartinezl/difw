# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

# %% TEST 

tess_size = 5
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
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


# %% OPTIMIZATION

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 2

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

# theta_1 = T.identity(batch_size, epsilon=0)
theta_1 = T.sample_transformation(batch_size)
grid_t1 = T.transform_grid(grid, theta_1)

theta_2 = T.sample_transformation(batch_size)
theta_2.requires_grad = True

lr = 1e-1
optimizer = torch.optim.Adam([theta_2], lr=lr)

loss_values = []
maxiter = 50
for i in range(maxiter):
    optimizer.zero_grad()
    grid_t2 = T.transform_grid(grid, theta_2, mode="closed_form")
    loss = torch.norm(grid_t2 - grid_t1)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())


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
