# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

# %%

tess_size = 5
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1
basis = "svd"
basis = "sparse"
basis = "rref"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
theta = T.sample_transformation_with_prior(batch_size)
theta = T.identity(batch_size, epsilon=0)
grid_t = T.transform_grid(grid, theta)

# T.visualize_tesselation()
# T.visualize_velocity(theta)
# T.visualize_deformgrid(theta)
# T.visualize_deformgrid(theta, method='numeric')
T.visualize_gradient(theta)
# T.visualize_gradient(theta, method="numeric")


# %% Data Transform

width = 50
channels = 3

# Generation
# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)
T.visualize_deformdata(data, theta)


# %%

tess_size = 3
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 100
batch_size = 1

basis = "svd"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_svd = T.params.B

basis = "rref"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_rref = T.params.B

basis = "sparse"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_sparse = T.params.B

# %%
from scipy.linalg import orth

B_svd, orth(B_svd), B_rref, orth(B_rref), B_sparse, orth(B_sparse)
# %%

# 
def metric1(B):
    data = []
    for i in np.arange(0, B.shape[0], 2):
        for j in np.arange(0, B.shape[0], 2):
            d = np.sum(B[i, ] * B[j, ])
            data.append([i, j, d])

    return data

def metric2(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric3(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric4(B):
    data = []
    data = np.empty((B.shape[1], B.shape[1]))
    for i in np.arange(0, B.shape[1]):
        for j in np.arange(0, B.shape[1]):
            d = np.dot(B[:, i], B[:, j])
            # data.append([i, j, d])
            data[i,j] = d

    return np.round(data,2)

metric = metric4
metric(B_svd), metric(B_rref), metric(B_sparse)

# %%

plt.figure()
plt.spy(metric(B_svd), precision=1e-7)

plt.figure()
plt.spy(metric(B_rref), precision=1e-7)

plt.figure()
plt.spy(metric(B_sparse), precision=1e-7)

# %%

plt.figure()
plt.spy(B_svd)

plt.figure()
plt.spy(B_rref)

plt.figure()
plt.spy(B_sparse)

# %%

x = 0
y = 0
for B in [B_svd, B_rref, B_sparse]:
    B = orth(B)
    plt.figure()
    for j in range(B.shape[1]):
        dx = B[0, j]
        dy = B[2, j]
        plt.arrow(x, y, dx, dy, head_width=0.04)

    plt.axis("equal")
    plt.scatter(0,0, color="orange")

# %%
