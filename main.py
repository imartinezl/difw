# %%

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
outsize = 10
batch_size = 2

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
theta = T.sample_transformation_with_prior(batch_size)
grid_t = T.transform_grid(grid, theta)

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, method='numeric')
T.visualize_gradient(theta)
T.visualize_gradient(theta, method="numeric")

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

