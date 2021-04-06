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
grid_t = T.transform_grid(grid, theta)

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, method='numeric')
T.visualize_gradient(theta)
T.visualize_gradient(theta, method="numeric")

