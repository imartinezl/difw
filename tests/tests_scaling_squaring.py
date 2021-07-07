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


tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 1000
batch_size = 1
basis = "svd"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

torch.manual_seed(0)
theta_1 = T.identity(batch_size, epsilon=2.0)
theta_1 = T.sample_transformation(batch_size)
grid_t1 = T.transform_grid(grid, theta_1)

theta_2 = torch.autograd.Variable(T.identity(batch_size, epsilon=0.0), requires_grad=True)

lr = 1e-3
optimizer = torch.optim.Adam([theta_2], lr=lr)
# optimizer = torch.optim.SGD([theta_2], lr=lr)

# torch.set_num_threads(1)
loss_values = []
maxiter = 1
start_time = time.process_time_ns()
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        N = 2
        t = 1.0
        # grid_t2 = T.transform_grid_ss(grid, theta_2, method="closed_form", time=1.0, N=N)
        
        # t = t / 2**N
        grid_t2 = T.transform_grid(grid, theta_2, method="closed_form")
        for i in range(N):
            grid_t2 = T.backend.interpolate_grid(grid_t2)
        
        loss = torch.norm(grid_t2 - grid_t1, dim=1).mean()
        loss.backward()
        optimizer.step()

        if torch.any(torch.isnan(theta_2)):
            print("AHSDASD")
            break

        L = loss.item()
        loss_values.append(L)
        pb.update()
        pb.set_postfix({'loss': L})
        
    pb.close()
stop_time = time.process_time_ns()

plt.figure()
plt.plot(grid_t1.T)
plt.plot(grid_t2.detach().numpy().T)

plt.figure()
plt.plot(loss_values)

(stop_time-start_time) * 1e-6

# %%

plt.plot(grid_t1.T)
plt.plot(T.transform_grid(grid, 2**N * theta_2.detach()).T)


# %%
%%timeit -r 20 -n 1

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 1000
batch_size = 1
basis = "rref"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

torch.manual_seed(0)
theta_1 = T.identity(batch_size, epsilon=1.0)
# theta_1 = T.sample_transformation(batch_size)
grid_t1 = T.transform_grid(grid, theta_1)

theta_2 = torch.autograd.Variable(theta_1, requires_grad=True)

N = 0
t = 1.0
# grid_t2 = T.transform_grid_ss(grid, theta_2, method="closed_form", time=1.0, N=N)       
# t = t / 2**N
grid_t2 = T.transform_grid(grid, theta_2 / 2**N, method="closed_form")
for i in range(N):
    grid_t2 = T.backend.interpolate_grid(grid_t2)

loss = torch.norm(grid_t2 - grid_t1, dim=1).mean()
loss.backward()

# %%
plt.plot(grid_t1.T)
plt.plot(grid_t2.detach().numpy().T)

# %%