# %%

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

# import importlib
# importlib.reload(cpab)

# %%
import torch
tess_size = 5
xmin = 0.0
xmax = 1.0
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True)
B = torch.tensor(T.params.B, dtype=torch.float32)

batch_size = 2
seed = np.random.randint(100)
torch.manual_seed(seed)
theta = T.sample_transformation(batch_size)
# theta = T.identity(batch_size, 0.1)
theta = torch.ones(batch_size, tess_size - 1)


outsize = 100
grid = T.uniform_meshgrid(outsize)
grid_t = T.backend.cpab_cpu.integrate_closed_form(grid, theta, B, xmin, xmax, tess_size)
grid_tp = T.backend.cpab_cpu.integrate_numeric(
    grid, theta, B, xmin, xmax, tess_size, 10, 100
)

plt.figure()
plt.title("Deformed grid")
plt.plot(grid, grid_t.T, color="blue", alpha=0.1)
plt.plot(grid, grid_tp.T, color="red", alpha=1)
plt.axline([0, 0], [1, 1], color="black", ls="dashed")

repetitions = 500
timing = timeit.Timer(
    lambda:  T.backend.cpab_cpu.integrate_closed_form(grid, theta, B, 0, 1, tess_size),
    # lambda:  T.backend.cpab_cpu.integrate_numerical(grid, theta, B, 0, 1, tess_size, 10, 100),
    # setup="gc.enable()"
).repeat(repetitions, 1)
print("Time: ", np.mean(timing), "+-", np.std(timing))

# %%
derivative = T.backend.cpab_cpu.derivative_closed_form(grid, theta, B, xmin, xmax, tess_size)

b = 0
plt.figure()
plt.title("Closed-form derivative")
for k in range(theta.shape[1]):
    plt.plot(grid, derivative[b, :, k])

h = 1e-3
plt.figure()
plt.title("Numerical derivative")
k = 0
for k in range(theta.shape[1]):
    grid_t1 = T.backend.cpab_cpu.integrate_closed_form(
        grid, theta, B, xmin, xmax, tess_size
    )
    grid_t1 = T.backend.cpab_cpu.integrate_numeric(
        grid, theta, B, xmin, xmax, tess_size, 10, 100
    )
    theta_h = theta.clone()
    theta_h[b][k] += h
    grid_t2 = T.backend.cpab_cpu.integrate_closed_form(
        grid, theta_h, B, xmin, xmax, tess_size
    )
    grid_t2 = T.backend.cpab_cpu.integrate_numeric(
        grid, theta_h, B, xmin, xmax, tess_size, 10, 100
    )

    dnum = (grid_t2 - grid_t1) / h

    plt.plot(grid, dnum[b])





# %%

import cpab
import timeit
import numpy as np
import matplotlib.pyplot as plt

tess_size = 5
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True)
outsize = 100
grid = T.uniform_meshgrid(outsize)

batch_size = 2
# seed = np.random.randint(100)
# np.random.seed(seed)
theta = T.sample_transformation(batch_size)
# theta = np.ones((batch_size, tess_size - 1))
# theta = np.array([[ 0.8651,  0.0284,  0.5256, -0.3633, -0.4169, -1.2650]])
grid_t = T.transform_grid(grid, theta)
# derivative = grid_t
# derivative

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, mode='numeric')
# T.visualize_gradient(theta)
T.visualize_gradient(theta, mode="numeric")

# b = 0
# for k in range(derivative.shape[2]):
#     plt.plot(derivative[b,:, k])

# plt.plot(grid, grid_t.T, color="blue", alpha=1)
# plt.axline([0, 0], [1, 1], color="black", ls="dashed")

# repetitions = 10
# n = 1
# timing = timeit.Timer(
#     lambda: T.transform_grid(grid, theta),
#     # setup="gc.enable()"
# ).repeat(repetitions, n)
# print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))
# %%
import torch

tess_size = 50
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True)

outsize = 100
batch_size = 2
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
    grid_t2 = T.transform_grid(grid, theta_2, mode="numeric")
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



# %%
np.random.seed(1)
torch.manual_seed(1)

batch_size = 4
# theta = cpab.sample_transformation(batch_size, np.random.normal(-1, 1, (99)), np.random.normal(1, 1, (99,99)))
theta = T.sample_transformation(batch_size)
# theta = np.random.uniform(-3, 3, (batch_size, tess_size-1))

outsize = 100
grid = T.uniform_meshgrid(outsize)
grid_t = T.transform_grid(grid, theta)
# T.test(grid, theta)

width = 50
channels = 2

# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)

data_t = T.transform_data(torch.tensor(data), theta, outsize)

# TODO: define visualize functions on cpab
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

# %%
import timeit

timing = []
reps = 50
for i in range(reps):
    start = time.time()
    grid_t = T.transform_grid(grid, theta)
    stop = time.time()
    timing.append(stop - start)
    break

timing2 = timeit.timeit(lambda: T.transform_grid(grid, theta), number=reps)
print(timing2 / reps)
print("Time: ", np.mean(timing), "+-", np.std(timing))
print("Grid: ", grid.shape, "->", grid_t.shape)

# plt.figure()
# plt.hist(timing, bins=20)

# %%
T.visualize_tesselation()
T.visualize_velocity(theta, n_points=50)
T.visualize_deformgrid(theta, n_points=100)
print("Done")

# %%
