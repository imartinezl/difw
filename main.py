# %%

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

# import importlib
# importlib.reload(cpab)
import torch.autograd.profiler as profiler
import torch.utils.benchmark as benchmark

# %%

tess_size = 5
backend = "numpy"
# backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="cpu")
T.params.use_slow = True
outsize = 10
grid = T.uniform_meshgrid(outsize)

batch_size = 2
# seed = np.random.randint(100)
# torch.manual_seed(0)
theta = T.sample_transformation(batch_size)
theta = T.identity(batch_size, epsilon=1.0)
grid_t = T.transform_grid(grid, theta)

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, mode='numeric')
T.visualize_gradient(theta)
T.visualize_gradient(theta, mode="numeric")

# %% TEST GPU
tess_size = 5
xmin = 0.0
xmax = 1.0
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="gpu")
T.params.use_slow = True

outsize = 100
grid = T.uniform_meshgrid(outsize)

batch_size = 1
torch.manual_seed(10)
theta = T.sample_transformation(batch_size)
theta = T.identity(batch_size, epsilon=1.)

# c = T.backend.cpab_gpu.get_cell(grid, xmin, xmax, tess_size)
# v = T.backend.cpab_gpu.get_velocity(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size)
# grid_t = T.backend.cpab_gpu.integrate_numeric(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size, 10, 10)
# grid_t = T.backend.cpab_gpu.integrate_closed_form(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size)
grad = T.backend.cpab_gpu.derivative_closed_form(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size)
# grad = T.gradient_grid(grid, theta)
# grid_t = T.transform_grid(grid, theta, mode="closed_form")

# plt.plot(c.cpu().T)
# plt.plot(v.cpu().T)
# plt.axline((0,0),(1,1), c="black")
# plt.plot(grid.cpu(), grid_t.cpu().T)

for k in range(theta.shape[1]):
    plt.plot(grid.cpu(), grad.cpu()[:, :, k].T)

# repetitions = 20
# n = 1
# timing = timeit.Timer(
#     # lambda: T.backend.cpab_cpu.get_velocity(grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc)
#     # lambda: T.transform_grid(grid, theta)
#     # lambda: T.backend.cpab_gpu.integrate_numeric(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size, 10, 10)
#     lambda: T.backend.cpab_gpu.integrate_closed_form(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size)
#     # setup="gc.enable()"
# ).repeat(repetitions, n)
# print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))


# %% TEST CPU
tess_size = 5
xmin = 0.0
xmax = 1.0
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="cpu")

outsize = 100
grid = T.uniform_meshgrid(outsize)

batch_size = 1

theta = T.sample_transformation(batch_size)
theta = T.identity(batch_size, epsilon=1)

# c = T.backend.cpab_cpu.get_cell(grid, xmin, xmax, tess_size)
# v = T.backend.cpab_cpu.get_velocity(grid, theta, T.params.B.contiguous(), xmin, xmax, tess_size)
# grid_t = T.backend.cpab_cpu.integrate_numeric(grid.contiguous(), theta.contiguous(), T.params.B.contiguous(), xmin, xmax, tess_size, 10, 10)
# grid_t = T.backend.cpab_cpu.integrate_closed_form(grid.contiguous(), theta.contiguous(), T.params.B.contiguous(), xmin, xmax, tess_size)


# TODO: DONE what does B.contiguous() do?? it messes everything up

# T.params.use_slow = True
grad = T.gradient_grid(grid, theta)
for k in range(theta.shape[1]):
    plt.plot(grid, grad[:, :, k].T)

repetitions = 1
n = 1
timing = timeit.Timer(
    # lambda: T.backend.cpab_cpu.get_velocity(grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc)
    lambda: T.transform_grid(grid, theta)
    # setup="gc.enable()"
).repeat(repetitions, n)
print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))

# %%

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    T.transform_grid(grid, theta)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
# prof.export_chrome_trace("trace.json")

# %%
t0 = benchmark.Timer(
    stmt="T.transform_grid(grid, theta)",
    globals={"T":T, "grid":grid, "theta": theta}
)
t0.timeit(1)

# %%
from itertools import product
import pickle
results = []

# tess_size_array = [3,5,10,20,50,100]
# outsize_array = [10,20,50,100,200]
# batch_size_array = [1,2,5,10,20,100]
# num_threads_array = [1,2,4]
tess_size_array = [100]
outsize_array = [200]
batch_size_array = [100]
num_threads_array = [1]

for tess_size, outsize, batch_size in product(tess_size_array, outsize_array, batch_size_array):
    backend = "pytorch"
    T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="cpu")
    grid = T.uniform_meshgrid(outsize)
    theta = T.identity(batch_size, epsilon=1)

    T.transform_grid(grid, theta)

    label = "Transform Grid"
    sub_label = f'[{tess_size}, {outsize}, {batch_size}]'

    print(sub_label)

    repetitions = 5

    for num_threads in num_threads_array:


        T.params.use_slow = False
        t0 = benchmark.Timer(
            stmt="T.transform_grid(grid, theta, mode='closed_form')",
            globals={"T": T, "grid": grid, "theta": theta},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="CPU / FAST / CLOSED FORM"
        )
        results.append(t0.timeit(repetitions))


        T.params.use_slow = True
        t1 = benchmark.Timer(
            stmt="T.transform_grid(grid, theta, mode='closed_form')",
            globals={"T": T, "grid": grid, "theta": theta},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="CPU / SLOW / CLOSED FORM"
        )
        results.append(t1.timeit(repetitions))

        T.params.use_slow = False
        t2 = benchmark.Timer(
            stmt="T.transform_grid(grid, theta, mode='numeric')",
            globals={"T": T, "grid": grid, "theta": theta},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="CPU / FAST / NUMERIC"
        )
        results.append(t2.timeit(repetitions))

        T.params.use_slow = True
        t3 = benchmark.Timer(
            stmt="T.transform_grid(grid, theta, mode='numeric')",
            globals={"T": T, "grid": grid, "theta": theta},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="CPU / SLOW / NUMERIC"
        )
        results.append(t3.timeit(repetitions))
    


# %%
compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()


# %%
%prun -D program.prof T.transform_grid(grid, theta)

# %% TO REMOVE
import torch
tess_size = 5
xmin = 0.0
xmax = 1.0
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="gpu")
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
import torch
import matplotlib.pyplot as plt

tess_size = 5
backend = "numpy"
backend = "pytorch"
T = cpab.Cpab(tess_size, backend, zero_boundary=True, device="gpu")
outsize = 10
grid = T.uniform_meshgrid(outsize)

batch_size = 1
# seed = np.random.randint(100)
# np.random.seed(0)
# torch.manual_seed(0)
theta = T.sample_transformation(batch_size)
# theta = np.ones((batch_size, tess_size - 1))
# theta = torch.ones((batch_size, tess_size - 1))
# theta = np.array([[ 0.8651,  0.0284,  0.5256, -0.3633, -0.4169, -1.2650]])
grid_t = T.transform_grid(grid, theta)
# derivative = grid_t
# derivative

# plt.plot(grid_t.numpy().T)
# plt.plot(grid_t.T)
T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, mode='numeric')
T.visualize_gradient(theta)
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
