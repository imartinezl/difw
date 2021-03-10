# %%

import time
import numpy as np
import matplotlib.pyplot as plt
import cpab
# import importlib
# importlib.reload(cpab)

# %%
tess_size = 100
backend = "numpy"
T = cpab.Cpab(tess_size, backend)

np.random.seed(1)

batch_size = 4
# theta = cpab.sample_transformation(batch_size, np.random.normal(-1, 1, (99)), np.random.normal(1, 1, (99,99)))
theta = T.sample_transformation(batch_size)
# theta = np.random.uniform(-3, 3, (batch_size, 99))

outsize = 120
grid = T.uniform_meshgrid(outsize)

width = 100
channels = 2

# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels))*2*np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)

data_t = T.transform_data(data, theta, outsize)
# %%

# TODO: define visualize functions on cpab
# plot data
batch_size, width, channels = data.shape

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
    ax[i,0].plot(data[:,:,i].T, color="blue", alpha=0.1)

# plot transformed data per batch
fig, ax = plt.subplots(nrows=channels, ncols=batch_size, sharex=True, squeeze=False)
for i in range(channels):
    for j in range(batch_size):
        ax[i,j].plot(data[j,:,i], color="blue")
        ax[i,j].plot(data_t[j,:,i], color="red")

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
        ax[i,0].plot(data[:,:,i].T, color="blue")
        ax[i,0].plot(data_t[:,:,i].T, color="red")

# %%
import timeit
timing = []
reps = 50
for i in range(reps):
    start = time.time()
    grid_t = T.transform_grid(grid, theta)
    stop = time.time()
    timing.append(stop - start)
    # break

timing2 = timeit.timeit(lambda: T.transform_grid(grid, theta), number=reps)
print(timing2/reps)
print("Time: ", np.mean(timing), "+-", np.std(timing))
print("Grid: ", grid.shape, "->", grid_t.shape)

# plt.figure()
# plt.hist(timing, bins=20)

T.visualize_tesselation()
T.visualize_velocity(theta, n_points=50)
T.visualize_deformgrid(theta, n_points=100)
T.visualize_velocity2deform(theta, n_points=100)
print("Done")

# %%
