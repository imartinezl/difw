# %%

import importlib
import time
import numpy as np
import cpab

# %%
importlib.reload(cpab)

tess_size = 100
backend = "numpy"
T = cpab.Cpab(tess_size, backend)

np.random.seed(1)

batch_size = 40
# theta = cpab.sample_transformation(batch_size, np.random.normal(-1, 1, (99)), np.random.normal(1, 1, (99,99)))
theta = T.sample_transformation(batch_size)
# theta = np.random.uniform(-3, 3, (batch_size, 99))

outsize = 200
grid = T.uniform_meshgrid(outsize)

timing = []
for i in range(50):
    start = time.time()
    grid_t = T.transform_grid(grid, theta)
    stop = time.time()
    timing.append(stop - start)
    break

print("Time: ", np.mean(timing), "+-", np.std(timing))
print("Grid: ", grid.shape, "->", grid_t.shape)

# plt.figure()
# plt.hist(timing, bins=20)

T.visualize_tesselation()
T.visualize_velocity(theta, n_points=50)
T.visualize_deformgrid(theta, n_points=100)

print("Done")
