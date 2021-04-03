# %%

import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

import torch.autograd.profiler as profiler
import torch.utils.benchmark as benchmark

# %% SETUP
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

# %%

repetitions = 1
n = 1
timing = timeit.Timer(
    lambda: T.transform_grid(grid, theta),
    # setup="gc.enable()"
).repeat(repetitions, n)
print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))

# %% PYTORCH PROFILER

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

    repetitions = 1

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


# %% snakeviz
# %prun -D program.prof T.transform_grid(grid, theta)