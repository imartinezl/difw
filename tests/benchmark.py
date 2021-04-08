# %%

import sys

sys.path.insert(0, "..")

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

import torch.autograd.profiler as profiler
import torch.utils.benchmark as benchmark

# %% SETUP
tess_size = 50
backend = "pytorch"  # ["pytorch", "numpy"]
device = "gpu"  # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 20
method = "closed_form"

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
theta = T.identity(batch_size, epsilon=1.0)

# T.params.nSteps1 = 5
# T.params.nSteps2 = 5
grid_t = T.transform_grid(grid, theta, method)

# plt.plot(grid_t.cpu().T)
print(1)

# %% PYTORCH BENCHMARK
t0 = benchmark.Timer(
    stmt="""
    theta_grad = torch.autograd.Variable(theta, requires_grad=True)
    grid_t = T.transform_grid(grid, theta_grad, method)
    loss = torch.norm(grid_t)
    loss.backward() 
    """, 
    globals={"T": T, "grid": grid, "theta": theta, "method": method}
)
# t0.timeit(1)
t0.blocked_autorange(min_run_time=0.5)
# %% CPROFILE

import cProfile

cProfile.run(
    """
theta_grad = torch.autograd.Variable(theta, requires_grad=True)
for i in range(1000): 
    grid_t = T.transform_grid(grid, theta_grad, method)
    # loss = torch.norm(grid_t)
    # loss.backward()
""",
    sort="cumtime",
)
# %% YEP + PPROF
import yep

# torch.set_num_threads(1)

theta_grad = torch.autograd.Variable(theta, requires_grad=True)
yep.start("profile.prof")
for i in range(100):
    grid_t = T.transform_grid(grid, theta_grad, method)
    # loss = torch.norm(grid_t)
    # loss.backward()

yep.stop()

# %% TIMEIT

repetitions = 1000
n = 10
timing = timeit.Timer(
    lambda: T.transform_grid(grid, theta),
    # setup="gc.enable()"
).repeat(repetitions, n)
print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))

# %% PYTORCH PROFILER

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    T.transform_grid(grid, theta, method)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
# prof.export_chrome_trace("trace.json")

# %% snakeviz
# %prun -D program.prof T.transform_grid(grid, theta)

# %%

from itertools import product

results = []

num_threads_arr = [1] # [1, 2, 4]

backend_arr = ["pytorch"] # ["pytorch", "numpy"]
device_arr = ["cpu", "gpu"] # ["cpu", "gpu"]
method_arr = ["closed_form"] # ["closed_form", "numeric"]
use_slow_arr = [False] # [True, False]
zero_boundary_arr = [True] # [True, False]

tess_size_arr = [50]
outsize_arr = [1000]
batch_size_arr = [200]

for (
    backend,
    device,
    method,
    use_slow,
    zero_boundary,
    tess_size,
    outsize,
    batch_size,
) in product(
    backend_arr,
    device_arr,
    method_arr,
    use_slow_arr,
    zero_boundary_arr,
    tess_size_arr,
    outsize_arr,
    batch_size_arr,
):
    # SETUP
    T = cpab.Cpab(tess_size, backend, device, zero_boundary)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)
    theta = T.identity(batch_size, epsilon=1)

    label = "CPAB: backend, device, method, use_slow, zero_boundary, tess_size, outsize, batch_size"
    # sub_label = f"[{backend}, {device}, {method}, {'slow' if use_slow else 'fast'}, {'zero_boundary' if zero_boundary else 'no_zero_boundary'}, {tess_size}, {outsize}, {batch_size}]"
    sub_label = f"[{backend}, {device}, {method}, {use_slow}, {zero_boundary}, {tess_size}, {outsize}, {batch_size}]"
    print(sub_label)
    for num_threads in num_threads_arr:
        repetitions = 1

        # FORWARD
        t0 = benchmark.Timer(
            stmt=
            """
            grid_t = T.transform_grid(grid, theta, method)
            """,
            globals={"T": T, "grid": grid, "theta": theta, "method": method},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Forward",
        )
        # results.append(t0.timeit(repetitions))
        results.append(t0.blocked_autorange(min_run_time=0.5))
        # results.append(t0.adaptive_autorange())

        # BACKWARD
        t1 = benchmark.Timer(
            stmt=
            """
            theta_grad = torch.autograd.Variable(theta, requires_grad=True)
            grid_t = T.transform_grid(grid, theta_grad, method)
            loss = torch.norm(grid_t)
            loss.backward()            
            """,
            globals={"T": T, "grid": grid, "theta": theta, "method": method},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Backward",
        )
        # results.append(t1.timeit(repetitions))
        results.append(t1.blocked_autorange(min_run_time=0.5))
        # results.append(t1.adaptive_autorange())


# %%
compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

# %% RESULTS TO LATEX
import pandas as pd

df = [
    pd.DataFrame({
        'experiment': t.as_row_name.replace('[', '').replace(']', ''), 
        'description': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        'time': t.raw_times,
        'time_mean': np.mean(t.raw_times),
        'time_std': np.std(t.raw_times),
        })
    for t in results
]
df = pd.concat(df, ignore_index=True)


header = ['Backend', 'Device', 'Method', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
parameters = pd.DataFrame(df["experiment"].str.split(',', expand=True).values, columns=header)

a = pd.concat([parameters, df], axis=1).drop(columns=['experiment'])
a.to_latex(index=False, escape=False)


# %% RESULTS TO PLOT
import seaborn as sns
import pandas as pd

df = [
    pd.DataFrame({
        'experiment': t.as_row_name, 
        'description': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        'time': t.raw_times})
    for t in results
]
df = pd.concat(df, ignore_index=True)
df['experiment_id'] = df.groupby('experiment', sort=False).ngroup().apply(str)

n = pd.unique(df.experiment_id)
exps = pd.unique(df.experiment)
caption = '\n'.join([k + ": " + exps[int(k)] for k in n])

header = ['Backend', 'Device', 'Method', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
cell_text = [e.replace('[','').replace(']','').split(', ') for e in exps]

vlen = np.vectorize(len)
w = np.max(vlen(cell_text + [header]), axis=0)

# %%
import matplotlib


with sns.axes_style("whitegrid"):
    g = sns.catplot(
        x="time", y="experiment_id", 
        hue="threads", col="description",
        data=df, kind="box", ci=None, sharex=True,
        fliersize=2, linewidth=1, width=0.75)
    sns.despine(top=False, right=False, left=False, bottom=False)
    plt.xticks(np.logspace(-10,-1, num=10))
    # plt.figtext(0, -0.1, caption, wrap=True, 
    #     verticalalignment='top', horizontalalignment='left', fontsize=10)

    table = plt.table(
        cellText=cell_text, 
        rowLabels=n, 
        colLabels=header, 
        colWidths = w,
        cellLoc='center',
        loc='bottom',
        # fontsize=50
        bbox=[-1.0,-0.5, 1.2, 0.35]
        )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # table.auto_set_column_width(n)
    # table.scale(1, 1)

    for ax in g.axes[0]:
        ax.set_xscale('log')
        ax.grid(axis="x", which="minor", ls="--", c='gray', alpha=0.2)

plt.savefig('example.png')

# %%
