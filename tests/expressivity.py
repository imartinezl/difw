# %% IMPORT LIBRARIES

import sys

sys.path.insert(0, "..")

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab
import pandas as pd
import seaborn as sns

# %% BSPLINE

t = np.linspace(0, 1, 10)
c = np.random.normal(0, 1, 10)
f = BSpline(t, c, 3)

x = np.linspace(0, 1, 100)
y = f(x)

plt.figure()
plt.plot(t, c, c="red")
plt.plot(x, y, c="blue")

plt.figure()
plt.plot(x, f.derivative()(x))


# %% DELANNOY 


def delannoy(m, n):
    D = np.empty((m, n))

    D[0, :] = 1
    D[:, 0] = 1
    for i in range(1, m):
        for j in range(1, n):
            D[i, j] = D[i - 1, j] + D[i, j - 1] + D[i - 1, j - 1]
    return D

m = n = 100
# m = 2
# n = 3
D = delannoy(m, n)
D.shape

Dmn = D[-1,-1]
path = D*np.flip(D)
path_norm = path / Dmn


plt.imshow(np.log(path_norm), origin="lower")
plt.colorbar()
plt.savefig("delannoy.png", dpi=300)


# %% CURVES / FUNCTIONS

from scipy.interpolate import interp1d, BSpline

def PiecewiseLinear(x, xp, yp):
    f = interp1d(xp, yp, kind="linear")
    return f(x)


def PiecewiseNearest(x, xp, yp):
    f = interp1d(xp, yp, kind="nearest")
    return f(x)

def PiecewiseZero(x, xp, yp):
    f = interp1d(xp, yp, kind="zero")
    return f(x)

def PiecewiseQuadratic(x, xp, yp):
    f = interp1d(xp, yp, kind="quadratic")
    return f(x)

def PiecewiseCubic(x, xp, yp):
    f = interp1d(xp, yp, kind="cubic")
    return f(x)

def PiecewiseSlinear(x, xp, yp):
    f = interp1d(xp, yp, kind="slinear")
    return f(x)

def PiecewiseNext(x, xp, yp):
    f = interp1d(xp, yp, kind="next")
    return f(x)

def Ramsay(x, p):
    K = p // 2
    v = 1 / np.sqrt(K)

    phi = np.random.uniform(-v, v, size=K)
    psi = np.random.uniform(-v, v, size=K)

    phi = np.random.normal(scale=1, size=K)
    psi = np.random.normal(scale=1, size=K)

    k = np.arange(K)
    w = np.zeros_like(x)
    for k in range(K):
        w += phi[k] * np.sin(2*np.pi*k*x) + psi[k] * np.cos(2*np.pi*k*x)
    
    h = np.cumsum(np.exp(w))
    Z = np.max(h)
    return h / Z




def CPA(grid, T):
    # T = cpab.Cpab(tess_size=3)
    theta = T.sample_transformation(1)
    # d = T.params.d
    # theta = np.random.laplace(loc=0, scale=1, size=d)[np.newaxis, :]
    # theta = T.sample_transformation_with_prior(1, length_scale=1e-3, output_variance=0.5)
    # theta = theta * np.linalg.norm(theta)
    # grid = T.uniform_meshgrid(100)
    return T.transform_grid(grid, theta)[0]

def sampler(p, method="uniform"):
    if method == "uniform":
        yp = np.random.uniform(0, 1/p, p)
        yp = np.cumsum(yp / np.sum(yp))
        yp = np.insert(yp, 0, 0)

    if method == "uniform2":
        yp = np.random.uniform(0, 1, p-1)
        yp = np.append(yp, 1)
        yp = np.append(yp, 0)
        yp = np.sort(yp)


    if method == "dirichlet":
        # random concentration
        # alpha = np.random.uniform(1e-3, 20, p)

        # constant concentration
        # e = np.random.randint(-1, 2)
        # b = np.random.randint(1,10)
        # alpha = np.repeat(b**e, p)
        # alpha = np.repeat(1/p, p)
        alpha = np.repeat(1, p)
        yp = np.random.dirichlet(alpha)
        yp = np.cumsum(yp)
        yp = np.insert(yp, 0, 0)
        

    if method == "reject":
        done = False
        reject = 0
        while not done:
            yp = np.random.uniform(0, 1, p-1)
            yp = np.insert(yp, 0, 0)
            yp = np.append(yp, 1)

            jump = yp - np.r_[0, yp[:-1]]
            if any(jump < 0):
                reject += 1
                continue
            done = True

    if method == "exponential":
        u = np.random.uniform(0, 1, p)
        e = -np.log(u)
        x = e / np.sum(e)
        yp = np.cumsum(x)
        # yp = np.append(yp, 1)
        yp = np.append(yp, 0)
        yp = np.sort(yp)

    return yp
            

def generate_path(x, p, function, method="uniform", xp=None, T=None):

    if "Piecewise" in function:      
        yp = sampler(p, method)
        
    if function == "PiecewiseLinear":
        y = PiecewiseLinear(x, xp, yp)
    elif function == "PiecewiseNearest":
        y = PiecewiseNearest(x, xp, yp)
    elif function == "PiecewiseZero":
        y = PiecewiseZero(x, xp, yp)
    elif function == "PiecewiseQuadratic":
        y = PiecewiseQuadratic(x, xp, yp)
    elif function == "PiecewiseCubic":
        y = PiecewiseCubic(x, xp, yp)
    elif function == "PiecewiseSlinear":
        y = PiecewiseSlinear(x, xp, yp)
    elif function == "PiecewiseNext":
        y = PiecewiseNext(x, xp, yp)
    elif function == "CPA":
        y = CPA(x, T)
    elif function == "Ramsay":
        y = Ramsay(x, p)

    return y

# %% COUNT PATHS
def count_paths(p, n, function="PiecewiseLinear", samples=2000, plot=False):
    print(p, n, function, samples)
    
    data = np.zeros((n,n))
    x = np.linspace(0, 1, n)

    T = None
    if function == "CPA":
        T = cpab.Cpab(tess_size=p, basis="svd")

    xp = None
    method = "exponential"
    if "Piecewise" in function:
        xp = np.linspace(0, 1, p+1)

    for i in range(samples):  
        y = generate_path(x, p, function, method, xp, T)

        px = (x*n).astype(int)
        py = (y*n).astype(int)

        px = np.clip(px, 0, n-1)
        py = np.clip(py, 0, n-1)
        # if any(py < 0) or any(py > n-1):
        #     reject += 1
        #     continue
        data[py, px] += 1

    if plot:
        # plt.plot(xp, yp)
        # plt.plot(x,y) 
        cmap = "plasma"
        fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(11, 7))
        
        d0 = np.log((data+1)/np.max(data))
        # d = -np.log(-d)
        im0 = ax[0].imshow(d0, origin="lower", cmap=cmap)
        ax[0].set_title("Path Count (log)")
        ax[0].set_xticks(np.arange(0, n + 1, n // 5))
        ax[0].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im0, ax=ax[0], location="bottom", 
            # ticks=[-6, -4, -2, 0]
            )

        d1 = np.log(path_norm)
        # d = -np.log(-d)
        im1 = ax[1].imshow(d1, origin="lower", cmap=cmap, vmin=np.min(d0))
        ax[1].set_title("Delannoy Count (log)")
        ax[1].set_xticks(np.arange(0, n + 1, n // 5))
        ax[1].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im1, ax=ax[1], location="bottom", 
            # ticks=[-160, -120, -80, -40, 0]
            )
        
        import matplotlib.colors as colors
        d2 = data/samples - path_norm
        im2 = ax[2].imshow(d2, origin="lower", norm=colors.CenteredNorm(), cmap="seismic")
        ax[2].set_title("Path - Delannoy")
        ax[2].set_xticks(np.arange(0, n + 1, n // 5))
        ax[2].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im2, ax=ax[2], location="bottom")

    # return np.sum(data == 0)
    # option 1
    
    # error_mean = np.sum(path_norm - data/samples)
    error_abs = path_norm - (data/samples)

    # error_abs = np.clip(error_abs, 0, None)
    # error_abs = np.abs(error_abs)
    # error_rel = error_abs / path_norm
    error_rel = error_abs

    error_norm = np.linalg.norm(error_rel)
    # error_norm = np.mean(error_rel)

    # option 2

    # visited = data != 0
    # visited_count = np.sum(visited)

    # error_abs = path_norm[visited] - (data[visited]/samples)
    # error_abs = np.clip(error_abs, 0, None)
    # error_norm = np.linalg.norm(error_abs / path_norm[visited])

    return error_norm

n = 100
p = 5
function = "PiecewiseNearest"
function = "PiecewiseQuadratic"
function = "PiecewiseZero"
function = "PiecewiseSlinear"
function = "PiecewiseCubic"
function = "PiecewiseNext"
function = "Ramsay"
function = "CPA"
function = "PiecewiseLinear"
count_paths(p, n, function, samples=500, plot=True)

# %%
p_arr = np.arange(4,100,10)
# p_arr = [102]
function_arr = [
    "PiecewiseNearest",
    "PiecewiseZero",
    "PiecewiseSlinear",
    "PiecewiseLinear",
    "PiecewiseCubic",
    "PiecewiseNext",
    "PiecewiseQuadratic",
    "CPA",
    "Ramsay"
]
results = [[p, f, count_paths(p, n, f, samples=500, plot=False)] 
    for f in function_arr for p in p_arr]
# results = np.array(results)
results

# %% PLOT RESULTS

results_df = pd.DataFrame(results, columns=["parameters", "function", "error"])
plt.figure(figsize=(6,4), constrained_layout=True)
sns.lineplot(x=results_df.parameters, y=results_df.error, hue=results_df.function)
plt.grid()
plt.savefig("expressivity.png", dpi=300)

# %% METRICS
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def paths_metrics(p, n, function="PiecewiseLinear", samples=2000, plot=False):
    print(p, n, function, samples)
    
    metrics = []

    x = np.linspace(0, 1, n)
    if plot:
        y_samples = []
        yp_samples = []
        ypp_samples = []
        yppp_samples = []

    T = None
    if function == "CPA":
        T = cpab.Cpab(tess_size=p, basis="svd")

    xp = None
    method = "exponential"
    if "Piecewise" in function:
        xp = np.linspace(0, 1, p+1)

    for i in range(samples):  
        y = generate_path(x, p, function, method, xp, T)
        yp = np.diff(y)
        ypp = np.diff(yp)
        yppp = np.diff(ypp)

        # pos = np.where(yp < 0)[0]
        # y[pos] = y[pos-1]
        
        # yp = np.clip(yp, 0.0, None)
        # y = np.cumsum(yp)
        # if np.any(yp < 0):
        #     break

        # yp = yp / np.linalg.norm(yp, ord=np.inf)
        # ypp = ypp / np.linalg.norm(ypp, ord=np.inf)
        # yppp = yppp / np.linalg.norm(yppp, ord=np.inf)
        # ypp = ypp / np.linalg.norm(ypp, ord=np.inf)
        # k = np.abs(np.append(ypp, 0)) / (1 + yp**2)**(3/2)

        ord = 2
        s = np.linalg.norm(y, ord=ord)
        sp = np.linalg.norm(yp, ord=ord)
        spp = np.linalg.norm(ypp, ord=ord)
        sppp = np.linalg.norm(yppp, ord=ord)

        f = lambda x: np.mean(np.abs(x))
        s = f(y)
        sp = f(yp)
        spp = f(ypp)
        sppp = f(yppp)
        length = 0 
        
        condition1 = np.sign(ypp * shift(ypp, 1)) == -1
        condition2 = ~np.isclose(ypp, 0)
        condition3 = ~np.isclose(shift(ypp, 1), 0)
        length = np.sum(condition1 & condition2 & condition3)
        length = np.sum(condition1)
        # length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


        metrics.append([p, n, function, length, s, sp, spp, sppp])
        if plot:
            y_samples.append(y)
            yp_samples.append(yp)
            ypp_samples.append(ypp)
            yppp_samples.append(yppp)
    
    
    if plot:
        fig, ax = plt.subplots(1, 4, constrained_layout=True, figsize=(10, 3))
        
        ax[0].plot(np.array(y_samples).T, color="blue", alpha=0.1)
        ax[1].plot(np.array(yp_samples).T, color="blue", alpha=0.1)
        ax[2].plot(np.array(ypp_samples).T, color="blue", alpha=0.1)
        ax[3].plot(np.array(yppp_samples).T, color="blue", alpha=0.1)

    return metrics

n = 1000
p = 6
function = "PiecewiseZero"
function = "PiecewiseQuadratic"
function = "PiecewiseCubic"
function = "CPA"
function = "PiecewiseLinear"
function = "PiecewiseNext"
function = "Ramsay"
metrics = paths_metrics(p, n, function, samples=1, plot=True)
metrics

# %% 
p_arr = [6]
function_arr = [
    # "PiecewiseLinear",
    "PiecewiseQuadratic",
    # "PiecewiseNext",
    # "PiecewiseZero",
    "PiecewiseCubic",
    "CPA",
    "Ramsay"
]
metrics = [paths_metrics(p, n, f, samples=200) 
    for f in function_arr for p in p_arr]

import itertools
metrics = list(itertools.chain(*metrics))

# %% PLOT RESULTS
metrics_df = pd.DataFrame(metrics, columns=["parameters", "n", "function", "length", "s", "sp", "spp", "sppp"])
plt_df = metrics_df.melt(id_vars=["parameters", "n", "function"])

# sns.displot(metrics_df, x="length", bins=20)
g = sns.FacetGrid(plt_df, 
    hue="function", col="variable", 
    sharex=False, sharey=False, height=3.5, aspect=.65)
g.map(sns.histplot, "value", bins=30, common_bins=False, common_norm=False, 
    multiple="fill", element="step", alpha=0.1)
g.add_legend()
# %%
