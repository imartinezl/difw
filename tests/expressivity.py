# %%

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

# %%


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


# %%

from scipy.interpolate import interp1d

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


def CPA(grid, T):
    # T = cpab.Cpab(tess_size=3)
    theta = T.sample_transformation(1)
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
            

def count_paths(p, n, function="PiecewiseLinear", samples=2000, plot=False):
    print(p, n, function, samples)
    
    data = np.zeros((n,n))
    x = np.linspace(0, 1, n)

    # p = 3 # number of intervals
    # samples = 10000
    # sampler = "uniform"

    if function == "CPA":
        T = cpab.Cpab(tess_size=p, basis="svd")

    if "Piecewise" in function:
        xp = np.linspace(0, 1, p+1)

    for i in range(samples):  

        if "Piecewise" in function:      
            yp = sampler(p, "exponential")
            
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
        elif function == "CPA":
            y = CPA(x, T)

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
        fig, ax = plt.subplots(1, 4, constrained_layout=True, figsize=(11, 7))
        
        d = np.log((data+1)/np.max(data))
        im0 = ax[0].imshow(d, origin="lower", cmap=cmap)
        ax[0].set_title("Path Count (log)")
        ax[0].set_xticks(np.arange(0, n + 1, n // 5))
        ax[0].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im0, ax=ax[0], location="bottom", 
            ticks=[-6, -4, -2, 0])

        d = np.log(path_norm)
        im1 = ax[1].imshow(d, origin="lower", cmap=cmap)
        ax[1].set_title("Delannoy Count (log)")
        ax[1].set_xticks(np.arange(0, n + 1, n // 5))
        ax[1].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im1, ax=ax[1], location="bottom", 
            # ticks=[-6, -4, -2, 0])
            ticks=[-160, -120, -80, -40, 0])

        d = np.log(path_norm)
        im1b = ax[2].imshow(d, origin="lower", cmap=cmap, vmin=-7)
        ax[2].set_title("Delannoy Count Subset (log)")
        ax[2].set_xticks(np.arange(0, n + 1, n // 5))
        ax[2].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im1b, ax=ax[2], location="bottom", 
            ticks=[-6, -4, -2, 0])
        
        import matplotlib.colors as colors
        d = data/samples - path_norm
        im2 = ax[3].imshow(d, origin="lower", norm=colors.CenteredNorm(), cmap="seismic")
        ax[3].set_title("Path - Delannoy")
        ax[3].set_xticks(np.arange(0, n + 1, n // 5))
        ax[3].set_yticks(np.arange(0, n + 1, n // 5))
        plt.colorbar(im2, ax=ax[3], location="bottom")

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
p = 10
function = "PiecewiseNearest"
function = "PiecewiseQuadratic"
function = "PiecewiseZero"
function = "PiecewiseSlinear"
function = "PiecewiseLinear"
function = "CPA"
count_paths(p, n, function, samples=1000, plot=True)

# %%
p_arr = np.arange(2,150,10)
# p_arr = [102]
function_arr = [
    # "PiecewiseNearest",
    # "PiecewiseZero",
    # "PiecewiseSlinear",
    "PiecewiseLinear",
    # "PiecewiseQuadratic",
    "CPA"
]
results = [[p, f, count_paths(p, n, f, samples=2000, plot=False)] 
    for f in function_arr for p in p_arr]
# results = np.array(results)
results

# %%

results_df = pd.DataFrame(results, columns=["parameters", "function", "error"])
plt.figure(figsize=(6,4), constrained_layout=True)
sns.lineplot(x=results_df.parameters, y=results_df.error, hue=results_df.function)
plt.grid()
# plt.savefig("expressivity.png", dpi=300)

# %%
n = 100
parr = [3,4,5,10,20,50]
parr = np.arange(3,20,5)
errors = [count_paths(p, n) for p in parr]
plt.plot(parr, errors)

# %%

