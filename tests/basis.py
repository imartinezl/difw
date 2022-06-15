# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import difw
from tqdm import tqdm 

# %%
from numpy.linalg import cond
from scipy.linalg import orth, norm, pinv

def plot_basis_velocity(T, B=None):
    if B is not None:
        T.params.B = B
    plt.figure()
    for k in range(T.params.d):
        theta = T.identity()

        theta[0][k] = 1

        grid = T.uniform_meshgrid(outsize)
        v = T.calc_velocity(grid, theta)

        # Plot
        plt.plot(grid, v.T)
        plt.grid()
        plt.title("CPA Velocity basis with: " + T.params.basis)
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.savefig("basis_" + T.params.basis + ".png", dpi=300)

def sparsity(sparse):
    return (sparse == 0).sum() / sparse.size

def orthonormality(A):
    return norm(A.T.dot(A) - np.eye(A.shape[1]))

def orthonormal(A):
    return np.allclose(A.T.dot(A), np.eye(A.shape[1]))

def orthogonality(A):
    E = A.T.dot(A)
    return np.sum(~np.isclose(E - np.diag(np.diagonal(E)), np.zeros_like(E))) / A.size

def orthogonal(A):
    return orthogonality(A) == 0

# %% 
# The Sparse Null Space Basis Problem

tess_size = 5
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 100
batch_size = 1

basis = "svd"
T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
B_svd = T.params.B
plot_basis_velocity(T)

basis = "rref"
T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
B_rref = T.params.B
plot_basis_velocity(T)

basis = "sparse"
T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
B_sparse = T.params.B
plot_basis_velocity(T)

basis = "qr"
T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
B_qr = T.params.B
plot_basis_velocity(T)

L = T.tess.constrain_matrix()
sparsity(L)

# %% Properties
def properties_table(results):
    header = ["basis", "sparsity", "cond num", "orth", "orthnorm", "norm", "norm inv"]
    print(" | ".join(header))
    print("-"*70)
    for v in results:
        name = v[0]
        B = v[1]
        Binv = pinv(B)
        values = [
            name, 
            np.round(sparsity(B),2), 
            np.round(cond(B),2), 
            orthogonal(B), 
            np.round(orthogonality(B), 2), 
            orthonormal(B), 
            np.round(orthonormality(B), 2), 
            np.round(norm(B),2), 
            np.round(norm(Binv),2)
        ]
        print("\t".join(map(str, values)))

results = [("svd", B_svd), ("rref", B_rref), ("sparse", B_sparse), ("qr", B_qr)]
properties_table(results)

# %% MATLAB

from scipy.io import loadmat
B1 = loadmat("/home/imartinez/Documents/MATLAB/B1.mat")["B"]
B2 = loadmat("/home/imartinez/Documents/MATLAB/B2.mat")["B"]

B1 = B1 / norm(B1, axis=0)

plot_basis_velocity(T, B1)
plot_basis_velocity(T, B2)

properties_table([("B1", B1), ("B2", B2)])

# %%
 
def metric1(B):
    data = []
    for i in np.arange(0, B.shape[0], 2):
        for j in np.arange(0, B.shape[0], 2):
            d = np.sum(B[i, ] * B[j, ])
            data.append([i, j, d])

    return data

def metric2(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric3(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric4(B):
    data = []
    data = np.empty((B.shape[1], B.shape[1]))
    for i in np.arange(0, B.shape[1]):
        for j in np.arange(0, B.shape[1]):
            d = np.dot(B[:, i], B[:, j])
            # data.append([i, j, d])
            data[i,j] = d

    return np.round(data,2)

metric = metric4
metric(B_svd), metric(B_rref), metric(B_sparse)

# %%

plt.figure()
plt.spy(metric(B_svd), precision=1e-7)

plt.figure()
plt.spy(metric(B_rref), precision=1e-7)

plt.figure()
plt.spy(metric(B_sparse), precision=1e-7)

# %%

names = ["svd", "qr", "rref", "sparse"]
k = 0
fig, ax = plt.subplots(ncols=4, figsize=(8,4), sharex=True, sharey=True)
for B in [B_svd, B_qr, B_rref, B_sparse]:
    ax[k].spy(B)
    ax[k].set_title("Basis " + names[k])
    k += 1
plt.savefig("basis_spy.png", dpi=300)

# %%

x = 0
y = 0
for B in [B_svd, B_rref, B_sparse]:
    B = orth(B)
    plt.figure()
    for j in range(B.shape[1]):
        dx = B[0, j]
        dy = B[2, j]
        plt.arrow(x, y, dx, dy, head_width=0.04)

    plt.axis("equal")
    plt.scatter(0,0, color="orange")

# %%

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, minimize
from scipy.linalg import null_space, norm

fun = lambda x: norm(x-c)**2

n = 3
c = np.ones(n)*1
x0 = np.zeros(n)
fun(x0)

A = np.array([
    [1,4,0],
    [4,0,2]
])
m = A.shape[0]
ub = np.zeros(m)
lb = np.zeros(m)
constraint = LinearConstraint(A, lb, ub)
# res = minimize(fun, x0, constraints=constraint, method="SLSQP")
res = minimize(fun, x0, constraints=constraint, method="trust-constr",
    options={
        "factorization_method": None,
        "verbose": 3
        })
res.x

A = np.array([
    [2,3,5],
    [-4,2,3]
])
Ap = null_space(A)
Ap

