import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space


class Tessellation:
    def __init__(self, nc, xmin=0, xmax=1, zero_boundary=True, basis="custom"):
        self.nc = nc
        self.nv = nc + 1
        self.ns = nc - 1

        self.xmin = xmin
        self.xmax = xmax
        self.xr = self.xmax - self.xmin

        self.zero_boundary = zero_boundary

        if basis == "svd":
            self.L = self.constrain_matrix()
            self.B = self.basis_svd()
        elif basis == "custom":
            self.B = self.generate_basis()

        self.D, self.d = self.B.shape

        # TODO: additional methods?: get_cell_centers, find_verts, find_shared_verts

    def constrain_matrix(self):
        vertices = np.linspace(self.xmin, self.xmax, self.nv)
        shared_vertices = vertices[1:-1]

        rows = self.ns
        cols = 2 * self.nc
        L = np.zeros((rows, cols))
        for i, x in enumerate(shared_vertices):
            L[i, 2 * i : 2 * (i + 2)] = [x, 1, -x, -1]

        if self.zero_boundary:
            Ltemp = self.zero_boundary_constrains()
            L = np.vstack((L, Ltemp))

        return L

    def zero_boundary_constrains(self):
        L = np.zeros((2, 2 * self.nc))
        L[0, :2] = [-self.xmin, -1]
        L[1, -2:] = [-self.xmax, -1]
        return L

    def generate_basis(self):
        # return self.basis_svd()
        if self.zero_boundary:
            return self.basis_zb()
        else:
            return self.basis()

    def basis_svd(self):
        return null_space(self.L)

    # with zero boundary
    def basis_zb(self):
        rows = self.nc - 1
        cols = 2 * self.nc
        B = np.zeros((rows, cols))

        a = self.xmin
        b = self.xmax
        n = self.nc
        s = (b - a) / n
        B[:, 0] = a + s
        B[:, 1] = -a * (a + s)
        for k in range(1, self.nc):
            B[k - 1, 2 : 2 * k : 2] = s
            B[k - 1, 2 * k] = -(a + k * s)
            B[k - 1, 2 * k + 1] = (a + k * s) * (a + (k + 1) * s)

        # normalize
        B = B.T / np.linalg.norm(B, axis=1)
        return B

    # without zero boundary
    def basis(self):
        rows = self.nc + 1
        cols = 2 * self.nc
        B = np.zeros((rows, cols))

        a = self.xmin
        b = self.xmax
        n = self.nc
        s = (b - a) / n

        B[0, 0] = 1
        B[0, 1] = -(a + s)
        B[-2, ::2] = 1
        B[-1, :-2:2] = 1
        B[-1, -1] = a + (n - 1) * s
        for k in range(1, n - 1):
            B[k, : 2 * k : 2] = s
            B[k, 2 * k] = -(a + k * s)
            B[k, 2 * k + 1] = (a + k * s) * (a + (k + 1) * s)

        # normalize
        B = B.T / np.linalg.norm(B, axis=1)
        return B

    # with zero boundary
    def basis_zb_new(self):
        rows = self.nc - 1
        cols = 2 * self.nc
        B = np.zeros((rows, cols))

        a = self.xmin
        b = self.xmax
        n = self.nc
        B[:, 0] = (n - 1) * a + b
        B[:, 1] = -a * ((n - 1) * a + b)
        for k in range(1, self.nc):
            B[k - 1, 2 : 2 * k : 2] = b - a
            B[k - 1, 2 * k] = -((n - k) * a + k * b)
            B[k - 1, 2 * k + 1] = (
                ((n - k) * a + k * b) * ((n - k - 1) * a + (k + 1) * b) / n
            )

        # normalize
        B = B.T / np.linalg.norm(B, axis=1)
        return B

    # without zero boundary
    def basis_new(self):
        rows = self.nc + 1
        cols = 2 * self.nc
        B = np.zeros((rows, cols))

        a = self.xmin
        b = self.xmax
        n = self.nc

        B[0, 0] = n
        B[0, 1] = -((n - 1) * a + b)
        B[-2, ::2] = n
        B[-1, :-2:2] = n
        B[-1, -1] = a + (n - 1) * b
        for k in range(1, n - 1):
            B[k, : 2 * k : 2] = b - a
            B[k, 2 * k] = -((n - k) * a + k * b)
            B[k, 2 * k + 1] = (
                ((n - k) * a + k * b) * ((n - k - 1) * a + (k + 1) * b) / n
            )

        # normalize
        B = B.T / np.linalg.norm(B, axis=1)
        return B

    def plot_basis(self):
        plt.figure()
        plt.spy(self.B)
