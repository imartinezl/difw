# %%

import difw

# %%

tess_size = 5
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True # [True, False]
# use_slow = False # [True, False]
outsize = 100
batch_size = 1
basis = "svd" # ["svd", "sparse", "rref", "qr"]
basis = "sparse"
basis = "qr"
basis = "rref"

T = difw.Cpab(tess_size, backend, device, zero_boundary, basis)
# T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
theta = T.sample_transformation_with_prior(batch_size)
theta = T.identity(batch_size, epsilon=1)
grid_t = T.transform_grid(grid, theta)

T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, method='numeric')
T.visualize_gradient(theta)
T.visualize_gradient(theta, method="numeric")

# %% Data Transform

width = 50
channels = 3

# Generation
# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)
T.visualize_deformdata(data, theta)


