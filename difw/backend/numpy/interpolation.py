import numpy as np


def interpolate(data, grid, outsize):
    # Problem size
    n_batch, width, n_channels = data.shape

    # Extract points
    x = grid.flatten()

    # Scale to domain
    x = x * (width - 1)

    # Do sampling
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1

    # Clip values
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)

    # Batch effect
    r = np.arange(n_batch).repeat(outsize)

    # Index
    y0 = data[r, x0, :]
    y1 = data[r, x1, :]

    # Interpolation weights
    xd = (x - x0.astype(np.float32)).reshape((-1, 1))

    # Do interpolation
    y = y0 * (1 - xd) + y1 * xd

    newdata = np.reshape(y, (n_batch, outsize, n_channels))
    return newdata


def interpolate_grid(data):
    # Problem size
    n_batch, width = data.shape

    # Extract points
    x = data.flatten()

    # Scale to domain
    x = x * (width - 1)

    # Do sampling
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1

    # Clip values
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)

    # Batch effect
    r = np.arange(n_batch).repeat(width)

    # Index
    y0 = data[r, x0]
    y1 = data[r, x1]

    # Interpolation weights
    xd = x - x0.astype(np.float32)

    # Do interpolation
    y = y0 * (1 - xd) + y1 * xd

    newdata = np.reshape(y, (n_batch, width))
    return newdata
