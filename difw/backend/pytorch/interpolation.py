import torch


def interpolate(data, grid, outsize):
    # Problem size
    n_batch, width, n_channels = data.shape

    # Extract points
    x = grid.flatten()

    # Scale to domain
    x = x * (width - 1)

    # Do sampling
    x0 = torch.floor(x).to(torch.int64)
    x1 = x0 + 1

    # Clip values
    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)

    # Batch effect
    # r = torch.arange(n_batch).repeat(outsize)
    # r = torch.arange(n_batch).repeat(outsize, 1).t().flatten()
    r = torch.arange(n_batch).repeat_interleave(outsize)

    # Index
    y0 = data[r, x0, :]
    y1 = data[r, x1, :]

    # Interpolation weights
    xd = (x - x0.to(torch.float32)).reshape((-1, 1))

    # Do interpolation
    y = y0 * (1 - xd) + y1 * xd

    newdata = torch.reshape(y, (n_batch, outsize, n_channels))

    return newdata


def interpolate_grid(data):
    # Problem size
    n_batch, width = data.shape

    # Rescale to interval [0-1] per batch
    # xmin, xmax = data[:,0].view(-1,1), data[:,-1].view(-1,1)
    # xdelta = xmax - xmin
    # data = (data - xmin) / xdelta

    # Extract points
    x = data.flatten()

    # Scale to domain
    x = x * (width - 1)

    # Do sampling
    x0 = torch.floor(x).to(torch.int64)
    x1 = x0 + 1

    # Clip values
    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)

    # Batch effect
    # r = torch.arange(n_batch).repeat_interleave(width)
    r = torch.arange(n_batch).view(-1, 1).repeat(1, width).flatten()

    # Index
    y0 = data[r, x0]
    y1 = data[r, x1]

    # Interpolation weights
    xd = (x - x0).to(torch.float32)

    # Do interpolation
    y = y0 * (1 - xd) + y1 * xd

    newdata = torch.reshape(y, (n_batch, width))

    # newdata = xmin + newdata * xdelta
    return newdata
