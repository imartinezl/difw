# %%

import numpy as np
import matplotlib.pyplot as plt
from .core.utility import Parameters
from .core.tessellation import Tessellation



class Cpab:
    """ Core class for this library. This class contains all the information
        about the tesselation, transformation ect. The user is not meant to
        use anything else than this specific class.
        
    Arguments:
        tess_size: list, with the number of cells in each dimension
        
        backend: string, computational backend to use. Choose between 
            "numpy" (default), "pytorch" or "tensorflow"
        
        device: string, either "cpu" (default) or "gpu". For the numpy backend
            only the "cpu" option is valid
        
        zero_boundary: bool, determines is the velocity at the boundary is zero 
        
        
    Methods:
        @uniform_meshgrid
        @sample_transformation
        @sample_transformation_with_prior
        @identity
        @transform_grid
        @interpolate
        @transform_data
        @get_velocity
        @visualize_velocity
        @visualize_deformgrid
        @visualize_tesselation
    """

    def __init__(self, tess_size, backend="python", device="cpu", zero_boundary=True):
        # Check input
        self._check_input(tess_size, backend, device, zero_boundary)

        # Parameters
        self.params = Parameters()
        self.params.nc = tess_size
        self.params.zero_boundary = zero_boundary
        self.params.xmin = 0
        self.params.xmax = 1
        self.params.nSteps1 = 10
        self.params.nSteps2 = 10
        self.params.precomputed = False # TODO: review if necessary
        self.params.use_slow = False

        # Initialize tesselation
        self.tess = Tessellation(
            self.params.nc,
            self.params.xmin,
            self.params.xmax,
            self.params.zero_boundary,
            basis="custom",
        )

        # Extract parameters from tesselation
        self.params.B = self.tess.B
        self.params.D, self.params.d = self.tess.D, self.tess.d

        self.backend_name = backend

        # Load backend and set device
        self.device = device.lower()
        self.backend_name = backend
        if self.backend_name == "numpy":
            from .backend.numpy import functions as backend
        elif self.backend_name == "numba":
            pass
            # from .tensorflow import functions as backend
        elif self.backend_name == "pytorch":
            from .backend.pytorch import functions as backend
            self.params.B = backend.to(self.params.B, device=self.device)
            # torch.tensor(params.B, dtype=torch.float32)
        self.backend = backend

        # Assert that we have a recent version of the backend
        self.backend.assert_version()

    def uniform_meshgrid(self, n_points):
        """ Constructs a meshgrid 
        Arguments:
            n_points: number of points
        Output:
            grid: vector of points
        """
        return self.backend.uniform_meshgrid(
            self.params.xmin, self.params.xmax, n_points, self.device
        )

    def sample_transformation(self, n_sample, mean=None, cov=None):
        """ Method for sampling transformation from simply multivariate gaussian
            As default the method will sample from a standard normal
        Arguments:
            n_sample: integer, number of transformations to sample
            mean: [d,] vector, mean of multivariate gaussian
            cov: [d,d] matrix, covariance of multivariate gaussian
        Output:
            samples: [n_sample, d] matrix. Each row is a independent sample from
                a multivariate gaussian
        """
        if mean is not None:
            self._check_type(mean)
            self._check_device(mean)
        if cov is not None:
            self._check_type(cov)
            self._check_device(cov)

        samples = self.backend.sample_transformation(
            self.params.d, n_sample, mean, cov, self.device
        )

        return self.backend.to(samples, device=self.device)

    # TODO: prepare backend functions to sample with prior
    def sample_transformation_with_prior(
        self, n_sample, mean=None, cov=None, length_scale=0.1, output_variance=1
    ):
        """ Function for sampling smooth transformations. The smoothness is determined
            by the distance between cell centers. The closer the cells are to each other,
            the more the cell parameters should correlate -> smooth transistion in
            parameters. The covariance in the D-space is calculated using the
            squared exponential kernel.
                
        Arguments:
            n_sample: integer, number of transformation to sample
            mean: [d,] vector, mean of multivariate gaussian
            length_scale: float>0, determines how fast the covariance declines 
                between the cells 
            output_variance: float>0, determines the overall variance from the mean
        Output:
            samples: [n_sample, d] matrix. Each row is a independent sample from
                a multivariate gaussian
        """

        cov_theta = self.backend.matmul()
        samples = self.sample_transformation(n_sample, mean=mean, cov=cov_theta)

        # Smoothness priors on CPA
        s = output_variance  # 0.01
        l = length_scale  # 0.1
        kernel = lambda x, y: (s ** 2) * np.exp(-((x - y) ** 2) / (2 * l ** 2))

        D, d = B.shape
        grid = np.linspace(0, 1, D)
        X, Y = np.meshgrid(grid, grid)
        cov_pa = kernel(X, Y)

        cov_cpa = B.T.dot(cov_pa).dot(B)

        theta = sample_transformation(d, mean, cov_cpa)
        return theta

    def identity(self, n_sample=1, epsilon=0):
        """ Method for getting the parameters for the identity 
            transformation (vector of zeros) 
        Arguments:
            n_sample: integer, number of transformations to sample
            epsilon: float>0, small number to add to the identity transformation
                for stability during training
        Output:
            samples: [n_sample, d] matrix. Each row is a sample    
        """
        return self.backend.identity(self.params.d, n_sample, epsilon, self.device)

    def transform_grid(self, grid, theta, mode=None):
        """ Main method of the class. Integrates the grid using the parametrization
            in theta.
        Arguments:
            grid: [n_points] vector or [n_batch, n_points] tensor i.e.
                either a single grid for all theta values, or a grid for each theta
                value
            theta: [n_batch, d] matrix,
            mode: one of {"closed_form", "numeric"}
        Output:
            transformed_grid: [n_batch, n_points] tensor, with the transformed
                grid. The slice transformed_grid[i] corresponds to the grid being
                transformed by theta[i]
        """
        self._check_type(grid)
        self._check_device(grid)
        self._check_type(theta)
        self._check_device(theta)
        transformed_grid = self.backend.transformer(grid, theta, self.params, mode)
        return transformed_grid

    def gradient_grid(self, grid, theta, mode=None):
        """ Integrates and return the gradient of the transformation
        using the parametrization in theta.
        Arguments:
            grid: [n_points] vector or [n_batch, n_points] tensor i.e.
                either a single grid for all theta values, or a grid for each theta
                value
            theta: [n_batch, d] matrix,
        Output:
            transformed_grid: [n_batch, n_points] tensor, with the transformed
                grid. The slice transformed_grid[i] corresponds to the grid being
                transformed by theta[i]
            gradient_grid: [n_batch, n_points, d] tensor, with the gradient grid.
                The slice gradient_grid[i, :, j] corresponds to the gradient of grid being
                transformed by theta[i] and with respect to the parameter theta[i,j]
        """
        self._check_type(grid)
        self._check_device(grid)
        self._check_type(theta)
        self._check_device(theta)
        transformed_grid = self.backend.gradient(grid, theta, self.params, mode)
        return transformed_grid

    def interpolate(self, data, grid, outsize):
        """ Linear interpolation method
        Arguments:
            data: [n_batch, *data_shape] tensor, with input data. The format of
                the data_shape depends on the backend that is being used. 
                In tensorflow and numpy: [n_batch, number_of_features, n_channels]
                In pytorch: [n_batch, n_channels, number_of_features]
            grid: [n_batch, n_points] tensor with grid points that are 
                used to interpolate the data
            outsize: number of points in the output
        Output:
            interpolated: [n_batch, outsize, n_channels] tensor with the interpolated data
        """
        self._check_type(data)
        self._check_device(data)
        self._check_type(grid)
        self._check_device(grid)
        return self.backend.interpolate(data, grid, outsize)

    def transform_data(self, data, theta, outsize, mode=None):
        """ Combination of the transform_grid and interpolate methods for easy
            transformation of data.
        Arguments:
            data: [n_batch, *data_shape] tensor, with input data. The format of
                the data_shape depends on the backend that is being used. 
                In tensorflow and numpy: [n_batch, number_of_features, n_channels]
                In pytorch: [n_batch, n_channels, number_of_features]
            theta: [n_batch, d] matrix with transformation parameters. Each row
                correspond to a transformation.
            outsize: number of points that is transformed and interpolated
        Output:
            data_t: [n_batch, outsize, n_channels] tensor, transformed and interpolated data
        """

        self._check_type(data)
        self._check_device(data)
        self._check_type(theta)
        self._check_device(theta)
        # TODO: DONE assert n_batch in data and theta are the same
        assert (
            data.shape[0] == theta.shape[0]
        ), """Batch sizes should be the same on arguments data and theta"""
        grid = self.uniform_meshgrid(outsize)
        grid_t = self.transform_grid(grid, theta, mode)
        data_t = self.interpolate(data, grid_t, outsize)
        return data_t

    def calc_velocity(self, grid, theta):
        """ For each point in grid, calculate the velocity of the point based
            on the parametrization in theta
        Arguments:
            grid: [n_points] vector
            theta: [1, d] single parametrization vector
        Output:    
            v: [n_points] vector with velocity vectors for each point
        """
        self._check_type(grid)
        self._check_device(grid)
        self._check_type(theta)
        self._check_device(theta)
        v = self.backend.calc_velocity(grid, theta, self.params)
        return v

    def visualize_velocity(self, theta, n_points=None, fig=None):
        """ Utility function that helps visualize the vectorfield for a specific
            parametrization vector theta 
        Arguments:    
            theta: [n_batch, d] single parametrization vector
            n_points: number of points to plot
            fig: matplotlib figure handle
        Output:
            plot: handle to lines plot
        """
        self._check_type(theta)
        if fig is None:
            fig = plt.figure()

        if n_points is None:
            n_points = self.params.nc

        # Calculate vectorfield and convert to numpy
        grid = self.uniform_meshgrid(n_points)
        v = self.calc_velocity(grid, theta)

        grid = self.backend.tonumpy(grid)
        v = self.backend.tonumpy(v)

        # Plot
        ax = fig.add_subplot(1, 1, 1)
        # ax.quiver(grid, np.zeros_like(grid), np.zeros_like(v), v)
        ax.axhline(color="black", ls="dashed")
        alpha = max(0.01, 1/np.sqrt(len(theta)))
        ax.plot(grid, v.T, color="blue", alpha=alpha)
        ax.plot(grid, v.T, marker='o', linestyle="None", color="orange", alpha=alpha)
        ax.set_xlim(self.params.xmin, self.params.xmax)
        ax.set_title("Velocity Field " + r'$v(x)$')
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$v(x)$', rotation='horizontal')
        ax.grid(alpha=0.3)
        return ax

    def visualize_deformgrid(self, theta, mode=None, n_points=100, fig=None):
        """ Utility function that helps visualize a deformation
        Arguments:
            theta: [n_batch, d] single parametrization vector
            n_points: int, number of points
            fig: matplotlib figure handle
        Output:
            plot: handle to lineplot
        """
        pass
        if fig is None:
            fig = plt.figure()

        # Transform grid and convert to numpy
        grid = self.uniform_meshgrid(n_points)
        grid_t = self.transform_grid(grid, theta, mode)
        grid = self.backend.tonumpy(grid)
        grid_t = self.backend.tonumpy(grid_t)

        # Plot
        ax = fig.add_subplot(1, 1, 1)
        ax.axline((0, 0), (1, 1), color="black", ls="dashed")
        alpha = max(0.01, 1/np.sqrt(len(theta))) 
        ax.plot(grid, grid_t.T, c="blue", alpha=alpha)
        ax.set_title("Grid Deformation " + r'$\phi(x,t)$')
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$\phi(x,t)$', rotation='horizontal')
        ax.grid(alpha=0.3)
        
        return ax

    def visualize_tesselation(self, n_points=50, fig=None):
        """ Utility function that helps visualize the tesselation.
        Arguments:
            n_points: number of points in each dimension
            fig: matplotlib figure handle
        Output:
            plot: handle to tesselation plot
        """
        if fig is None:
            fig = plt.figure()

        grid = self.uniform_meshgrid(n_points)

        # Find cellindex and convert to numpy
        idx = self.backend.get_cell(grid, self.params)
        idx = self.backend.tonumpy(idx)
        grid = self.backend.tonumpy(grid)

        # Plot
        ax = fig.add_subplot(1, 1, 1)
        # plot = ax.scatter(grid.flatten(), np.zeros_like(grid).flatten(), c=idx)
        plot = ax.scatter(grid.flatten(), idx, c=idx)
        ax.set_title("Tesselation " + r'$N_\mathcal{P}=$' + str(self.params.nc))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel("Cell Index")
        ax.set_yticks(np.arange(np.max(idx)+1))
        # ax.tick_params(left=False, labelleft=False)
        ax.grid(alpha=0.3)
        return plot

    def visualize_velocity2deform(self, theta, mode=None, n_points=100, fig=None):
        """ Utility function that helps visualize deformation vs velocity
        Arguments:
            theta: [n_batch, d] single parametrization vector
            n_points: int, number of points
            fig: matplotlib figure handle
        Output:
            plot: handle to lineplot
        """
        pass
        if fig is None:
            fig = plt.figure()

        # Calculate velocity, transform grid and convert to numpy
        grid = self.uniform_meshgrid(n_points)
        grid_t = self.transform_grid(grid, theta, mode)
        v = self.calc_velocity(grid, theta)

        grid = self.backend.tonumpy(grid)
        grid_t = self.backend.tonumpy(grid_t)
        v = self.backend.tonumpy(v)

        # Plot
        ax = fig.add_subplot(2, 1, 1)
        ax.axhline(color="black", ls="dashed")
        ax.plot(grid, v.T, color="blue", alpha=0.1)
        ax.set_title("Velocity Field " + r'$v(x)$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$v(x)$')
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(2, 1, 2)
        ax.axhline(color="black", ls="dashed")
        ax.plot(grid, (grid_t - grid).T, c="blue", alpha=0.1)
        ax.set_title("Grid Deformation " + r'$\phi(x,t)$')
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$\phi(x,t)$', rotation='horizontal')
        ax.grid(alpha=0.3)

        return ax

    def visualize_gradient(self, theta, mode=None, n_points=100, fig=None):
        """ Utility function that helps visualize the gradient
        Arguments:
            theta: [n_batch, d] single parametrization vector
            n_points: int, number of points
            fig: matplotlib figure handle
        Output:
            plot: handle to lineplot
        """
        if fig is None:
            fig = plt.figure()

        # Gradient grid and convert to numpy
        grid = self.uniform_meshgrid(n_points)
        grad = self.gradient_grid(grid, theta, mode)
        grid = self.backend.tonumpy(grid)
        grad = self.backend.tonumpy(grad)

        # Plot only gradient
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(color="black", ls="dashed")
        for g in grad:
            ax.plot(grid, g)
        ax.set_title("Gradient " + r'$\dfrac{\partial \phi(x,t)}{\partial \theta}$')
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$\dfrac{\partial \phi(x,t)}{\partial \theta}$', rotation='horizontal', labelpad=10)
        ax.grid(alpha=0.3)
        return ax


    def _check_input(self, tess_size, backend, device, zero_boundary):
        """ Utility function used to check the input to the class.
            Not meant to be called by the user. """
        assert tess_size > 0, """tess size must be positive"""
        assert backend in [
            "numpy",
            "tensorflow",
            "pytorch",
        ], """Unknown backend, choose between 'numpy', 'tensorflow' or 'pytorch' """
        assert device in [
            "cpu",
            "gpu",
        ], """Unknown device, choose between 'cpu' or 'gpu' """
        if backend == "numpy":
            assert device == "cpu", """Cannot use gpu with numpy backend """
        assert (
            type(zero_boundary) == bool
        ), """Argument zero_boundary must be True or False"""
        # TODO: review backend possibilities

    def _check_type(self, x):
        """ Assert that the type of x is compatible with the class i.e
                numpy backend expects np.array
                pytorch backend expects torch.tensor
                tensorflow backend expects tf.tensor
        """
        assert isinstance(
            x, self.backend.backend_type()
        ), """ Input has type {0} but expected type {1} """.format(
            type(x), self.backend.backend_type()
        )

    def _check_device(self, x):
        """ Assert that x is on the same device (cpu or gpu) as the class """
        assert self.backend.check_device(
            x, self.device
        ), """Input is placed on 
            device {0} but the class expects it to be on device {1}""".format(
            str(x.device), self.device
        )

    def __repr__(self):
        output = """
        CPAB transformer class. 
            Parameters:
                Tesselation size:           {0}
                Theta size:                 {1}
                Domain lower bound:         {2}
                Domain upper bound:         {3}
                Zero Boundary:              {4}
            Backend:                        {5}
        """.format(
            self.params.nc,
            self.params.d,
            self.params.xmin,
            self.params.xmax,
            self.params.zero_boundary,
            self.backend_name,
        )
        return output

    def test(self, grid, theta):
        return self.backend.derivative(grid, theta, self.params)


# %%
