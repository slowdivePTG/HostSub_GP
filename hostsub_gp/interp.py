# hostsub_gp/interp.py

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from jax._src.typing import ArrayLike, Array
from typing import Optional

from ._utils import msgs

# jax.config.update("jax_enable_x64", True)

###################################################################################################
################################# Interpolation on a regular grid #################################
###################################################################################################


class Interp1D_Grid:
    """
    1D interpolation using a regular grid.
    A wrapper around jax.numpy.interp.
    """

    def __init__(self, points: ArrayLike, values: ArrayLike, method="linear"):
        self.method = method
        self.points = points
        self.values = jnp.asarray(values)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        return jax.vmap(lambda x: jnp.interp(x, self.points, self.values))(x)


class Interp2D_Grid:
    """
    2D interpolation using a regular grid.
    A wrapper around jax.scipy.interpolate.RegularGridInterpolator.
    """

    def __init__(self, points: tuple[ArrayLike, ArrayLike], values: ArrayLike, method="linear"):
        self.method = method
        self.points = points
        self.values = jnp.array(values)

    def __call__(self, x):
        if self.method in ["nearest", "linear"]:
            return jax.scipy.interpolate.RegularGridInterpolator(self.points, self.values, method=self.method)(x)


###################################################################################################
################################# Interpolation on a irregular grid ###############################
###################################################################################################

from abc import ABC, abstractmethod


class Interp2D_base(ABC):
    """
    Abstract base class for 2D interpolation on semi-uniform grids.
    """

    def __init__(self, scales: tuple | ArrayLike = (1, 1)):
        """
        Initialize base interpolator for semi-uniform grid (nx, ny).

        Parameters
        ----------
        scales : tuple or ArrayLike
            Scales for each dimension
        """
        self.scales = jnp.asarray(scales)
        self.points = None
        self.values = None
        self.shape = None
        self.x_grid = None
        self.y_grid = None

    def fit(self, points: ArrayLike, values: ArrayLike) -> None:
        """
        Store the training data and grid information.

        Parameters
        ----------
        points : ArrayLike
            Array of shape (nx, ny, 2) containing the 2D coordinates
        values : ArrayLike
            Array of shape (nx, ny) containing the values at each point
        """
        points = jnp.asarray(points, dtype=jnp.float32)
        self.shape = points.shape[:2]
        if values.shape != self.shape:
            raise ValueError(f"Shape mismatch between points {(points.shape)} and values {(values.shape)}")

        self.points = points / self.scales
        self.values = jnp.asarray(values, dtype=jnp.float32)
        # Separate x and y coordinates for easier access
        self.x_grid = self.points[..., 0]
        self.y_grid = self.points[..., 1]

    # @partial(jax.jit, static_argnums=(0,))
    def _find_nearest_cell(self, query_point: Array) -> tuple[Array, Array]:
        """
        Find the cell in the semi-uniform grid containing or nearest to the query point
        using binary search (searchsorted).

        Parameters
        ----------
        query_point : Array
            Point to interpolate at, shape (2,)

        Returns
        -------
        Tuple[Array, Array]
            i, j: indices of the lower-left corner of the containing cell
        """
        qx, qy = query_point

        # def _searchsorted_vector(x, q):
        #     return jax.vmap(lambda x: jnp.searchsorted(x, q))(x)

        # breakpoint()
        # # Find the first i index in each row where x[i] > qx
        # i_index = _searchsorted_vector(self.x_grid, qx) - 1
        # i_index = jnp.clip(i_index, 0, self.shape[0] - 2)

        # # Find the first j indice in the array consisting of y values of the found i row where y[j] > qy
        # j_index = jnp.searchsorted(jnp.array([self.y_grid[i_index[idx], idx] for idx in range(self.shape[1])]), qy) - 1
        # j_index = jnp.clip(j_index, 0, self.shape[1] - 2)

        # First search for x position in middle column
        mid_row = self.shape[1] // 2
        i_mid = jnp.searchsorted(self.x_grid[:, mid_row], qx) - 1
        i_mid = jnp.clip(i_mid, 0, self.shape[0] - 2)

        # Get y values for this row
        y_row = self.y_grid[i_mid, :]
        j_index = jnp.searchsorted(y_row, qy) - 1
        j_index = jnp.clip(j_index, 0, self.shape[1] - 2)

        # Refine x search in the found column
        i_index = jnp.searchsorted(self.x_grid[:, j_index], qx) - 1
        i_index = jnp.clip(i_index, 0, self.shape[0] - 2)

        return i_index, j_index

    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def _get_cell_points_and_values(self, i: int, j: int, di: int = 4, dj: int = 2) -> tuple[Array, Array]:
        """
        Get the neighboring cells of a grid cell and their values.

        Parameters
        ----------
        i, j : int
            Indices of the lower-left corner of the cell
        di, dj : int, optional
            Number of cells to include in each direction, default is 2

        Returns
        -------
        Tuple[Array, Array]
            points: Array of shape (4, 2) containing corner coordinates
            values: Array of shape (4,) containing corner values
        """
        i_idx = jnp.arange(-di // 2, di // 2)
        j_idx = jnp.arange(-dj // 2, dj // 2)
        i_idx = jnp.clip(i + i_idx, 0, self.shape[0] - 1)
        j_idx = jnp.clip(j + j_idx, 0, self.shape[1] - 1)

        # Stacking all di x dj points (with meshgrid)
        i_grid, j_grid = jnp.meshgrid(i_idx, j_idx, indexing="ij")
        i_grid = i_grid.ravel()
        j_grid = j_grid.ravel()
        points = jnp.stack([self.x_grid[i_grid, j_grid], self.y_grid[i_grid, j_grid]], axis=1)
        values = self.values[i_grid, j_grid]

        return points, values

    @abstractmethod
    def _compute_weights(self, cell_points: Array, query_point: Array, cell_values: Optional[Array] = None) -> Array:
        """
        Compute interpolation weights for given cell points and query point.
        To be implemented by subclasses.

        Parameters
        ----------
        cell_points : Array
            Corner points of the cell, shape (4, 2)
        query_point : Array
            Point to interpolate at, shape (2,)
        cell_values : Array, optional
            Values at the corner points, shape (4,)

        Returns
        -------
        Array
            Weights for interpolation, shape (4,)
        """
        raise NotImplementedError

    # @partial(jax.jit, static_argnums=(0,))
    def _interpolate_single(self, query_point: Array) -> Array:
        """
        Interpolate value at a single query point with NaN handling
        """
        # Find containing/nearest cell
        i, j = self._find_nearest_cell(query_point)

        # Get cell points and values
        cell_points, cell_values = self._get_cell_points_and_values(i, j)

        # Check for NaN values
        valid_mask = ~jnp.isnan(cell_values)
        n_valid = jnp.sum(valid_mask)

        def interpolate():
            # Compute weights using subclass implementation
            weights = self._compute_weights(cell_points, query_point, cell_values)
            # Handle NaN values by redistributing weights
            weights = jnp.where(valid_mask, weights, 0.0)
            weights = weights / (jnp.sum(weights) + 1e-10)
            return jnp.sum(jnp.where(valid_mask, cell_values, 0.0) * weights)

        # Return NaN if not enough valid values
        return jax.lax.cond(n_valid >= 3, lambda: interpolate(), lambda: jnp.nan)

    def predict(self, query_points: ArrayLike) -> Array:
        """
        Make predictions at new points with NaN handling

        Parameters
        ----------
        query_points : ArrayLike
            Array of shape (n_queries, 2) containing points to interpolate

        Returns
        -------
        Array
            Array of interpolated values at query_points
        """
        query_points = jnp.asarray(query_points)

        # Check for NaN in query points
        valid_queries = ~jnp.any(jnp.isnan(query_points), axis=1)

        # Scale query points
        scaled_points = query_points / self.scales

        # Vectorize the single point interpolation
        predictions = jax.vmap(self._interpolate_single)(scaled_points)

        # Ensure NaN for invalid query points
        return jnp.where(valid_queries, predictions, jnp.nan)


class Interp2D_Linear(Interp2D_base):
    """
    2D bilinear interpolation on semi-uniform grid.
    """

    def _compute_weights(self, cell_points: Array, query_point: Array, cell_values: Optional[Array] = None) -> Array:
        """
        Compute weights for bilinear interpolation
        """
        # Weights proportional to the inverse of the distance
        distances = jnp.linalg.norm(cell_points - query_point, axis=1)
        # Handle zero distance (exact match)
        return jax.lax.cond(
            jnp.any(distances < 1e-10), lambda: jnp.where(distances == 0, 1.0, 0.0), lambda: 1.0 / distances
        )


class Interp2D_RBF(Interp2D_base):
    """
    2D interpolation using radial basis functions.
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        epsilon: float = 1.0,
        scales: tuple | ArrayLike = (1, 1),
    ):
        """
        Initialize RBF interpolator

        Parameters
        ----------
        kernel : str
            Type of RBF kernel ('gaussian', 'multiquadric', or 'inverse_multiquadric')
        epsilon : float
            Shape parameter for the RBF kernel
            Minimum number of valid neighbors required for interpolation
        scales : tuple or ArrayLike
            Scales for each dimension
        """
        super().__init__(scales)
        self.epsilon = epsilon
        self.kernel = self._get_kernel(kernel)

    def _get_kernel(self, kernel_name: str) -> Callable:
        """Define the RBF kernel function"""

        def gaussian(r):
            return jnp.where(jnp.isfinite(r), jnp.exp(-((self.epsilon * r) ** 2)), 0)

        def multiquadric(r):
            return jnp.where(jnp.isfinite(r), jnp.sqrt(1 + (self.epsilon * r) ** 2), 0)

        def inverse_multiquadric(r):
            return jnp.where(jnp.isfinite(r), 1 / jnp.sqrt(1 + (self.epsilon * r) ** 2), 0)

        kernels = {"gaussian": gaussian, "multiquadric": multiquadric, "inverse_multiquadric": inverse_multiquadric}
        return kernels[kernel_name]

    def _compute_weights(self, cell_points: Array, query_points: Array, cell_values: Array) -> Array:
        """
        Compute weights using RBF kernel
        """
        distances = jnp.linalg.norm(cell_points - query_points, axis=1)
        kernel_matrix = self.kernel(distances)
        try:
            weights = jnp.linalg.solve(kernel_matrix, cell_values)
            return self.kernel(distances) @ weights
        except:
            msgs.warning("Singular matrix in RBF interpolation")
            return jax.lax.cond(
                jnp.any(distances < 1e-10), lambda: jnp.where(distances == 0, 1.0, 0.0), lambda: 1.0 / distances
            )

class Interp2D_Scipy(Interp2D_base):
    """
    2D interpolation using scipy's griddata.
    """

    def __init__(
        self,
        method: str = "linear",
        scales: tuple | ArrayLike = (1, 1),
    ):
        """
        Initialize Scipy interpolator

        Parameters
        ----------
        method : str
            Type of interpolation method ('linear', 'nearest', 'cubic')
        scales : tuple or ArrayLike
            Scales for each dimension
        """
        super().__init__(scales)
        self.method = method

    def _compute_weights(self, 
                        cell_points: Array, 
                        query_point: Array,
                        dx: float,
                        dy: float) -> Array:
        """
        Compute weights using scipy's griddata. This is a dummy implementation.
        Actual interpolation is done in the predict method.
        """
        return jnp.ones(4)  # This is not used in this class

    def predict(self, query_points: ArrayLike) -> Array:
        """
        Make predictions at new points with scipy's griddata

        Parameters
        ----------
        query_points : ArrayLike
            Array of shape (n_queries, 2) containing points to interpolate

        Returns
        -------
        Array
            Array of interpolated values at query_points
        """
        from scipy.interpolate import griddata
        
        query_points = jnp.asarray(query_points) / self.scales

        # Flatten the grid points and values
        flat_points = self.points.reshape(-1, 2)
        flat_values = self.values.ravel()

        # Use scipy's griddata for interpolation
        interpolated_values = griddata(
            flat_points,
            flat_values,
            query_points,
            method=self.method
        )

        return jnp.asarray(interpolated_values)