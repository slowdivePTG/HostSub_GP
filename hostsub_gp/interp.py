import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial
from typing import Callable

import numpy as np

jax.config.update("jax_enable_x64", True)


class Interp1D_Grid:
    """
    1D interpolation using a regular grid.
    A wrapper around jax.numpy.interp.
    """

    def __init__(self, points: tuple[jnp.ndarray, jnp.ndarray], values: jnp.ndarray, method="linear"):
        self.method = method
        self.points = points
        self.values = jnp.array(values)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        return jax.vmap(lambda x: jnp.interp(x, self.points, self.values))(x)


class Interp2D_Grid:
    """
    2D interpolation using a regular grid.
    A wrapper around jax.scipy.interpolate.RegularGridInterpolator.
    """

    def __init__(self, points: tuple[jnp.ndarray, jnp.ndarray], values: jnp.ndarray, method="linear"):
        self.method = method
        self.points = points
        self.values = jnp.array(values)

    def __call__(self, x):
        if self.method in ["nearest", "linear"]:
            return jax.scipy.interpolate.RegularGridInterpolator(self.points, self.values, method=self.method)(x)


class Interp2D_RBF:
    """
    2D interpolation using radial basis functions.
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        epsilon: float = 1.0,
        n_neighbors: int = 10,
        min_neighbors: int = 3,
        scales: tuple | list = (1, 1),
    ):
        """
        Initialize Robust RBF interpolator with NaN handling

        Args:
            kernel: Type of RBF kernel ('gaussian', 'multiquadric', or 'inverse_multiquadric')
            epsilon: Shape parameter for the RBF kernel
            n_neighbors: Maximum number of nearest neighbors to use
            min_neighbors: Minimum number of valid neighbors required for interpolation
        """
        self.epsilon = epsilon
        self.kernel = self._get_kernel(kernel)
        self.n_neighbors = n_neighbors
        self.min_neighbors = min_neighbors
        self.scales = scales

        self.points = None
        self.values = None

    def _get_kernel(self, kernel_name: str) -> Callable:
        """Define the RBF kernel function"""

        def gaussian(r):
            return jnp.exp(-((self.epsilon * r) ** 2))

        def multiquadric(r):
            return jnp.sqrt(1 + (self.epsilon * r) ** 2)

        def inverse_multiquadric(r):
            return 1 / jnp.sqrt(1 + (self.epsilon * r) ** 2)

        kernels = {"gaussian": gaussian, "multiquadric": multiquadric, "inverse_multiquadric": inverse_multiquadric}
        return kernels[kernel_name]

    @partial(jax.jit, static_argnums=(0,))
    def _compute_distances(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise distances between points"""
        diff = x1[:, None] - x2
        return jnp.sqrt(jnp.sum(diff**2, axis=-1))

    def _find_valid_neighbors(self, query_point: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Find valid k-nearest neighbors for a query point, excluding NaN values

        Returns:
            Tuple of (valid_indices, valid_points, valid_values)
        """
        # Compute distances to all points
        distances = jnp.sum((self.points - query_point) ** 2, axis=1)

        # Create mask for valid points and values
        valid_points_mask = ~jnp.any(jnp.isnan(self.points), axis=1)
        valid_values_mask = ~jnp.isnan(self.values)
        valid_mask = valid_points_mask & valid_values_mask

        # Set distances for invalid points to infinity
        distances = jnp.where(valid_mask, distances, jnp.inf)

        # Get indices of nearest valid neighbors
        indices = jnp.argsort(distances)[: self.n_neighbors]

        return (indices, self.points[indices], self.values[indices])

    def fit(self, points: jnp.ndarray, values: jnp.ndarray) -> None:
        """
        Store the training data and handle initial NaN values

        Args:
            points: Array of shape (n_points, 2) containing the 2D coordinates
            values: Array of shape (n_points,) containing the values at each point
        """
        self.points = jnp.array(points) / jnp.array(self.scales)
        self.values = jnp.array(values)

        # Check if we have enough valid data points
        valid_points_mask = ~jnp.any(jnp.isnan(points), axis=1)
        valid_values_mask = ~jnp.isnan(values)
        valid_mask = valid_points_mask & valid_values_mask
        valid_count = jnp.sum(valid_mask)

        if valid_count < self.min_neighbors:
            raise ValueError(
                f"Not enough valid data points. Found {valid_count}, " f"need at least {self.min_neighbors}"
            )

    @partial(jax.jit, static_argnums=(0,))
    def _interpolate_single(self, query_point: jnp.ndarray) -> jnp.ndarray:
        """Interpolate value at a single query point with NaN handling"""
        # Find valid nearest neighbors
        valid_indices, neighbor_points, neighbor_values = self._find_valid_neighbors(query_point)

        # Check if we have enough valid neighbors
        n_valid = valid_indices.shape[0]

        def interpolate():
            # Compute local RBF interpolation
            distances = self._compute_distances(neighbor_points, neighbor_points)
            kernel_matrix = self.kernel(distances)

            # Add regularization term
            kernel_matrix = kernel_matrix + jnp.eye(n_valid) * 1e-10

            # Solve local system with robust solver
            try:
                weights = jnp.linalg.solve(kernel_matrix, neighbor_values)
                query_distances = self._compute_distances(jnp.expand_dims(query_point, 0), neighbor_points)
                query_kernel = self.kernel(query_distances[0])
                return jnp.dot(query_kernel, weights)
            except:
                return jnp.nan

        # Return NaN if not enough valid neighbors
        return jax.lax.cond(n_valid >= self.min_neighbors, lambda: interpolate(), lambda: jnp.nan)

    def predict(self, query_points: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions at new points with NaN handling

        Args:
            query_points: Array of shape (n_queries, 2) containing points to interpolate

        Returns:
            Array of interpolated values at query_points, with NaN for points
            where interpolation failed
        """
        query_points = jnp.array(query_points)

        # Check for NaN in query points
        valid_queries = ~jnp.any(jnp.isnan(query_points), axis=1)

        # Vectorize the single point interpolation
        predictions = jax.vmap(self._interpolate_single)(query_points / jnp.array(self.scales))

        # Ensure NaN for invalid query points
        return jnp.where(valid_queries, predictions, jnp.nan)
