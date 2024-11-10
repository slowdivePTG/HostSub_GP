# hostsub_gp/gp.py

__all__ = ["GP"]

from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

from tinygp import GaussianProcess, kernels, transforms

from jax._src.typing import ArrayLike, Array

from ._gp import _transform_unbound_to_bound, _transform_bound_to_unbound, _check_params


class GP:
    """
    Gaussian Process.
    A wrapper around tinygp.GaussianProcess.
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        yerr: ArrayLike | float = None,
        params: dict = None,
        params_init: dict = None,
        params_limit: dict = None,
        kernel_type: str = "ExpSquared",
        optimization: bool = False,
        verbose: bool = False,
    ):
        """Initialize the Gaussian Process."""
        # Initialize the input arrays
        self.X = jnp.asarray(X)
        if yerr is None:
            self.yerr = jnp.zeros_like(y)
        elif isinstance(yerr, (int, float)):
            self.yerr = jnp.ones_like(y) * yerr
        else:
            self.yerr = jnp.asarray(yerr)

        # Initialize the parameters
        self.params_limit = params_limit if params_limit is not None else {}
        if optimization:
            try:
                _check_params(params_init, required_all=True)
            except ValueError as e:
                raise ValueError("Optimization: " + str(e))
            if y is None:
                raise ValueError("Optimization: y must be provided")
            self.y = jnp.asarray(y)

            self.params_init = _transform_bound_to_unbound(params_init, self.params_limit)
            self.params = self.optimize(X, self.y, self.yerr, verbose=verbose)
        else:
            try:
                _check_params(params, required_all=True)
            except ValueError as e:
                raise ValueError("Initializating GP: " + str(e))
            self.params = _transform_unbound_to_bound(params, self.params_limit)

        # Build the GP
        self.gp = _build_gp(self.params, self.X, self.yerr)(kernel_type=kernel_type)

    def optimize(self, X: Array, y: Array, yerr: Array, verbose: bool = False) -> dict:
        solver = jaxopt.ScipyMinimize(fun=_neg_log_prob)
        neg_log_prob_init = _neg_log_prob(self.params_init, self.params_limit, X, y, yerr)
        if ~jnp.isfinite(neg_log_prob_init):
            raise ValueError("Invalid initial parameters")
        soln = solver.run(self.params_init, X=X, y=y, yerr=yerr, params_limit=self.params_limit)
        params = soln.params
        params_bound = _transform_unbound_to_bound(params, self.params_limit)
        if verbose:
            self._print_params(params_bound)
            print(f"Final negative log-probability: {_neg_log_prob(params_bound, None, X, y, yerr):.1f}")
        return params_bound

    def _print_params(self, params: dict = None):
        if params is None:
            params = self.params
        try:
            _check_params(params)
        except ValueError as e:
            raise ValueError("Printing parameters: " + str(e))
        print("Amp: {:.3e}".format(10 ** params.get("log_amp")))
        print("Scale: " + ",".join(["{:.3e}".format(10**log_scale) for log_scale in params.get("log_scale")]))
        if "log_jitter" in params:
            print("Jitter: {:.3e}".format(10 ** params.get("log_jitter")))
        print("Mean: {:.3e}".format(params.get("mean")))


class _build_gp:
    """Build the Gaussian Process."""

    def __init__(self, params: dict, X: Array, yerr: Array):
        # Check if necessary parameters are provided
        try:
            _check_params(params, required_all=True)
        except ValueError as e:
            raise ValueError("Building GP: " + str(e))

        self.mean = jnp.asarray(params.get("mean"), dtype=jnp.float64)
        self.log_amp = jnp.asarray(params.get("log_amp"), dtype=jnp.float64)
        self.log_scale = jnp.asarray(params.get("log_scale"), dtype=jnp.float64)
        if params.get("log_jitter") is None:
            self.log_jitter = jnp.ones_like(self.mean) * -6
        else:
            self.log_jitter = jnp.asarray(params.get("log_jitter"), dtype=jnp.float64)

        # Check the validity of the parameter dimensions
        if not (self.log_amp.size == self.log_jitter.size == self.mean.size):
            print(self.log_amp.size, self.log_scale.size, self.log_jitter.size, self.mean.size)
            raise ValueError("Invalid shape of kernel parameters")

        self.X = jnp.asarray(X)
        self.yerr = jnp.asarray(yerr)

    def __call__(self, kernel_type: str = "ExpSquared") -> GaussianProcess:
        kernel = self._build_kernel(kernel_type)
        return GaussianProcess(kernel=kernel, X=self.X, diag=10**self.log_jitter + self.yerr**2, mean=self.mean)

    def _build_kernel(self, kernel_type: str) -> kernels.Kernel:
        amp = 10**self.log_amp
        scale = 10**self.log_scale
        if kernel_type != "composite":
            if self.log_amp.size != 1:
                raise ValueError("The kernel requires only 1 set of parameters")
            if kernel_type == "ExpSquared":
                kernel = amp * transforms.Linear(1 / scale, kernel=kernels.ExpSquared())
            elif kernel_type == "Matern":
                kernel = amp * transforms.Linear(1 / scale, kernel=kernels.Matern52())
        elif kernel_type == "composite":
            if self.log_amp.size != 2:
                raise ValueError("The composite kernel requires 2 set of parameters")
            kernel_expsqr = amp[0] * transforms.Linear(1 / scale[0], kernel=kernels.ExpSquared())
            kernel_matern = amp[1] * transforms.Linear(1 / scale[1], kernel=kernels.Matern52())
            kernel = kernel_expsqr + kernel_matern
        else:
            raise ValueError("Invalid kernel type: supported types are 'ExpSquared', 'Matern', and 'composite'")
        return kernel


@partial(jax.jit, static_argnames=("kernel_type",))
def _neg_log_prob(
    params: dict, params_limit: dict, X: Array, y: Array, yerr: Array, kernel_type: str = "ExpSquared"
) -> jnp.float64:
    """Negative log-probability of the Gaussian Process."""
    params = _transform_unbound_to_bound(params, params_limit)
    gp = _build_gp(params, X, yerr)(kernel_type=kernel_type)
    neg_log_prob = -gp.log_probability(y)
    return neg_log_prob
