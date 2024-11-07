# hostsub_gp/host_model.py

__all__ = ["gp"]

from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp

import jaxopt
from functools import partial

from tinygp import GaussianProcess, kernels, transforms

from jax._src.typing import ArrayLike, Array


class _gp:
    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        yerr: ArrayLike | float = None,
        params: dict = None,
        params_init: dict = None,
        params_limits: dict = None,
        optimization: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the Gaussian Process."""
        # Initialize the input arrays
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        if yerr is None:
            self.yerr = jnp.zeros_like(y)
        if isinstance(yerr, (int, float)):
            self.yerr = jnp.ones_like(y) * yerr
        else:
            self.yerr = jnp.asarray(yerr)

        # Initialize the parameters
        self.params_init = (
            params_init
            if params_init is not None
            else {
                "log_amp": jnp.float64(0),
                "log_scale": jnp.zeros(1, dtype=jnp.float64),
                "log_jitter": jnp.float64(-6),
                "mean": jnp.float64(0),
            }
        )
        self.params_limits = params_limits if params_limits is not None else {}

        # Build the GP
        if optimization:
            if self.yerr is None:
                raise ValueError("yerr must be provided for optimization")
            self.params = self.optimize(X, self.y, self.yerr, verbose=verbose)
        else:
            if params is None:
                raise ValueError("params must be provided for non-optimization")
            self.params = params
        self.gp = _build_gp(self.params, self.X, self.yerr)

    def optimize(self, X: Array, y: Array, yerr: Array, verbose: bool = False) -> dict:
        solver = jaxopt.ScipyMinimize(fun=_neg_log_prob)
        soln = solver.run(self.params_init, X=X, y=y, yerr=yerr, params_limit=self.params_limits)
        params = transform_params(soln.params, self.params_limits)
        if verbose:
            self._print_params(params)
            print(f"Final negative log-probability: {_neg_log_prob(params, self.params_limits, X, y, yerr):.1f}")
        return params

    def _print_params(self, params: dict = None) -> None:
        if params is None:
            params = self.params
        print("Amp: {:.3e}".format(10 ** params.get("log_amp")))
        print("Scale: {:.3e}".format(10 ** params.get("log_scale")))
        print("Jitter: {:.3e}".format(10 ** params.get("log_jitter")))
        print("Mean: {:.3e}".format(params.get("mean")))


def _build_gp(params: dict, X: Array, yerr: Array) -> GaussianProcess:
    log_amp = params.get("log_amp", jnp.float64(0))
    log_scale = params.get("log_scale", jnp.zeros(1, dtype=jnp.float64))
    jitter = params.get("log_jitter", jnp.float64(-6))
    mean = params.get("mean", jnp.float64(0))
    kernel = 10**log_amp * transforms.Linear(10 ** (-log_scale), kernel=kernels.ExpSquared())
    gp = GaussianProcess(kernel=kernel, X=jnp.asarray(X), diag=10**jitter + yerr**2, mean=mean)
    return gp


@jax.jit
def _neg_log_prob(params: dict, params_limit: dict, X: Array, y: Array, yerr: Array) -> jnp.float64:
    params = transform_params(params, params_limit)
    gp = _build_gp(params, X, yerr)
    neg_log_prob = -gp.log_probability(y)
    return neg_log_prob

def transform_params(params, params_limit):
    transformed_params = {}
    for key, val in params.items():
        if key in params_limit:
            lower, upper = params_limit[key]
            transformed_params[key] = lower + (upper - lower) * jax.nn.sigmoid(val)
        else:
            transformed_params[key] = val
    return transformed_params