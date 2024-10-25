# hostsub_gp/host_model.py

__all__ = ["gp"]

from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp

import jaxopt

from tinygp import GaussianProcess, kernels, transforms


class _gp:
    def __init__(
        self,
        X,
        y=None,
        params: dict = None,
        params_init: dict = {
            "log_amp": jnp.float64(0),
            "log_scale": jnp.zeros(1, dtype=jnp.float64),
            "log_jitter": jnp.float64(-6),
            "mean": jnp.float64(0),
        },
        optimization: bool = False,
    ) -> None:
        self.params_init = params_init

        if params is not None:
            self.params = params
            self.gp = _build_gp(params, X)
        if optimization:
            assert y is not None, "y must be provided for optimization"
            self.params = self.optimize(X, y)
            self.gp = _build_gp(self.params, X)

    def optimize(self, X, y, verbose: bool = False) -> dict:
        solver = jaxopt.ScipyMinimize(fun=_neg_log_prob)
        soln = solver.run(self.params_init, X=X, y=y)
        if verbose:
            print(f"Optimization status: {soln.status}")
            print(f"Final parameters: {soln.params}")
            print(f"Final negative log likelihood: {soln.state.fun_val}")
        return soln.params


def _build_gp(params: dict, X: any) -> GaussianProcess:
    log_amp = params.get("log_amp", jnp.float64(0))
    log_scale = params.get("log_scale", jnp.zeros(1, dtype=jnp.float64))
    jitter = params.get("log_jitter", jnp.float64(-6))
    mean = params.get("mean", jnp.float64(0))
    kernel = 10**log_amp * transforms.Linear(10 ** (-log_scale), kernel=kernels.ExpSquared())
    gp = GaussianProcess(kernel=kernel, X=jnp.asarray(X), diag=10**jitter, mean=mean)
    return gp


@jax.jit
def _neg_log_prob(params, X, y):
    gp = _build_gp(params, X)
    return -gp.log_probability(y)
