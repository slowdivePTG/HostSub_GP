# hostsub_gp/host_model.py

__all__ = ["gp"]

from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp

import jaxopt

from tinygp import GaussianProcess, kernels, transforms


class gp:
    def __init__(
        self,
        X,
        y,
        mean: any = jnp.float64(0),
        params: dict = None,
        params_init: dict = {
            "log_amp": jnp.float64(0.0),
            "log_scale": jnp.zeros(1, dtype=jnp.float64),
            "jitter": jnp.float64(1e-6),
        },
        optimization: bool = True,
    ) -> None:
        self.params_init = params_init

        if params is not None:
            self.params = params
            self.gp = build_gp(params, X, mean)
        if optimization:
            self.params = self.optimize(X, y, mean)
            self.gp = build_gp(self.params, X, mean)

    def optimize(self, X, y, mean) -> dict:
        solver = jaxopt.ScipyMinimize(fun=neg_log_prob)
        soln = solver.run(self.params_init, X=X, y=y, mean=mean)
        print(f"Final parameters: {soln.params}")
        print(f"Final negative log likelihood: {soln.state.fun_val}")
        return soln.params


def build_gp(params: dict, X: any, mean: any = jnp.float64(0)) -> GaussianProcess:
    log_amp = params.get("log_amp", jnp.float64(0.0))
    log_scale = params.get("log_scale", jnp.zeros(1, dtype=jnp.float64))
    jitter = params.get("jitter", jnp.float64(1e-6))
    kernel = 10**log_amp * transforms.Linear(10 ** (-log_scale), kernel=kernels.ExpSquared())
    gp = GaussianProcess(kernel=kernel, X=jnp.asarray(X), diag=jitter, mean=mean)
    return gp


@jax.jit
# @jax.value_and_grad
def neg_log_prob(params, X, y, mean):
    gp = build_gp(params, X, mean)
    return -gp.log_probability(y)
