# hostsub_gp/gp.py

__all__ = ["GP"]

import numpy as np
from scipy import integrate

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

from tinygp import GaussianProcess, kernels, transforms
from tinygp.kernels.distance import L1Distance, L2Distance

from jax._src.typing import ArrayLike, Array

from ._par import _transform_unbound_to_bound, _transform_bound_to_unbound, _init_params, _check_params, _print_params
from ._msgs import msgs

import warnings


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
        **kwargs,
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
        self.kernel_type = kernel_type

        # Initialize the parameters
        self.params_limit = params_limit if params_limit is not None else {}
        if optimization:
            try:
                self.params_init = _init_params(params_init)
            except Exception as e:
                raise ValueError("Optimization: " + str(e))
            self.params_init_unbound = _transform_bound_to_unbound(self.params_init, self.params_limit)

            if y is None:
                raise ValueError("Optimization: y must be provided")
            self.y = jnp.asarray(y)

            self.params_unbound = self._optimize(X, self.y, self.yerr)
            self.params = _transform_unbound_to_bound(self.params_unbound, self.params_limit)
            _print_params(self.params)
        else:
            self.params = params
            self.params_unbound = _transform_bound_to_unbound(self.params, self.params_limit)

        # Build the GP
        self.gp = _build_gp(self.params, self.X, self.yerr)(kernel_type=self.kernel_type, **kwargs)

    def _optimize(self, X: Array, y: Array, yerr: Array) -> dict:
        solver = jaxopt.ScipyMinimize(fun=partial(_neg_log_prob, kernel_type=self.kernel_type))
        valid = jnp.isfinite(y)

        # Check if the initial parameters are valid
        neg_log_prob_init = _neg_log_prob(
            self.params_init_unbound,
            params_limit=self.params_limit,
            X=X[valid],
            y=y[valid],
            yerr=yerr[valid],
            kernel_type=self.kernel_type,
        )
        if ~jnp.isfinite(neg_log_prob_init):
            msgs.parameter(f"Initial parameters (bound):")
            _print_params(self.params_init)
            msgs.parameter(f"Initial parameters (unbound):")
            _print_params(self.params_init_unbound)
            raise ValueError("Invalid initial parameters")
            
        soln = solver.run(
            self.params_init_unbound, params_limit=self.params_limit, X=X[valid], y=y[valid], yerr=yerr[valid]
        )
        params_unbound = soln.params
        msgs.parameter(f"Initial negative log-probability: {neg_log_prob_init:.1f}")
        msgs.parameter(f"Final negative log-probability: {soln.state.fun_val:.1f}")
        return params_unbound


@partial(jax.jit, static_argnames=("kernel_type",))
def _neg_log_prob(
    params: dict, params_limit: dict, X: Array, y: Array, yerr: Array, kernel_type: str = "ExpSquared"
) -> jnp.float64:
    """Negative log-probability of the Gaussian Process."""
    params = _transform_unbound_to_bound(params, params_limit)
    gp = _build_gp(params, X, yerr)(kernel_type=kernel_type)
    neg_log_prob = -gp.log_probability(y)
    return neg_log_prob


class _build_gp:
    """Build the Gaussian Process."""

    def __init__(self, params: dict, X: Array, yerr: Array):
        # Check if necessary parameters are provided
        try:
            _check_params(params)
        except ValueError as e:
            raise ValueError("Building GP: " + str(e))

        # Initialize the parameters
        self.log_amp = params.get("log_amp")
        self.log_scale = params.get("log_scale")
        self.mean = params.get("mean")
        # For EmissionLine kernel only
        self.log_amp_line = params.get("log_amp_line")
        self.scale_line = params.get("scale_line")

        self.X = jnp.asarray(X)
        self.yerr = jnp.asarray(yerr)

    def __call__(self, kernel_type: str = "ExpSquared", **kwargs) -> GaussianProcess:
        kernel = self._build_kernel(kernel_type, **kwargs)
        return GaussianProcess(kernel=kernel, X=self.X, diag=self.yerr**2, mean=self.mean)

    def _build_kernel(self, kernel_type: str, **kwargs) -> kernels.Kernel:
        amp = 10**self.log_amp
        scale = 10**self.log_scale

        # Standard single kernel types
        if not kernel_type in ["composite", "EmissionLine"]:
            if self.log_amp.size != 1:
                raise ValueError(f"The {kernel_type} kernel requires only 1 set of parameters")
            if kernel_type == "ExpSquared":
                kernel = amp * transforms.Linear(1 / scale, kernel=kernels.ExpSquared())
            elif kernel_type == "Matern":
                kernel = amp * transforms.Linear(1 / scale, kernel=kernels.Matern52(distance=L2Distance()))

        # Composite kernel - combination of ExpSquared and Matern52
        elif kernel_type == "composite":
            # if self.log_amp.size != 2:
            #     raise ValueError("The composite kernel requires 2 set of parameters")
            if self.log_amp.ndim == 1:
                # kernel1 : ExpSquared - long-term variations (continuum)
                kernel_expsqr = amp[0] * transforms.Linear(1 / scale[0], kernel=kernels.ExpSquared())
                # kernel2 : Matern - short-term variations (sky lines, emission lines)
                kernel_matern = amp[1] * transforms.Linear(1 / scale[1], kernel=kernels.Matern52(distance=L2Distance()))
                kernel = kernel_expsqr + kernel_matern
            else:
                # spatially varying parameters
                # TODO: only 3 free parameters in amp
                # kernel1 : ExpSquared - long-term variations (continuum)
                # evaluate the kernel only on the spatial coordinates
                kernel_spat_expsqr = amp[0, 0] * transforms.Linear(1 / scale[0, 0], kernel=kernels.ExpSquared())
                # kernel2 : Matern - short-term variations (sky lines, emission lines)
                kernel_spat_matern = amp[0, 1] * transforms.Linear(
                    1 / scale[0, 1], kernel=kernels.Matern52(distance=L2Distance())
                )
                kernel_spat = OneDKernel(kernel=kernel_spat_expsqr + kernel_spat_matern, axis=0)
                # spectral varying parameters
                # kernel1 : ExpSquared - long-term variations (continuum)
                kernel_spec_expsqr = amp[1, 0] * transforms.Linear(1 / scale[1, 0], kernel=kernels.ExpSquared())
                # kernel2 : Matern - short-term variations
                kernel_spec_matern = amp[1, 1] * transforms.Linear(
                    1 / scale[1, 1], kernel=kernels.Matern52(distance=L2Distance())
                )
                kernel_spec = OneDKernel(kernel=kernel_spec_expsqr + kernel_spec_matern, axis=1)
                kernel = kernel_spat * kernel_spec

        # EmissionLine kernel - to handle discontinuities at narrow emission lines
        elif kernel_type == "EmissionLine":
            if self.log_amp_line is None or self.scale_line is None:
                raise ValueError("EmissionLine kernel requires 'amp_line', and 'scale_line' parameters")
            emission_lines = kwargs.get("emission_lines")
            if emission_lines is None:
                warnings.warn(
                    "EmissionLine kernel: emission_lines not provided, the kernel is equivalent to ExpSquared"
                )
            base_kernel = amp * transforms.Linear(1 / scale, kernel=kernels.Matern52(distance=L2Distance()))
            emission_line_kernel = EmissionLineKernel(
                amp_line=10**self.log_amp_line,
                scale_line=self.scale_line,
                emission_lines=emission_lines,
            )
            kernel = base_kernel * emission_line_kernel

        # Invalid kernel type
        else:
            raise ValueError(
                "Invalid kernel type: supported types are 'ExpSquared', 'Matern', 'composite', 'EmissionLine'"
            )

        return kernel


class OneDKernel(kernels.Kernel):
    """A kernel only evaluated on the spatial coordinates."""

    kernel: kernels.Kernel
    axis: int

    def evaluate(self, X1: Array, X2: Array) -> Array:
        return self.kernel.evaluate(X1[..., self.axis], X2[..., self.axis])


class EmissionLineKernel(kernels.Kernel):
    """A kernel to handle discontinuities at narrow emission lines in the spectroscopic data."""

    amp_line: float
    scale_line: float
    emission_lines: ArrayLike

    def evaluate(self, X1: Array, X2: Array) -> Array:
        """Evaluate the kernel."""

        # Split coordinates
        x1_spec, x2_spec = X1[..., -1], X2[..., -1]

        k_x1_x2 = 1
        # Add emission line effects
        for line in self.emission_lines:
            # Calculate proximity to emission line for each point
            # x1_close = _gaussian(x1_spec, line, self.scale_line)
            # x2_close = _gaussian(x2_spec, line, self.scale_line)
            # x1_close = _sigmoid(x1_spec, line, self.scale_line)
            # x2_close = _sigmoid(x2_spec, line, self.scale_line)
            # x1_close = _tophat(x1_spec, line, self.scale_line)
            # x2_close = _tophat(x2_spec, line, self.scale_line)
            x1_close = _hyperbolic_tangent(x1_spec, line, self.scale_line)
            x2_close = _hyperbolic_tangent(x2_spec, line, self.scale_line)

            # Effect when both x1 and x2 are close to the line
            both_close = x1_close * x2_close

            # Effect when exactly one of x1 and x2 is close to the line
            one_close = x1_close * (1 - x2_close) + (1 - x1_close) * x2_close

            # Emission line effect
            # - decrease the covariance when exactly one point is close to the line
            # - increase the covariance when both points are close to the line
            emission_line_effect = 1.0 - one_close + self.amp_line * both_close

            k_x1_x2 *= emission_line_effect

        return k_x1_x2


@jax.jit
def _sigmoid(x: Array, mu: Array, width: Array) -> Array:
    return 1 / (1 + jnp.exp((jnp.abs(x - mu) / width - 1.0) / 0.1))


@jax.jit
def _tophat(x: Array, mu: Array, width: Array) -> Array:
    return jnp.where(jnp.abs(x - mu) < width, 1.0, 0.0)


@jax.jit
def _gaussian(x: Array, mu: Array, width: Array) -> Array:
    return jnp.exp(-0.5 * (x - mu) ** 2 / width**2)


@jax.jit
def _hyperbolic_tangent(x: Array, mu: Array, width: Array) -> Array:
    x_min, x_max = mu - width / 2, mu + width / 2
    return 0.5 * (jnp.tanh(2 * (x - x_min) / width) - jnp.tanh(2 * (x - x_max) / width))
