# hostsub_gp/_gp.py

__all__ = ["_transform_unbound_to_bound", "_transform_bound_to_unbound", "_check_params", "_split_params"]

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _transform_unbound_to_bound(params, params_limit=None):
    """Transform unbounded parameters to the bounded space with the sigmoid function."""
    if params_limit is None:
        return params
    transformed_params = {}
    for key, val in params.items():
        if key in params_limit:
            lower, upper = params_limit[key]
            transformed_params[key] = lower + (upper - lower) * jax.nn.sigmoid(val)
        else:
            transformed_params[key] = val
    return transformed_params


def _transform_bound_to_unbound(params, params_limit=None):
    """Transform bounded parameters to the unbounded space with the inverse sigmoid function."""
    if params_limit is None:
        return params
    transformed_params = {}
    for key, val in params.items():
        if key in params_limit:
            lower, upper = params_limit[key]
            transformed_params[key] = jnp.log((val - lower) / (upper - val))
            if ~jnp.isfinite(transformed_params[key]).all():
                transformed_params[key] = (lower + upper) / 2.0
        else:
            transformed_params[key] = val
    return transformed_params


def _check_params(params, required_all: bool = False):
    """Check if the parameters are valid."""
    if params is None:
        raise ValueError("params must be provided.")
    if required_all:
        key_required = ["log_amp", "log_scale", "mean"]
        for key in key_required:
            if key not in params:
                raise ValueError(f"params must contain {key}")


def _split_params(params: dict, require_all: bool = False) -> tuple[dict, dict]:
    """
    Split the parameters into 1D and 2D.
    """
    keys_required = ["log_amp", "log_scale", "mean"]
    keys = keys_required + ["log_jitter"]  # jitter is optional

    params_1d, params_2d = {}, {}

    if params is not None:
        for key in keys:
            key_1d = key + "_1d"
            key_2d = key + "_2d"
            if require_all and key in keys_required:
                if (params.get(key_1d) is None) or (params.get(key_2d) is None):
                    jax.debug.print("params: {}", params)
                    raise ValueError(f"Missing key: {key_1d} or {key_2d}")
            if key_1d in params:
                params_1d[key] = params[key_1d]
            if key_2d in params:
                params_2d[key] = params[key_2d]

    return params_1d, params_2d
