# hostsub_gp/_gp.py

__all__ = [
    "_transform_unbound_to_bound",
    "_transform_bound_to_unbound",
    "_init_params",
    "_check_params",
    "_print_params",
]

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from functools import wraps
from typing import Callable

DictInput = dict | tuple[dict, dict] | list[dict]
DictOutput = dict | list[dict]


def sequence_input(func: Callable[[dict], dict]) -> Callable[[DictInput], DictOutput]:
    """Decorator to handle sequence-like inputs."""

    @wraps(func)
    def wrapper(arg: DictInput, *args, **kwargs) -> DictOutput:
        if isinstance(arg, (tuple, list)):
            if not all((isinstance(item, dict) | (item is None)) for item in arg):
                raise TypeError("All elements in tuple must be dictionaries")
            return [func(item, *args, **kwargs) for item in arg]
        if not (isinstance(arg, dict) | (arg is None)):
            raise TypeError("Input must be a dictionary or tuple of dictionaries")
        return func(arg, *args, **kwargs)

    return wrapper


@sequence_input
def _check_params(params: dict, require_all: bool = True):
    """Check if the parameters are valid."""
    if params is None:
        raise ValueError("params must be provided")
    if require_all:
        key_required = ["log_amp", "log_scale", "mean"]
        for key in key_required:
            if key not in params:
                raise ValueError(f"params must contain {key}")


@sequence_input
def _init_params(params: dict, require_all: bool = True, params_type: str = "value") -> dict:
    """Initialize the parameters with appropriate typing."""
    if params is None:
        return None

    _check_params(params, require_all=require_all)

    def ensure_scalar_or_array(x):
        x = jnp.asarray(x)
        if x.size > 1:
            return jnp.asarray(x, dtype=jnp.float64)  # Array[float64]
        elif x.ndim == 0:
            return jnp.asarray(x, dtype=jnp.float64)  # Scalar[float64]
        else:
            return jnp.asarray(x[0], dtype=jnp.float64)  # Scalar[float64]

    params_output = {}
    for key in ["log_amp", "log_scale", "mean"]:
        if key in params:
            params_output[key] = ensure_scalar_or_array(params.get(key))

    # if "log_jitter" in params:
    #     params_output["log_jitter"] = ensure_scalar_or_array(params.get("log_jitter"))
    # elif params_type == "value":
    #     params_output["log_jitter"] = jnp.asarray(-6.0, dtype=jnp.float64)

    return params_output


@sequence_input
def _print_params(params: dict):
    """Print the parameters."""

    def _print_param(params, key, key_str):
        vals = 10 ** params.get(key) if "log" in key else params.get(key)
        if jnp.ndim(vals) == 0:
            print(f"{key_str}: {vals:.3e}")
        elif jnp.ndim(vals) == 1:
            print(f"{key_str}: " + ", ".join([f"{val:.3e}" for val in vals]))
        else:
            print(f"{key_str}: " + ", ".join(["({})".format(", ".join([f"{v:.3e}" for v in val])) for val in vals]))

    try:
        _check_params(params)
        _print_param(params, "log_amp", "Amp")
        _print_param(params, "log_scale", "Scale")
        # if "log_jitter" in params:
        #     _print_param(params, "log_jitter", "Jitter")
        _print_param(params, "mean", "Mean")
    except TypeError as e:
        raise TypeError("Invalid type for params: " + str(e))


def _transform_unbound_to_bound(
    params: dict | tuple[dict, dict], params_limit: dict | tuple[dict, dict] = None
) -> dict | tuple[dict, dict]:
    """Transform unbounded parameters to the bounded space with the sigmoid function."""
    if params_limit is None:
        return params

    def _transform(p, p_limit):
        if p_limit is None:
            return p
        transformed_params = {}
        for key, val in p.items():
            if key in p_limit:
                lower, upper = p_limit[key]
                transformed_params[key] = lower + (upper - lower) * jax.nn.sigmoid(val)
            else:
                transformed_params[key] = val
        return transformed_params

    if isinstance(params, tuple) and isinstance(params_limit, tuple):
        return _transform(params[0], params_limit[0]), _transform(params[1], params_limit[1])
    elif isinstance(params, dict) and isinstance(params_limit, dict):
        return _transform(params, params_limit)
    else:
        raise TypeError("Invalid type for params and params_limit")


def _transform_bound_to_unbound(
    params: dict | tuple[dict, dict], params_limit: dict | tuple[dict, dict] = None
) -> dict | tuple[dict, dict]:
    """Transform bounded parameters to the unbounded space with the inverse sigmoid function."""
    if params_limit is None:
        return params

    def _transform(p, p_limit):
        if p_limit is None:
            return p
        transformed_params = {}
        try:
            for key, val in p.items():
                if key in p_limit:
                    lower, upper = p_limit[key]
                    transformed_params[key] = jnp.log((val - lower) / (upper - val))
                else:
                    transformed_params[key] = val
        except Exception as e:
            raise ValueError(
                f"Invalid {key} to transform bound to unbound: value = {val}, upper/lower = {upper}/{lower}:" + str(e)
            )
        return transformed_params

    if isinstance(params, tuple) and isinstance(params_limit, tuple):
        return _transform(params[0], params_limit[0]), _transform(params[1], params_limit[1])
    elif isinstance(params, dict) and isinstance(params_limit, dict):
        return _transform(params, params_limit)
    else:
        raise TypeError("Invalid type for params and params_limit")
