# hostsub_gp/_gp.py

__all__ = [
    "_transform_unbound_to_bound",
    "_transform_bound_to_unbound",
    "_init_params",
    "_check_params",
    "_split_params",
    "_print_params",
]

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _transform_unbound_to_bound(
    params: dict | tuple[dict, dict], params_limit: dict | tuple[dict, dict] = None
) -> dict | tuple[dict, dict]:
    """Transform unbounded parameters to the bounded space with the sigmoid function."""
    if params_limit is None:
        return params

    def _transform(p, p_limit):
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


def _init_params(params: dict) -> dict:
    """Initialize the parameters with appropriate typing."""
    if params is None:
        return None

    _check_params(params, required_all=True)

    def ensure_scalar_or_array(x):
        x = jnp.asarray(x)
        if x.size > 1:
            return jnp.asarray(x, dtype=jnp.float64)
        elif x.ndim == 0:
            return jnp.asarray(x, dtype=jnp.float64)
        else:
            return jnp.asarray(x[0], dtype=jnp.float64)
        # else:
        #     return jnp.asarray(x.item(), dtype=jnp.float64)

    # Kernel parameters (Array[float64] - can be a scalar or ndarray)
    log_amp = ensure_scalar_or_array(params.get("log_amp"))
    log_scale = ensure_scalar_or_array(params.get("log_scale"))

    if log_amp.size > 1:
        if log_amp.size != log_scale.shape[-1]:
            raise ValueError("Invalid shape of kernel (amplitude and scale) parameters")

    # GP mean (float64)
    mean = ensure_scalar_or_array(params.get("mean"))

    # Jitter (float64)
    if params.get("log_jitter") is None:
        log_jitter = jnp.ones_like(mean, dtype=jnp.float64) * -6.0
    else:
        log_jitter = ensure_scalar_or_array(params.get("log_jitter"))

    return dict(log_amp=log_amp, log_scale=log_scale, log_jitter=log_jitter, mean=mean)


def _check_params(params: dict, required_all: bool = False):
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


def _print_params(params: dict | tuple[dict, dict]):
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
        _check_params(params, required_all=True)
        # only 1D or 2D parameters
        _print_param(params, "log_amp", "Amp")
        _print_param(params, "log_scale", "Scale")
        if "log_jitter" in params:
            _print_param(params, "log_jitter", "Jitter")
        _print_param(params, "mean", "Mean")

    except:
        # both 1D and 2D parameters
        if isinstance(params, dict):
            params = _split_params(params)
        for i in range(2):
            try:
                _check_params(params[i], required_all=True)
            except ValueError as e:
                raise ValueError("Printing parameters: " + str(e))

        for i in range(2):
            print(f"{i+1}D parameters:")
            _print_param(params[i], "log_amp", "Amp")
            _print_param(params[i], "log_scale", "Scale")
            if "log_jitter" in params[i]:
                _print_param(params[i], "log_jitter", "Jitter")
            _print_param(params[i], "mean", "Mean")
