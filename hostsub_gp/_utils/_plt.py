# hostsub_gp/_plt_config.py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.axes import Axes

from typing import Callable, Optional

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "font.size": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    }
)


def show_and_save(f: Callable) -> Callable:
    """
    A decorator that adds common plotting options (show and save) to any plotting function.

    Parameters
    ----------
    f : Callable
        The plotting function to be decorated

    Returns
    -------
    Callable
        The wrapped function with additional show and save parameters
    """
    from functools import wraps
    from ._msgs import msgs

    @wraps(f)
    def wrapper(*args, show: bool = True, save: Optional[str] = None, **kwargs):
        # Call the original plotting function
        result = f(*args, **kwargs)

        # Handle saving if a path is provided
        if save:
            plt.savefig(save)
            func_name = [i for i in f.__name__.split("_") if len(i) > 0]
            msgs.info(f"Saved the {' '.join(func_name[1:])} plot to {save}")

        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()

        return result

    return wrapper
