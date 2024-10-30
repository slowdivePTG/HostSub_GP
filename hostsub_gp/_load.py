# hostsub_gp/_load.py

__all__ = ["load_image", "load_pypeit"]


from astropy.io import fits
import numpy as np

from typing import Tuple

### Load image for the host profile
wv_eff_dict = dict(ps1=dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60))


def load_image(file_path: str, camera="ps1") -> Tuple[any, any]:
    """
    Load image data and header from fits file
    """
    if camera in ["ps1", "PS1", "pan-starrs", "Pan-STARRS"]:
        with fits.open(file_path) as hdulist:
            data = hdulist[0].data
            header = hdulist[0].header
    else:
        raise ValueError("Other telescopes/cameras not supported yet")
    return data, header


### Load pypeit 2d spectrum
def load_pypeit(file_path: str) -> np.ndarray:
    """
    Load 2d spectrum from pypeit fits file
    """
    with fits.open(file_path) as hdulist:
        data = hdulist[0].data
    return data
