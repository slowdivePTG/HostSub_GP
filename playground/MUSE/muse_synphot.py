import numpy as np
import glob
import os

# import jax.numpy as jnp
# import jax

# jax.config.update("jax_enable_x64", True)

from astropy.io import fits

from hostsub_gp._utils import plt, msgs

from numpy.typing import NDArray

import argparse

HDR_FILE = "./HEADER.toml"
GALAXY_TYPES = ["elliptical", "spiral"]
MODEL_TYPES = ["raw", "bad_phot", "bad_phot_match", "bad_spec", "bad_spec_match"]

WV_EFF = dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60)
FLTS = "riz"

# Median seeing in grz for PanSTARRS (Magnier+2020)
BAD_SEEING = dict(r=1.19, i=1.11, z=1.07)

parser = argparse.ArgumentParser(
    description="Test the HostSub_GP package on MUSE data cube."
)
parser.add_argument(
    "galaxy_type",
    type=str,
    choices=GALAXY_TYPES,
    help="Type of the galaxy (also used as the directory name).",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="raw",
    choices=MODEL_TYPES,
    help="Type of the model to run.",
)
parser.add_argument(
    "--overwrite",
    "-o",
    default=False,
    action="store_true",
    help="Overwrite the output files.",
)
parser.add_argument(
    "--clean",
    "-c",
    default=False,
    action="store_true",
    help="Remove all previous results.",
)
parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
parser.add_argument(
    "--kernel_width",
    type=float,
    default=1.0,
    help="Gaussian kernel width to downgrade the spatial resolution of the data.",
)

# Define args as a global variable
args = parser.parse_args()

PATH = f"{args.galaxy_type}/{args.model_type}/QA"


def get_synthetic_flux() -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Compute (or load) synthetic flux for a given filter."""
    from scipy.ndimage import gaussian_filter

    # Load the MUSE data cube
    data_cube_file = glob.glob(f"{args.galaxy_type}/*.fits")
    if len(data_cube_file) == 0:
        raise FileNotFoundError(
            f"No fits file found in the directory {args.galaxy_type}"
        )
    elif len(data_cube_file) > 1:
        raise FileNotFoundError(
            f"Multiple fits files found in the directory {args.galaxy_type}"
        )

    msgs.info(f"Loading the data cube from {data_cube_file[0]}...")
    hdul = fits.open(data_cube_file[0])
    dat = hdul[1].data
    dat_var = hdul[2].data

    # Load the wavelength and bin along the spectral axis
    wv = hdul[1].header["CRVAL3"] + hdul[1].header["CD3_3"] * np.arange(
        hdul[1].header["NAXIS3"]
    )

    syn_flux = {}
    syn_flux_var = {}

    for flt in FLTS:
        thpt = np.loadtxt(f"./PS1_filters/PAN-STARRS_PS1.{flt}.dat")
        wv_low = thpt[thpt[:, 1] >= 0.5 * thpt[:, 1].max(), 0][0]
        wv_high = thpt[thpt[:, 1] >= 0.5 * thpt[:, 1].max(), 0][-1]
        msgs.info(
            f"Computed wavelength range for the {flt}-band filter: {wv_low:.1f} - {wv_high:.1f} Angstrom"
        )

        flt_file = data_cube_file[0].split("/")[-1].replace(".fits", f".PS1_{flt}.npy")
        flt_path = f"{args.galaxy_type}/{args.model_type}/{flt_file}"

        # Check if the synthetic flux file already exists
        msgs.info(f"Computing synthetic photometry for the {flt}-band filter...")
        cube = np.where(np.isfinite(dat), dat, 0)
        cube_var = np.where(np.isfinite(dat_var), dat_var, 0)
        band_throuput = np.interp(wv, thpt[:, 0], thpt[:, 1])
        syn_flux[flt] = np.trapezoid(
            cube * band_throuput[:, np.newaxis, np.newaxis], wv, axis=0
        )
        syn_flux_var[flt] = np.trapezoid(
            cube_var * band_throuput[:, np.newaxis, np.newaxis] ** 2, wv, axis=0
        )
        msgs.info(
            f"Saving synthetic photometry for the {flt}-band filter to {flt_file}..."
        )
        # syn_flux_var[flt][syn_flux[flt] <= 0] = np.nan
        # syn_flux[flt][syn_flux[flt] <= 0] = np.nan
        np.save(flt_path, [syn_flux[flt], syn_flux_var[flt]])

        # If we want to downgrade the spatial resolution
        if "bad_phot" in args.model_type:
            msgs.info(
                f"Downgrading the spatial resolution of the {flt}-band filter by a Gaussian kernel with width {args.kernel_width} pixels..."
            )

    return syn_flux, syn_flux_var
