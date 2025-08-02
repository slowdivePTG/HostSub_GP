import numpy as np
import glob
import os

# import jax.numpy as jnp
# import jax

# jax.config.update("jax_enable_x64", True)

from astropy.io import fits
from astropy.stats import mad_std

from hostsub_gp import SpecModel, SpecData
from hostsub_gp._utils import plt, msgs

from numpy.typing import NDArray

import argparse

HDR_FILE = "./HEADER.toml"
GALAXY_TYPES = ["spiral", "spiral_2", "spiral_3"]
MODEL_TYPES = ["raw", "bad_phot", "bad_phot_match", "bad_spec", "bad_spec_match"]

WV_EFF = dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60)
FLTS = "riz"

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


def decode_toml() -> dict[str, dict]:
    """Decode a TOML string into a dictionary."""
    import tomllib

    with open(HDR_FILE, "rb") as f:
        hdr = tomllib.load(f)

    assert args.galaxy_type in hdr["basic"], (
        f"Galaxy type {args.galaxy_type} not found in {HDR_FILE}"
    )

    # Configuration for host subtraction modeling
    spec_model_gal_cfg = hdr["spec_model"].get(args.galaxy_type, None)
    assert spec_model_gal_cfg is not None, (
        f"Spec model configuration for {args.galaxy_type} not found in {HDR_FILE}"
    )

    spec_model_cfg = {
        **{k: v for k, v in hdr["basic"].items() if not k in GALAXY_TYPES},
        **spec_model_gal_cfg,
    }

    # Configuration for how the slits are randomly placed
    slit_range_cfg = hdr["slit_range"].get(args.galaxy_type, None)
    assert slit_range_cfg is not None, (
        f"Slit range configuration for {args.galaxy_type} not found in {HDR_FILE}"
    )

    # Basic information of the galaxy
    galaxy_cfg = hdr["basic"].get(args.galaxy_type, None)
    assert galaxy_cfg is not None, (
        f"Galaxy configuration for {args.galaxy_type} not found in {HDR_FILE}"
    )

    return {
        "spec_model": spec_model_cfg,
        "slit_range": slit_range_cfg,
        "galaxy": galaxy_cfg,
    }


def pack_2d_spectrum(
    dat: NDArray,
    dat_var: NDArray,
    ra: NDArray,
    dec: NDArray,
    wv: NDArray,
    targetid: str,
    *,
    # For SpecData & SpecModel
    row: int,
    col: int,
    mask_offset_pix: int,
    pixel_scale: float,
    slit_len: float,
    slit_wid: float,
    position_angle: float,
    spat_resln: float,
    spec_resln: float,
    spec_range: tuple[float, float],
    host_wid: float,
    mask_wid: float,
    sky_region: tuple[float, float],
) -> "SpecModel":
    """Pack the 2D spectrum into a SpecModel object."""
    mask_offset = mask_offset_pix * pixel_scale
    center_ra = ra[col]
    center_dec = dec[row]
    ra_offset = (ra - center_ra) * 3600
    dec_offset = (dec - center_dec) * 3600

    # Ensure the slit length does not exceed the FoV
    slit_len = min((max(ra_offset) - min(ra_offset)) * 0.8, slit_len)

    # Generate the synthetic 2D spectrum
    # Check if ra_offset is sorted in ascending or descending order
    ra_order = 1 if ra_offset[-1] > ra_offset[0] else -1
    ra_mask = np.abs(ra_offset) <= slit_len / 2
    
    flux_rect = np.nanmean(
        dat[:, row - 2 : row + 3, ra_mask].reshape(
            dat.shape[0], 5, -1, 1
        ),
        axis=(1, 3),
    )
    flux_ivar_rect = np.nanmean(
        dat_var[:, row - 2 : row + 3, ra_mask].reshape(
            dat.shape[0], 5, -1, 1
        ),
        axis=(1, 3),
    ) ** -1 * (5 * 1)

    # Make sure the spatial coordinates and flux values are properly sorted
    if ra_order == -1:
        flux_rect = flux_rect[:, ::-1]
        flux_ivar_rect = flux_ivar_rect[:, ::-1]
    
    spec_data = SpecData(
        pixel_scale=pixel_scale,
        center_ra=center_ra,
        center_dec=center_dec,
        slit_wid=slit_wid,
        position_angle=position_angle,
        spat_resln=spat_resln,
        spec_resln=spec_resln,
        spat_rect=ra_offset[ra_mask][::ra_order],  # Ensure ascending order for spat_rect
        spec_rect=wv,
        flux_rect=np.asarray(flux_rect, dtype=float).T,
        flux_ivar_rect=np.asarray(flux_ivar_rect, dtype=float).T,
    )

    sky_left = min(sky_region[0], -host_wid / 2 + mask_offset)
    sky_right = max(sky_region[1], host_wid / 2 + mask_offset)

    spec_model = spec_data.to_SpecModel(
        slit_len=slit_len,
        spec_range=spec_range,
        host_region=[-host_wid / 2, host_wid / 2],
        mask_wid=mask_wid,
        mask_offset=mask_offset,
        sky_region=(sky_left, sky_right),
    )

    spec_model.ra_offset = ra_offset
    spec_model.dec_offset = dec_offset

    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.5])
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1:])]

    # Plot the data cube
    dat_im_stack = np.nanmean(dat, axis=0)
    ax[0].imshow(
        dat_im_stack,
        origin="lower",
        cmap="grey",
        vmin=np.nanpercentile(dat_im_stack, 1),
        vmax=np.nanpercentile(dat_im_stack, 95),
    )
    ## The slit
    ax[0].plot(
        [
            col - slit_len / 2 / pixel_scale,
            col + slit_len / 2 / pixel_scale,
        ],
        [row, row],
        color="red",
    )
    ## The center of the slit
    ax[0].scatter(col + mask_offset_pix, row, color="red", s=100, marker="+")
    ax[0].set_xlabel("RA (pixels)")
    ax[0].set_ylabel("Dec (pixels)")

    # Plot the mock 2D spectrum
    ## Slit width = 5 pixels = 1 arcsec
    ## Create a 2D spectrum with the correct spatial orientation
    spectrum_2d = np.mean(dat[:, row - 2 : row + 3, :].reshape(dat.shape[0], 5, -1), axis=1).T
    
    # If RA is in descending order, we need to flip the image to maintain correct spatial orientation
    if ra_order == -1:
        spectrum_2d = spectrum_2d[::-1, :]
        extent = (wv[0], wv[-1], ra_offset[-1], ra_offset[0])  # Flip the y-axis extent
    else:
        extent = (wv[0], wv[-1], ra_offset[0], ra_offset[-1])
        
    ax[1].imshow(
        spectrum_2d,
        origin="lower",
        cmap="grey",
        vmin=np.nanpercentile(dat[:, row, :], 1),
        vmax=np.nanpercentile(dat[:, row, :], 95),
        extent=extent,
    )
    ax[1].set_aspect("auto")
    ax[1].set_ylim(-slit_len / 2, slit_len / 2)
    ax[1].axhline(mask_offset, color="red")
    ax[1].set_xlabel("Wavelength (Angstrom)")
    ax[1].set_ylabel("RA offset (arcsec)")

    plt.savefig(f"{PATH}/{targetid}_image.pdf")
    plt.close()

    return spec_model


def model_host_prior(
    spec_model: "SpecModel",
    row: int,
    col: int,
    mask_offset_pix: int,
    syn_flux: dict,
    syn_flux_var: dict,
    slit_len: float,
    targetid: str,
) -> "SpecModel":
    """Model the host prior."""

    mask_offset = mask_offset_pix * spec_model.pixel_scale

    counts_slit = []
    counts_err_slit = []
    spat_slit = []

    # Check if ra_offset is sorted in ascending or descending order
    ra_order = 1 if spec_model.ra_offset[-1] > spec_model.ra_offset[0] else -1
    on_slit = np.abs(spec_model.ra_offset - mask_offset) <= slit_len / 2
    
    for flt in FLTS:
        # Ensure consistent ordering with how spat_rect was created in pack_2d_spectrum
        spat_slit.append(spec_model.ra_offset[on_slit][::ra_order])
        counts_slit.append(
            np.nanmean(syn_flux[flt][row - 2 : row + 3, :], axis=0)[on_slit][::ra_order]
        )
        counts_err_slit.append(
            np.nanmean(syn_flux_var[flt][row - 2 : row + 3, :], axis=0)[on_slit][::ra_order] ** 0.5
            / 5**0.5
        )

    spec_model.build_host_prior(
        filters=FLTS,
        from_archival=False,
        wv_eff=[WV_EFF[flt] for flt in FLTS],
        spat_slit=spat_slit,
        counts_slit=counts_slit,
        counts_err_slit=counts_err_slit,
        save=f"{PATH}/{targetid}_host_prior.pdf",
    )

    return spec_model


def plot_QA(
    spec_model: "SpecModel",
    targetid: str,
) -> None:
    """Plot the QA figures."""

    spec_model._plot_host_profile_pred(save=f"{PATH}/{targetid}_host_profile_pred.pdf")
    msgs.info(
        f"Saving the posterior of the host profiles to {PATH}/{targetid}_host_profile_pred.pdf"
    )

    # Raw, model, and residual
    spec_model._plot_pred(save=f"{PATH}/{targetid}_pred.pdf")
    msgs.info(f"Saving the raw, model, and residual to {PATH}/{targetid}_pred.pdf")

    # Extract the science spectrum
    spec_model.extract_sci(save=f"{PATH}/{targetid}_sci.pdf")
    msgs.info(f"Saving the science spectrum to {PATH}/{targetid}_sci.pdf")

    # Save the 1D spectra
    dat = np.array(
        [
            spec_model.spec,
            spec_model.f_sci_pred_1d.y,
            spec_model.f_sci_pred_1d.yerr,
            spec_model.f_sci_classic_1d.y,
            spec_model.f_sci_classic_1d.yerr,
        ]
    ).T
    np.savetxt(f"{PATH}/{targetid}_sci.dat", dat)


def get_synthetic_flux(
    dat: NDArray, dat_var: NDArray, wv: NDArray, overwrite: bool = False
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Compute (or load) synthetic flux for a given filter."""
    from scipy.ndimage import gaussian_filter

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
        if overwrite or not os.path.exists(flt_path):
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
                    f"Downgrading the spatial resolution of the {flt}-band filter by a Gaussian kernel with width {args.kernel_width} arcsec..."
                )
                kernel_size = (
                    args.kernel_width / 2.355 / spec_model.pixel_scale
                )  # Convert FWHM to sigma in pixels
                syn_flux[flt] = gaussian_filter(
                    syn_flux[flt], sigma=kernel_size, mode="nearest"
                )
                syn_flux_var[flt] = gaussian_filter(
                    syn_flux_var[flt], sigma=kernel_size, mode="nearest"
                )

        # If the file exists, load it
        else:
            msgs.info(f"Loading synthetic photometry for the {flt}-band filter...")
            syn_flux[flt], syn_flux_var[flt] = np.load(flt_path)

    return syn_flux, syn_flux_var


def range_to_random_ints(range_tuple: tuple, n: int) -> NDArray:
    """Convert a range string to a list of random integers."""
    if range_tuple[1] > range_tuple[0]:
        return np.random.randint(*range_tuple, size=n)
    elif range_tuple[1] < range_tuple[0]:
        return np.random.randint(*range_tuple[::-1], size=n)
    else:
        return np.ones(n) * range_tuple[0]


if __name__ == "__main__":
    if args.clean:
        if os.path.exists(PATH):
            msgs.info(f"Removing the directory {PATH} and its contents.")
            for file in glob.glob(f"{PATH}/*"):
                os.remove(file)
            os.rmdir(PATH)
        else:
            msgs.info(f"The directory {PATH} does not exist, nothing to clean.")
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        msgs.info(f"Created directory {PATH} for output files.")

    # Load the configuration from the TOML file
    hdr = decode_toml()
    spec_model_cfg = hdr["spec_model"]
    slit_range_cfg = hdr["slit_range"]
    galaxy_cfg = hdr["galaxy"]

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

    # Load the WCS
    ref_ra = hdul[1].header["CRVAL1"]
    ref_dec = hdul[1].header["CRVAL2"]

    ra = ref_ra + hdul[1].header["CD1_1"] * (
        np.arange(hdul[1].header["NAXIS1"]) - (hdul[1].header["CRPIX1"] - 1)
    )
    dec = ref_dec + hdul[1].header["CD2_2"] * (
        np.arange(hdul[1].header["NAXIS2"]) - (hdul[1].header["CRPIX2"] - 1)
    )

    np.random.seed(42)
    has_identical_trials = True
    while has_identical_trials:
        col = range_to_random_ints(slit_range_cfg["col"], args.n_trials)
        row = range_to_random_ints(slit_range_cfg["row"], args.n_trials)
        mask_offset_pix = range_to_random_ints(
            slit_range_cfg["mask_offset"], args.n_trials
        )
        # Check if the trials are identical
        has_identical_trials = len(set(zip(col, row, mask_offset_pix))) < args.n_trials
    msgs.info(
        f"Randomly selected {args.n_trials} trials with unique (col, row, mask_offset_pix) pairs."
    )

    # Seeing matching
    dseeing_opt_list = []

    for n_trial in range(args.n_trials):
        targetid = (
            f"row_{row[n_trial]}_col_{col[n_trial]}_offset_{mask_offset_pix[n_trial]}"
        )
        msgs.info(f"Running trial {n_trial + 1}/{args.n_trials}...")
        msgs.info(f"Target ID: {targetid}")
        # Pack the 2D spectrum
        spec_model = pack_2d_spectrum(
            dat,
            dat_var,
            ra,
            dec,
            wv,
            targetid,
            row=row[n_trial],
            col=col[n_trial],
            mask_offset_pix=mask_offset_pix[n_trial],
            **spec_model_cfg,
        )

        syn_flux, syn_flux_var = get_synthetic_flux(
            dat=dat, dat_var=dat_var, wv=wv, overwrite=args.overwrite and n_trial == 0
        )

        # Model the host prior
        spec_model = model_host_prior(
            spec_model,
            row=row[n_trial],
            col=col[n_trial],
            mask_offset_pix=mask_offset_pix[n_trial],
            syn_flux=syn_flux,
            syn_flux_var=syn_flux_var,
            slit_len=spec_model_cfg["slit_len"],
            targetid=targetid,
        )

        spec_model.construct_spec_wrapper(
            f_obs=spec_model.f_obs,
            batch_2d=(2, 256) if args.galaxy_type == "spiral" else (2, 128),
            host_emission_cfg={
                "find_host_emission": args.galaxy_type == "spiral",
                "z": galaxy_cfg["z"],
                "z_err": 0.0001,
                "p_value": 0.05,
            },
            sigma_clip=None,
            save=f"{PATH}/{targetid}_raw.pdf",
        )

        if "match" in args.model_type:
            dseeing_opt, alpha_opt = spec_model.update_seeing(dseeing=None, dseeing_upper=1.5)
            dseeing_opt_list.append(dseeing_opt)

            dseeing_wv = (
                dseeing_opt
                / spec_model.pixel_scale
                * (spec_model.spec / spec_model.spec.mean()) ** (-alpha_opt)
            )
            spec_model.construct_spec_wrapper(
                f_obs=spec_model.f_obs.convolve(dseeing_wv),
                batch_2d=(1, 128),
                host_emission_cfg={
                    "find_host_emission": True,
                    "z": galaxy_cfg["z"],
                    "z_err": 0.001,
                    "p_value": 0.05,
                },
                sigma_clip=None,
                save=f"{PATH}/{targetid}_conv.pdf",
            )

        # Get the initial parameters
        log_amp_est = np.log10(((spec_model.f_host_1d.y) ** 2).max())
        mean_est = np.nanmean(spec_model.f_host_1d.y)
        params_init_1d = dict(
            log_amp=(
                log_amp_est,  # ExpSquared: Logarithm of the maximum squared value of the 1D spectrum
                log_amp_est - 2,  # Matern: Somewhat smaller
            ),
            log_scale=(
                2,  # ExpSquared: 100 Angstrom
                np.log10(
                    spec_model.spec_resln / 2.355
                ),  # Matern: Spectral resolution / 2.355
            ),
            mean=mean_est,  # Mean of the 1D spectrum
        )
        params_init_2d = dict(
            log_amp=-5.0,
            log_scale=(
                np.log10(spec_model.spat_resln),  # Spatial scale ~ seeing
                4,  # Spectral scale ~ 10^4 Angstrom
            ),
            mean=0.0,
            log_amp_line=1.0,  # Covariance within the host lines = covariance outside the host lines
            scale_line=spec_model.spec_resln
            / 2,  # Radius of the host lines: Half of the FWHM of the spectral resolution
        )
        params_init = (params_init_1d, params_init_2d)

        # Set the limits for the parameters
        params_limit_1d = dict(
            log_scale=np.log10(
                [
                    # log range of the slow varying component
                    [1e3, np.inf],
                    # log range of the fast varying component
                    # typical scale = spectral resolution
                    [spec_model.spec_resln / 2.355, spec_model.spec_resln],
                ]
            ).T
        )
        params_limit_2d = dict(
            log_scale=np.log10(
                [
                    # log range of the spatial component
                    # typical scale = spatial resolution
                    [spec_model.spat_resln / 2.355, np.inf],
                    # log range of the spectral component
                    # typical scale = spectral resolution
                    [spec_model.spec_resln / 2.355, np.inf],
                ]
            ).T,
            mean=np.array([-1e-1, 1e-1]),  # Mean of the host profile
            log_amp_line=np.array(
                [0, np.inf]
            ),  # Logarithm of the amplitude of the host lines
            scale_line=np.array(
                [spec_model.spec_resln / 2.355, spec_model.spec_resln]
            ),  # Scale of the host lines
        )
        params_limit = (params_limit_1d, params_limit_2d)

        # Prior and posterior of the host profiles
        spec_model._plot_host_profile_prior(
            save=f"{PATH}/{targetid}_host_profile_prior.pdf"
        )
        msgs.info(
            f"Saving the prior of the host profiles to {PATH}/{targetid}_host_profile_prior.pdf"
        )

        # Model the host
        spec_model.model_host(
            params_init=params_init,
            params_limit=params_limit,
            optimization=True,
            # optimization_kwargs={"maxiter": 1000, "tol": 1e-4},
        )

        # QA plots
        plot_QA(spec_model, targetid)

    if "match" in args.model_type:
        dseeing_opt_list = np.array(dseeing_opt_list)
        np.savetxt(f"{PATH}/dseeing_opt.dat", dseeing_opt_list)
