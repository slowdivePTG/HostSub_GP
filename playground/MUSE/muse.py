import numpy as np
from hostsub_gp._plt import plt
import glob
import os

import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from hostsub_gp._msgs import msgs

from hostsub_gp import SpecModel, SpecData, HostProfile
from hostsub_gp.spectrum_model import SpecWrapper

from numpy.typing import ArrayLike, NDArray

import argparse

parser = argparse.ArgumentParser(description="Test the HostSub_GP package on MUSE data cube.")
parser.add_argument("galaxy", type=str, help="Name of the galaxy (used as the directory name).")
parser.add_argument("-z", type=float, default=0.0, help="Redshift of the galaxy.")
parser.add_argument("--overwrite", "-o", default=False, action="store_true", help="Overwrite the output files.")
parser.add_argument(
    "--row_range", type=str, default="180:220", help="Range of possible row (RA) indices for the slit center."
)
parser.add_argument(
    "--col_range", type=str, default="180:220", help="Range of possible column (Dec) indices for the slit center."
)
parser.add_argument("--mask_offset_range", type=str, default="0:0", help="Range of offsets from the slit center.")
parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
args = parser.parse_args()


def pack_2d_spectrum(
    dat: NDArray,
    dat_var: NDArray,
    ra: ArrayLike,
    dec: ArrayLike,
    wv: ArrayLike,
    row: int,
    col: int,
    mask_offset_pix: int,
    slit_len: float,
    pixel_scale: float,
    z: float,
    sky_region: tuple[float, float],
    targetid: str,
) -> "SpecModel":
    """Pack the 2D spectrum into a SpecModel object."""
    mask_offset = mask_offset_pix * pixel_scale
    center_ra = ra[col]
    center_dec = dec[row]
    ra_offset = (ra - center_ra) * 3600


    flux_rect = np.nanmean(
        dat[:, row - 2 : row + 3, np.abs(ra_offset) <= slit_len / 2].reshape((dat.shape[0]) // 2, 2, 5, -1),
        axis=(1, 2),
    )[:, ::-1]
    flux_ivar_rect = np.nansum(
        dat_var[:, row - 2 : row + 3, np.abs(ra_offset) <= slit_len / 2].reshape((dat.shape[0]) // 2, 2, 5, -1),
        axis=(1, 2),
    )[:, ::-1] ** -1 * (
        2 * 5  # 2 pixels in the spatial direction and 5 pixels in the spectral direction
    )

    spec_data = SpecData(
        pixel_scale=pixel_scale,
        center_ra=center_ra,
        center_dec=center_dec,
        slit_wid=1,
        position_angle=90,
        spat_resln=1.5,  # Seeing FWHM = 0.9 arcsec
        spec_resln=2.7,  # Spectral resolution of MUSE
        spat_rect=ra_offset[np.abs(ra_offset) <= slit_len / 2][::-1],
        spec_rect=wv,
        flux_rect=np.asarray(flux_rect, dtype=float).T,
        flux_ivar_rect=np.asarray(flux_ivar_rect, dtype=float).T,
    )

    spec_model = spec_data.to_SpecModel(
        slit_len=slit_len,
        spec_range=(5500, 9200),  # Edge (throughput = half maximum) of the PS1 r and z filters
        host_wid=8,
        mask_wid=1.0,
        mask_offset=mask_offset,
        sky_region=sky_region,
        batch_2d=(2, 128),
        host_emission_cfg={"find_host_emission": True, "z": z, "z_err": 0.001, "p_value": 0.05},
        show=False,
        save=f"{args.galaxy}/QA/{targetid}.pdf",
    )

    spec_model.ra_offset = ra_offset

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
    ax[0].scatter(col - mask_offset_pix, row, color="red", s=100, marker="+")
    ax[0].set_xlabel("RA (pixels)")
    ax[0].set_ylabel("Dec (pixels)")

    # Plot the mock 2D spectrum
    ## Slit width = 5 pixels = 1 arcsec
    ax[1].imshow(
        np.mean(dat[:, row - 2 : row + 3, :].reshape(dat.shape[0] // 2, 2, 5, -1), axis=(1, 2)).T,
        origin="lower",
        cmap="grey",
        vmin=np.nanpercentile(dat[:, row, :], 1),
        vmax=np.nanpercentile(dat[:, row, :], 95),
        extent=(wv[0], wv[-1], ra_offset[0], ra_offset[-1]),
    )
    ax[1].set_aspect("auto")
    ax[1].set_ylim(-slit_len / 2, slit_len / 2)
    ax[1].axhline(mask_offset, color="red")
    ax[1].set_xlabel("Wavelength (Angstrom)")
    ax[1].set_ylabel("RA offset (arcsec)")

    plt.savefig(f"{args.galaxy}/QA/{targetid}_image.pdf")
    plt.close()

    return spec_model


def model_host_prior(
    spec_model: "SpecModel", row: int, col: int, mask_offset_pix: int, syn_flux: dict, slit_len: float, targetid: str
) -> "SpecModel":
    """Model the host prior."""

    mask_offset = mask_offset_pix * spec_model.pixel_scale

    wv_eff_dict = dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60)
    flts = "riz"

    counts_slit = []
    counts_err_slit = []
    spat_slit = []

    on_slit = np.abs(spec_model.ra_offset - mask_offset) <= slit_len / 2
    for flt in flts:
        counts_slit.append(np.nanmean(syn_flux[flt][row - 2 : row + 3], axis=0)[on_slit][::-1])
        # counts_err_slit.append(np.nanstd(syn_flux[flt][row - 2 : row + 3], axis=0)[on_slit][::-1] / 5**0.5)
        counts_slit_left_off = np.nanmean(syn_flux[flt][row - 3 : row + 2], axis=0)[on_slit][::-1]
        counts_slit_right_off = np.nanmean(syn_flux[flt][row - 1 : row + 4], axis=0)[on_slit][::-1]
        counts_err_slit.append(
            (np.abs(counts_slit_right_off - counts_slit[-1]) + np.abs(counts_slit_left_off - counts_slit[-1])) / 4
        )
        spat_slit.append(spec_model.ra_offset[on_slit][::-1])

    host_prof = HostProfile(
        flts=flts,
        wv_eff=[wv_eff_dict["r"], wv_eff_dict["i"], wv_eff_dict["z"]],
        spat_slit=spat_slit,
        counts_slit=counts_slit,
        counts_err_slit=counts_err_slit,
        spec_model=spec_model,
    )

    host_flux_prior = host_prof.model_host_profile_prior(show=False, save=f"{args.galaxy}/QA/{targetid}_host_prior.pdf")
    scale = lambda X: jnp.interp(
        X[:, 1],
        spec_model.spec,
        jnp.sum(host_flux_prior(spec_model.f_host.X)[0].reshape(spec_model.f_host.shape), axis=0),
    )

    def predict(X):
        prior, prior_var = host_flux_prior(X)
        return prior / scale(X), prior_var**0.5 / scale(X)

    spec_model.host_flux_prior = predict

    # The orignal 2D data in the host region
    prior_host, prior_host_std = spec_model.host_flux_prior(spec_model.f_host.X)
    spec_model.f_host_prior = SpecWrapper(
        points=(spec_model.f_host.spat, spec_model.f_host.spec),
        values=prior_host.reshape(spec_model.f_host.shape),
        values_err=prior_host_std.reshape(spec_model.f_host.shape),
    )

    # The batched 2D data
    prior_batch, prior_batch_std = spec_model.host_flux_prior(spec_model.f_batch_2d.X)
    spec_model.f_batch_prior = SpecWrapper(
        points=(spec_model.f_batch_2d.spat, spec_model.f_batch_2d.spec),
        values=prior_batch.reshape(spec_model.f_batch_2d.shape),
        values_err=prior_batch_std.reshape(spec_model.f_batch_2d.shape),
    )

    # Batched 2D data (host region)
    prior_host_batch, prior_host_batch_std = spec_model.host_flux_prior(spec_model.f_host_batch_2d.X)
    spec_model.f_host_batch_prior = SpecWrapper(
        points=(spec_model.f_host_batch_2d.spat, spec_model.f_host_batch_2d.spec),
        values=prior_host_batch.reshape(spec_model.f_host_batch_2d.shape),
        values_err=prior_host_batch_std.reshape(spec_model.f_host_batch_2d.shape),
    )

    return spec_model


def plot_QA(spec_model: "SpecModel", targetid: str):
    """Plot the QA figures."""
    # Raw, model, and residual
    spec_model._plot_pred()
    plt.savefig(f"{args.galaxy}/QA/{targetid}_pred.pdf")
    plt.close()

    # Extract the science spectrum
    spec_model.extract_sci()
    plt.close()
    local_sky_left = (spec_model.spat < -spec_model.mask_wid / 2 + spec_model.mask_offset) & (
        spec_model.spat > -spec_model.mask_wid * 2 / 2 + spec_model.mask_offset
    )
    local_sky_right = (spec_model.spat > spec_model.mask_wid / 2 + spec_model.mask_offset) & (
        spec_model.spat < spec_model.mask_wid * 2 / 2 + spec_model.mask_offset
    )
    local_sky = local_sky_left | local_sky_right

    classic_pred = np.mean((spec_model.f_sky_sub.Y - np.mean(spec_model.f_sky_sub.Y[local_sky], axis=0))[spec_model.spat_filter["mask"]], axis=0)
    classic_pred_err = np.std(spec_model.f_sky_sub.Y[local_sky], axis=0) / np.sum(local_sky)**0.5

    plt.figure(figsize=(10, 5))
    plt.plot(spec_model.spec, spec_model.f_sci_pred_1d.y, label=r"$\mathrm{GP}$", zorder=2, color="#8c96c6")
    plt.plot(spec_model.spec, classic_pred, label=r"$\mathrm{classic}$", color="grey", zorder=1, lw=1)
    plt.axhline(0, color="black", ls="--", zorder=3, lw=3)
    plt.ylabel(r"$\mathrm{Prediction}$")
    plt.ylim(-10, 15)
    plt.xlabel(r"$\mathrm{Wavelength\,[\r{A}]}$")
    plt.legend()

    plt.savefig(f"{args.galaxy}/QA/{targetid}_sci.pdf")

    # Save the 1D spectra
    dat = np.array([spec_model.spec, spec_model.f_sci_pred_1d.y, spec_model.f_sci_pred_1d.yerr, classic_pred, classic_pred_err]).T
    np.savetxt(f"{args.galaxy}/QA/{targetid}_sci.dat", dat)

def range_to_random_ints(range_str: str, n: int) -> ArrayLike:
    """Convert a range string to a list of random integers."""
    range_tuple = tuple(map(int, range_str.split(":")))
    if range_tuple[1] > range_tuple[0]:
        return np.random.randint(*range_tuple, size=n)
    elif range_tuple[1] < range_tuple[0]:
        return np.random.randint(*range_tuple[::-1], size=n)
    else:
        return np.ones(n) * range_tuple[0]


if __name__ == "__main__":
    # Load the MUSE data cube
    data_cube_file = glob.glob(f"{args.galaxy}/*.fits")
    if len(data_cube_file) == 0:
        raise FileNotFoundError(f"No fits file found in the directory {args.galaxy}")
    elif len(data_cube_file) > 1:
        raise FileNotFoundError(f"Multiple fits files found in the directory {args.galaxy}")
    if not os.path.exists(f"{args.galaxy}/QA"):
        os.mkdir(f"{args.galaxy}/QA")

    msgs.info(f"Loading the data cube from {data_cube_file[0]}...")
    hdul = fits.open(data_cube_file[0])
    dat = hdul[1].data
    dat_var = hdul[2].data

    # Load the wavelength and bin along the spectral axis
    wv = hdul[1].header["CRVAL3"] + hdul[1].header["CD3_3"] * np.arange(hdul[1].header["NAXIS3"])
    wv_bin = np.mean(wv.reshape(-1, 2), axis=1)

    # Load the WCS
    ref_ra = hdul[1].header["CRVAL1"]
    ref_dec = hdul[1].header["CRVAL2"]

    ra = ref_ra + hdul[1].header["CD1_1"] * (np.arange(hdul[1].header["NAXIS1"]) - (hdul[1].header["CRPIX1"] - 1))
    dec = ref_dec + hdul[1].header["CD2_2"] * (np.arange(hdul[1].header["NAXIS2"]) - (hdul[1].header["CRPIX2"] - 1))

    pixel_scale = np.abs(hdul[1].header["CD1_1"]) * 3600

    # Compute (or load) the synthetic photometry
    wv_range = {}
    syn_flux = {}
    for flt in "riz":
        thpt = np.loadtxt(f"./PS1_filters/PAN-STARRS_PS1.{flt}.dat")
        wv_low = thpt[thpt[:, 1] >= 0.5 * thpt[:, 1].max(), 0][0]
        wv_high = thpt[thpt[:, 1] >= 0.5 * thpt[:, 1].max(), 0][-1]
        wv_range[flt] = (wv_low, wv_high)
        msgs.info(f"Computed wavelength range for the {flt}-band filter: {wv_low:.1f} - {wv_high:.1f} Angstrom")

        flt_file = data_cube_file[0].replace(".fits", f".PS1_{flt}.dat")
        if args.overwrite or not os.path.exists(flt_file):
            msgs.info(f"Computing synthetic photometry for the {flt}-band filter...")
            cube = np.where(np.isfinite(dat), dat, 0)
            band_throuput = np.interp(wv, thpt[:, 0], thpt[:, 1])
            flux_integrated = np.trapz(cube * band_throuput[:, np.newaxis, np.newaxis], wv, axis=0)
            syn_flux[flt] = flux_integrated
            msgs.info(f"Saving synthetic photometry for the {flt}-band filter to {flt_file}...")
            np.savetxt(flt_file, syn_flux[flt])
        else:
            msgs.info(f"Loading synthetic photometry for the {flt}-band filter...")
            syn_flux[flt] = np.loadtxt(flt_file)

    np.random.seed(42)
    col = range_to_random_ints(args.col_range, args.n_trials)
    row = range_to_random_ints(args.row_range, args.n_trials)
    mask_offset_pix = range_to_random_ints(args.mask_offset_range, args.n_trials)

    for i in range(args.n_trials):
        targetid = f"row_{row[i]}_col_{col[i]}_offset_{mask_offset_pix[i]}"
        msgs.info(f"Running trial {i + 1}/{args.n_trials}...")
        msgs.info(f"Target ID: {targetid}")
        # Pack the 2D spectrum
        spec_model = pack_2d_spectrum(
            dat,
            dat_var,
            ra,
            dec,
            wv_bin,
            row=row[i],
            col=col[i],
            mask_offset_pix=mask_offset_pix[i],
            slit_len=60,
            pixel_scale=pixel_scale,
            z=args.z,
            sky_region=(-15, None),
            targetid=targetid,
        )

        # Model the host prior
        spec_model = model_host_prior(
            spec_model, row=row[i], col=col[i], mask_offset_pix=mask_offset_pix[i], syn_flux=syn_flux, slit_len=60, targetid=targetid
        )

        # Get the initial parameters
        params_init_1d = None
        params_init_2d = {}
        params_init_2d["log_scale"] = np.log10([0.5, 5e3])
        params_init = [params_init_1d, params_init_2d]

        # Get limits for the parameters
        def _set_params_limit(params_limit_dict):
            """Integrate upper and lower limits of each parameter."""
            upper = {k.replace("_upper", ""): v for k, v in params_limit_dict.items() if "upper" in k}
            lower = {k.replace("_lower", ""): v for k, v in params_limit_dict.items() if "lower" in k}
            return {k: (lower[k], upper[k]) for k in lower}

        params_limit_1d = _set_params_limit({})
        params_limit_2d = _set_params_limit({})

        params_limit_1d["log_scale"] = params_limit_1d.get(
            "log_scale",
            np.array(
                [
                    # log range of the slow varying component
                    [1, 4],
                    # log range of the fast varying component
                    # typical scale = spectral resolution
                    np.log10([spec_model.spec_resln / 2.355, spec_model.spec_resln * 10]),
                ]
            ).T,
        )
        params_limit_2d["log_scale"] = params_limit_2d.get(
            "log_scale",
            np.array(
                [
                    # log range of the spatial component
                    # typical scale = spatial resolution
                    np.log10([spec_model.spat_resln / 2.355, spec_model.spat_resln]),
                    # log range of the spectral component
                    # typical scale = spectral resolution
                    np.log10([1e3, 1e5]),
                ]
            ).T,
        )
        params_limit_2d["mean"] = params_limit_2d.get("mean", np.array([-1e-1, 1e-1]).T)
        params_limit_2d["log_amp_line"] = params_limit_2d.get("log_amp_line", np.array([0, 5]).T)
        params_limit = [params_limit_1d, params_limit_2d]

        # Model the host
        spec_model.model_host(
            params_init=params_init,
            params_limit=params_limit,
            optimization=True,
            optimization_kwargs={"maxiter": 1000, "tol": 1e-2},
        )

        # QA plots
        plot_QA(spec_model, targetid)

