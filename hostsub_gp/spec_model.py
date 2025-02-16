# hostsub_gp/spec_model.py

__all__ = ["SpecModel"]

import numpy as np

import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

from ._utils import plt, msgs
from ._utils._plt import show_and_save
from ._utils._par import _transform_unbound_to_bound, _transform_bound_to_unbound, _init_params, _print_params
from .gp import GP
from .host_model import HostProfile
from .spec_wrapper import SpecWrapper

from jax._src.typing import ArrayLike, Array
from matplotlib.axes import Axes


class SpecModel:
    """
    A class for the host galaxy modeling on a rectified 2D spectrum.

    Attributes
    ----------
    spat : ArrayLike
        The spatial grids of the 2D spectrum.
    spec : ArrayLike
        The spectral grids of the 2D spectrum.
    shape : tuple[int, int]
        The shape of the 2D spectrum.
    pixel_scale : float
        The instrumental pixel scale of the 2D spectrum on the spatial axes (arcsec per pixel).
    center_ra : float
        The right ascension of science object.
    center_dec : float
        The declination of science object.
    slit_wid : float
        The width of the slit (arcsec).
    slit_len : float
        The length of the slit (arcsec).
    position_angle : float
        The position angle of the slit (degree).
    spat_resln : float
        The spatial resolution (FWHM/seeing) of the 2D spectrum (arcsec).
    spec_resln : float
        The spectral resolution of the 2D spectrum (angstrom).
    host_wid : float
        The width of the host galaxy region used for the fitting (in arcsec).
    mask_wid : float
        The width of the mask to mask the source trace (in spat_resln/seeing).
    sky_region: tuple
        The (inner edges of) global sky region defined as the distance from the source (in arcsec).
    batch_2d : tuple
        The batch size for modeling the slowly varying host profiles.
    f_obs : SpecWrapper
        The 2D spectrum of the observed data.
    f_sky : SpecWrapper
        The 1D spectrum of the global sky background (mean of the sky region).
    f_sky_sub : SpecWrapper
        The 2D spectrum of the sky-subtracted data.
    f_host : SpecWrapper
        The 2D spectrum of the sky-subtracted host galaxy (within the host region).
    f_host_1d : SpecWrapper
        The 1D spectrum of the sky-subtracted host galaxy (within the host region).
    f_host_batch_2d : SpecWrapper
        The batched 2D spectrum (batch size = batch_2d) of the sky-subtracted host galaxy (within the host region).
    f_batch_2d : SpecWrapper
        The batched 2D spectrum (batch size = batch_2d) of the sky-subtracted data.
    f_host_prior : SpecWrapper
        GP model of the photometric data (within the host region).
    f_batch_prior : SpecWrapper
        GP model of the photometric data (batched 2D spectrum).
    f_host_batch_prior : SpecWrapper
        GP model of the photometric data (batched 2D spectrum within the host region).
    dist_host_batch_2d : SpecWrapper
        The distance relative to the prior of the batched 2D spectrum (within the host region).
    dist_batch_2d : SpecWrapper
        The distance relative to the prior of the batched 2D spectrum.
    spat_filter : dict[str, ArrayLike]
        Spatial filters
        - mask: mask the source trace
        - host: mask the host galaxy region
        - sky: mask the sky region

    Methods
    -------
    model_host_prior
        Build the prior of the host galaxy using Gaussian Process regression.
    model_host
        Model the host galaxy using Gaussian Process regression.
    extract_sci
        Extract the 1D spectrum after host galaxy subtraction.
    """

    def __init__(
        self,
        dat: ArrayLike,  # 2D spectrum (spatial x spectral)
        dat_err: ArrayLike = None,  # 2D error spectrum
        *,
        spat: ArrayLike = None,  # spatial grids
        spec: ArrayLike = None,  # spectral grids
        pixel_scale: float = None,  # arcsec/pixel
        center_ra: float = None,  # RA of the center
        center_dec: float = None,  # DEC of the center
        slit_wid: float = 1.0,  # arcsec
        slit_len: float = None,  # arcsec
        position_angle: float = None,  # degree
        spat_resln: float = 1.0,  # arcsec, FWHM/seeing
        spec_resln: float = 7.5,  # LRIS, 1'' slit
        host_wid: float = 10.0,  # in arcsec, host region used for the fitting
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        mask_offset: float = 0.0,  # offset of the mask center (when the SN is not at the center)
        sky_region: tuple = (-5.0, 5.0),  # in arcsec, sky region
    ):
        # Load spectral configuration
        self.pixel_scale = pixel_scale
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.slit_wid = slit_wid
        if slit_len is None:
            self.slit_len = spat[-1] - spat[0] + spat[2] - spat[1]
        else:
            self.slit_len = slit_len
        self.position_angle = position_angle
        self.spat_resln = spat_resln
        self.spec_resln = spec_resln

        # Load the grid
        self.spat, self.spec = spat, spec
        self.shape = (len(spat), len(spec))

        # Define the mask, sky, and host regions
        self.mask_wid = mask_wid * self.spat_resln
        self.mask_offset = mask_offset
        self.sky_region = sky_region
        self.host_wid = host_wid
        self.spat_filter = self._get_spat_filter()

        # Load the the raw data
        self.f_obs = SpecWrapper(points=(spat, spec), values=dat, values_err=dat_err)
        msgs.info(f"Loading the 2D spectrum with the shape: {self.f_obs.shape}")

    def _get_spat_filter(self) -> dict:
        """
        Setup the spatial filters for the host galaxy modeling.
        """
        spat_filter = {"mask": None, "host": None, "sky": None}

        # The sky region
        # Adjust the sky edges to the nearest integer multiple of the pixel scale
        # Add 0.5 so the mask boundary is at the edge of the pixel
        if self.sky_region[0] is None:
            sky_region_left = -jnp.inf
            msgs.info(f"Excluding the left sky region")
        else:
            sky_region_left = jnp.ceil(self.sky_region[0] / self.pixel_scale - 0.5) * self.pixel_scale
            msgs.info(
                f"Sky region (left): {sky_region_left:.2f} arcsec = {sky_region_left / self.pixel_scale:.0f} pixels"
            )
        if self.sky_region[1] is None:
            sky_region_right = jnp.inf
            msgs.info(f"Excluding the right sky region")
        else:
            sky_region_right = jnp.ceil(self.sky_region[1] / self.pixel_scale + 0.5) * self.pixel_scale
            msgs.info(
                f"Sky region (right): {sky_region_right:.2f} arcsec = {sky_region_right / self.pixel_scale:.0f} pixels"
            )
        self.sky_region = (sky_region_left, sky_region_right)

        sky_left = self.spat < self.sky_region[0]
        sky_right = self.spat > self.sky_region[1]
        spat_filter["sky"] = sky_left | sky_right
        if np.nansum(spat_filter["sky"]) / spat_filter["sky"].ravel().size < 0.1:
            msgs.warning(r"Sky region is < 10% of the overall pixels.")
        if np.nansum(spat_filter["sky"]) == 0:
            raise ValueError("No sky region is defined.")

        # Mask the trace from the source (|spat| < mask_wid / 2)
        if min(np.abs(self.sky_region)) <= self.mask_wid:
            raise ValueError("sky_region boundary is inside the aperture mask")
        # Adjust the mask width to the nearest integer multiple of the pixel scale
        # Add 0.5 so the mask boundary is at the edge of the pixel
        self.mask_wid = (jnp.round(self.mask_wid / 2 / self.pixel_scale) * 2 + 1) * self.pixel_scale
        self.mask_offset = jnp.round(self.mask_offset / self.pixel_scale) * self.pixel_scale
        spat_filter["mask"] = (self.spat >= -self.mask_wid / 2 + self.mask_offset) & (
            self.spat <= self.mask_wid / 2 + self.mask_offset
        )
        msgs.info(
            f"Masking the source trace with the width: {self.mask_wid:.2f} arcsec = {self.mask_wid / self.pixel_scale:.0f} pixels"
        )

        # Define the host galaxy pixels (outside the mask)
        # Adjust the mask width to the nearest integer multiple of the pixel scale
        # Add 0.5 so the mask boundary is at the edge of the pixel
        self.host_wid = (jnp.round(self.host_wid / 2 / self.pixel_scale) * 2 + 1) * self.pixel_scale
        host_left = (self.spat < -self.mask_wid / 2 + self.mask_offset) & (
            self.spat > -self.host_wid / 2 + self.mask_offset
        )
        host_right = (self.spat > self.mask_wid / 2 + self.mask_offset) & (
            self.spat < self.host_wid / 2 + self.mask_offset
        )
        spat_filter["host"] = host_left | host_right

        return spat_filter

    def construct_spec_wrapper(
        self,
        f_obs: SpecWrapper,
        batch_2d: tuple[int, int] = (2, 64),  # batch size for modeling slowing varying host profiles
        host_emission_cfg: dict = None,  # parameters for identifying host emission lines
        sigma_clip: float = 5.0,  # sigma clipping threshold
        show: bool = False,
        save: str = None,
    ):
        """
        Construct the SpecWrapper objects corresponding to the 1D/2D spectrum in the global sky region, the host region, and in the batched 2D grids.

        Parameters
        ----------
        f_obs : SpecWrapper
            The 2D spectrum of the observed data.
        batch_2d : tuple[int, int], optional
            The spatial and spectral batch size for modeling the slowly varying host profiles, by default (2, 64).
        host_emission_cfg : dict, optional
            Parameters for identifying host emission lines, by default None.
        sigma_clip : float, optional
            Sigma clipping threshold, by default 5.0.
        show : bool, optional
            Whether to show the diagnostic plots, by default False.
        save : str, optional
            The path to save the diagnostic plots, by default None (do not save).
        """
        # Estimate the global sky background (sky + host): mean of the sky region along the spectral direction
        msgs.info(f"Estimating the global sky background")
        self.f_sky = f_obs.apply_spatial_filter(self.spat_filter["sky"]).sigma_clip(sigma=sigma_clip).fill_nan()
        self.f_sky_1d = self.f_sky.marginalize(margin_type="mean")
        # The 2D sky-subtracted, spectrum (to be sigma clipped)
        self.f_sky_sub = f_obs.subtract(self.f_sky_1d)
        # The 2D spectrum in the host galaxy region (outside the mask, to be sigma clipped)
        self.f_host = self.f_sky_sub.apply_spatial_filter(self.spat_filter["host"])

        # Obtain the batched 2D grids
        # 1. To reduce the computational cost in optimizing the 2D GP model
        # 2. To sigma clip the sky-subtracted 2D spectrum in each batch
        msgs.info(f"Batching the 2D galaxy spectrum (outside the mask) with the size: {batch_2d}")
        self.batch_2d = batch_2d
        # Spatial batch (only for the host galaxy pixels outside the mask)
        self._spat_batch_2d_idx, _spat_batch_2d_idx_in_host = self._get_spat_batches()
        # Spectral batch
        self._spec_batch_2d_idx = self._get_spec_batches(
            **host_emission_cfg, show=show, save=None if save is None else save.replace(".pdf", "_host_emission.pdf")
        )
        # The 2D sky-subtracted, sigma-clipped spectrum
        self.f_sky_sub = self.f_sky_sub.sigma_clip(
            sigma=sigma_clip, batch_idx=(self._spat_batch_2d_idx, self._spec_batch_2d_idx)
        )
        # The 2D spectrum in the host galaxy region: outside the mask
        self.f_host = self.f_sky_sub.apply_spatial_filter(self.spat_filter["host"])
        # The 1D grids for the sky-subtracted host galaxy spectra: sum along the spatial direction outside the mask
        # Sigma clip the 2D spectrum in each batch
        # Central wavelength in each row: spec
        # Total flux in each row: weighted sum of the flux in each row
        self.f_host_1d = self.f_host.fill_nan().marginalize(margin_type="sum")

        self.f_batch_2d = self.get_normalized_batch_spec(
            self._spat_batch_2d_idx, self._spec_batch_2d_idx, f_2d=self.f_sky_sub, f_1d_norm=self.f_host_1d
        )
        self.f_host_batch_2d = self.f_batch_2d.apply_spatial_filter(_spat_batch_2d_idx_in_host)
        msgs.info(f"Batched 2D galaxy spectrum: {self.f_host_batch_2d.shape}")

        self._plot_raw(show=show, save=save)

    def model_host(
        self,
        params_init: tuple[dict, dict] | list[dict] = None,
        params_limit: tuple[dict, dict] | list[dict] = None,
        optimization: bool = False,
        optimization_kwargs: dict = {},
    ):
        """
        Model the host galaxy using Gaussian Process regression.

        Parameters
        ----------
        params_init : dict
            Initial parameters for optimization.
        optimization : bool, optional (default: False)
            Whether to optimize the model with the jaxopt.ScipyMinimize solver.
        """

        # Make sure the host flux prior is built
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please build the host flux prior first.")

        # Calculate the distance relative to the prior
        msgs.info("Calculating the distance relative to the prior")
        self.dist_batch_2d = self.f_batch_2d.subtract(self.f_batch_prior)
        self.dist_host_batch_2d = self.f_host_batch_2d.subtract(self.f_host_batch_prior)

        # Initialize the parameters
        params_init_1d = self._set_params_init(params_init[0], ndim=1)
        params_init_2d = self._set_params_init(params_init[1], ndim=2)
        params_init = [params_init_1d, params_init_2d]

        # Set the limits for the parameters
        if params_limit is None:
            params_limit = [None, None]
        else:
            params_limit = _init_params(params_limit, require_all=False)

        if optimization:
            # Initialize the limits for the parameters
            params_limit[0] = self._set_params_limit(params_limit[0], ndim=1)
            params_limit[1] = self._set_params_limit(params_limit[1], ndim=2)

            msgs.info("Round 1: Fitting the 1D spectrum of the host galaxy")
            # Update the initial parameters with the 1D results
            params_init[0] = self._model_host_1d_opt(params_init=params_init[0], params_limit=params_limit[0])

            msgs.info("Round 2: Fitting the 2D spatial profile")
            self.gp_params = self._model_host_2d_opt(
                params_init=tuple(params_init),
                params_limit=tuple(params_limit),
                **optimization_kwargs,
            )
        else:
            self.gp_params = params_init

        self._gp_1d, self._gp_2d = self._build_host_gp(params=self.gp_params)

        # Predict the host galaxy flux on the entire 2D spectrum
        msgs.info("Predicting the host galaxy flux on the entire 2D spectrum")
        X_obs = self.f_obs.X.reshape(self.f_obs.shape[0], self.f_obs.shape[1], -1)
        self._f_1d_pred, self._f_2d_pred, self._f_pred = self._get_pred(self._gp_1d, self._gp_2d, X=X_obs)

    def model_host_prior(
        self,
        filters="ugrizy",
        from_archival: bool = True,
        wv_eff: ArrayLike = None,
        spat_slit: ArrayLike = None,
        counts_slit: ArrayLike = None,
        counts_err_slit: ArrayLike = None,
        **kwargs,
    ):
        """
        Build the prior of the host galaxy using Gaussian Process regression.
        """
        if from_archival:
            # Load the archival photometric data (PS1, SDSS)
            host_prof = HostProfile.from_archival(spec_model=self, filters=filters)
        elif (wv_eff is None) or (spat_slit is None) or (counts_slit is None) or (counts_err_slit is None):
            raise ValueError("Please provide the photometric data for modeling the host prior.")
        else:
            # Call the constructor of the HostProfile class
            host_prof = HostProfile(
                filters=filters,
                wv_eff=wv_eff,
                spec_model=self,
                spat_slit=spat_slit,
                counts_slit=counts_slit,
                counts_err_slit=counts_err_slit,
            )
        host_flux_prior = host_prof.model_host_profile_prior(return_var=True, **kwargs)

        _f_host = self.f_obs.apply_spatial_filter(self.spat_filter["host"])

        # Scale the host flux prior to the observed data
        # All pixels on the host region summed along the spatial axis = 1
        scale = lambda X: jnp.interp(
            X[..., 1].ravel(), self.spec, jnp.sum(host_flux_prior(_f_host.X)[0].reshape(_f_host.shape), axis=0)
        )

        def predict(X: Array) -> tuple[Array, Array]:
            prior = host_flux_prior(X)
            return prior[0] / scale(X), prior[1] ** 0.5 / scale(X)

        self.host_flux_prior = predict

        msgs.info("Building the host flux prior")
        # The entire 2D data
        prior, prior_std = self.host_flux_prior(self.f_obs.X)
        self.f_prior = SpecWrapper(
            points=(self.f_obs.spat, self.f_obs.spec),
            values=prior.reshape(self.f_obs.shape),
            values_err=prior_std.reshape(self.f_obs.shape),
        )
        # Within the host region
        self.f_host_prior = self.f_prior.apply_spatial_filter(self.spat_filter["host"])
        # The batched 2D data
        prior_batch, prior_batch_std = self.host_flux_prior(self.f_batch_2d.X)
        self.f_batch_prior = SpecWrapper(
            points=(self.f_batch_2d.spat, self.f_batch_2d.spec),
            values=prior_batch.reshape(self.f_batch_2d.shape),
            # values_err=prior_batch_std.reshape(self.f_batch_2d.shape),
        )
        # Batched 2D data (host region)
        prior_host_batch, prior_host_batch_std = self.host_flux_prior(self.f_host_batch_2d.X)
        self.f_host_batch_prior = SpecWrapper(
            points=(self.f_host_batch_2d.spat, self.f_host_batch_2d.spec),
            values=prior_host_batch.reshape(self.f_host_batch_2d.shape),
            # values_err=prior_host_batch_std.reshape(self.f_host_batch_2d.shape),
        )

    @msgs.timer
    def _match_seeing(self, min_dseeing: float = 0.0, max_dseeing: float = 0.5, step_dseeing: float = 0.01) -> Array:
        """
        Match the seeing of the host galaxy profile with the instrumental seeing.
        """
        # Version 0.1: assuming the seeing of the archival images is worse than the spectra
        # Make sure the host flux prior is built
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please build the host flux prior first.")

        _spat_batch_2d_idx, _spat_batch_2d_idx_in_host = self._get_spat_batches()
        _spec_batch_2d_idx = self._get_spec_batches(find_host_emission=False)

        chi2 = []

        # If the seeing of the archival images is worse than the spectra
        _f_obs_raw = self.f_obs  # .sigma_clip().fill_nan()
        dseeing_lst = np.arange(min_dseeing, max_dseeing, step=step_dseeing) + step_dseeing
        for k, dseeing in enumerate(dseeing_lst):
            # Empirical wavelength dependence of the seeing: FWHM ~ lambda^(-1/2.75)
            dseeing_spec = dseeing / self.pixel_scale * (self.spec / self.spec.mean()) ** (-1 / 2.75)
            _f_obs = _f_obs_raw.convolve(kernel_wid=dseeing_spec)

            # Sky subtraction
            _f_sky = _f_obs.apply_spatial_filter(self.spat_filter["sky"])
            _f_sky_sub = _f_obs.subtract(_f_sky.marginalize(margin_type="mean"))
            _f_host = _f_sky_sub.apply_spatial_filter(self.spat_filter["host"])
            _f_host_1d = _f_host.marginalize(margin_type="sum")

            # Batch the 2D spectrum
            _f_batch_2d = self.get_normalized_batch_spec(
                _spat_batch_2d_idx, _spec_batch_2d_idx, f_2d=_f_sky_sub, f_1d_norm=_f_host_1d
            )
            _f_host_batch_2d = _f_batch_2d.apply_spatial_filter(_spat_batch_2d_idx_in_host)

            # Get the prior
            if k == 0:
                prior_host_batch, _ = self.host_flux_prior(_f_host_batch_2d.X)
                _f_host_batch_prior = SpecWrapper(
                    points=(_f_host_batch_2d.spat, _f_host_batch_2d.spec),
                    values=prior_host_batch.reshape(_f_host_batch_2d.shape),
                )

            # Calculate the distance relative to the prior
            _dist_host_batch_2d = _f_host_batch_2d.subtract(_f_host_batch_prior)

            # Calculate the chi2
            chi2.append(jnp.sum(_dist_host_batch_2d.y**2 / _dist_host_batch_2d.yerr**2))

        # Find the best seeing
        plt.figure(figsize=(6, 4), constrained_layout=True)
        plt.plot(dseeing_lst, chi2, color="tab:blue")
        plt.xlabel(r"$\Delta \mathrm{Seeing\ [arcsec]}$")
        plt.ylabel(r"$\chi^2$")
        plt.show()

        best_dseeing = dseeing_lst[np.argmin(chi2)]
        msgs.info(f"Best delta seeing: {best_dseeing:.2f} arcsec")
        return best_dseeing

    def update_seeing(self, dseeing: ArrayLike) -> None:
        """
        Update the seeing of the host galaxy profile with the instrumental seeing.
        """
        # Update the spatial resolution
        spat_resln_0 = self.spat_resln
        msgs.info(f"Original spatial resolution: {spat_resln_0:.2f} arcsec")
        self.spat_resln = (spat_resln_0 ** 2 + dseeing ** 2) ** 0.5
        msgs.info(f"Updated spatial resolution: {self.spat_resln:.2f} arcsec")

        # Update the mask, sky, and host regions
        self.mask_wid = self.mask_wid * self.spat_resln / spat_resln_0
        self.spat_filter = self._get_spat_filter()

    @show_and_save
    def extract_sci(self, method="boxcar") -> Axes:  # TODO: adopt the extraction method of pypeit
        """
        Extract the science spectrum after host galaxy subtraction (within the mask).
        """
        # Predict the host galaxy flux within the mask (including uncertainties)
        msgs.info("Extracting the science spectrum.")

        self.f_mask = self.f_sky_sub.apply_spatial_filter(self.spat_filter["mask"])
        X_mask = self.f_mask.X.reshape(self.f_mask.shape[0], self.f_mask.shape[1], -1)
        _, _, (f_mask_pred, f_mask_pred_err) = self._get_pred(self._gp_1d, self._gp_2d, X_mask, return_var=True)
        self.f_mask_pred = SpecWrapper(
            points=(self.f_mask.spat, self.f_mask.spec),
            values=f_mask_pred.reshape(self.f_mask.shape),
            # values_err=jnp.zeros(self.f_mask.shape),
            values_err=f_mask_pred_err.reshape(self.f_mask.shape),
            # TODO: add reasonable errors
        )
        self.f_sci_pred = self.f_mask.subtract(self.f_mask_pred)

        def gauss(x, mu, sigma):
            return jnp.exp(-0.5 * (x - mu) ** 2 / sigma**2) * (2 * jnp.pi * sigma**2) ** -0.5

        if method == "boxcar":
            extract_weights = None
        elif method == "optimal":
            extract_weights = gauss(self.f_mask.spat, self.mask_offset, self.spat_resln / 2.355)
        else:
            raise ValueError("Invalid extraction method.")

        self.f_sci_pred_1d = self.f_sci_pred.marginalize(margin_type="mean", weights=extract_weights)

        if not hasattr(self, "_f_pred"):
            raise AttributeError("Please model the host galaxy first.")
        _, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(self.f_sci_pred_1d.X, self.f_sci_pred_1d.y, color="tab:blue")
        ax.axhline(0, color="k", ls="--")
        ax.set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        ax.set_ylabel(r"$\mathrm{Counts}$")

        return ax

    def _set_params_init(self, params_init: dict = None, ndim: int = 1) -> dict:
        """
        Setup the initial parameters for the Gaussian Process model.

        Parameters
        ----------
        params_init : dict, optional
            The user input initial parameters for the Gaussian Process.
        ndim : int, optional
            The number of dimensions for the Gaussian Process.

        Returns
        -------
        dict
            The initial parameters for the Gaussian Process model.
        """
        # 1D spectrum of the host galaxy
        if ndim == 1:
            log_amp_est = np.log10(((self.f_host_1d.y) ** 2).max())
            mean_est = np.nanmean(self.f_host_1d.y)
            params_init_default = dict(
                log_amp=(
                    log_amp_est,  # ExpSquared: Logarithm of the maximum squared value of the 1D spectrum
                    log_amp_est - 2,  # Matern: Somewhat smaller
                ),
                log_scale=(
                    2,  # ExpSquared: 100 Angstrom
                    np.log10(self.spec_resln / 2.355),  # Matern: Spectral resolution / 2.355
                ),
                mean=mean_est,  # Mean of the 1D spectrum
            )
        elif ndim == 2:
            params_init_default = dict(
                log_amp=-4.0,
                log_scale=(
                    np.log10(self.spec_resln),  # Spatial scale ~ seeing
                    3,  # Spectral scale ~ 1000 Angstrom
                ),
                mean=0.0,
                log_amp_line=1.0,  # Covariance within the host lines = covariance outside the host lines
                scale_line=self.spec_resln / 2,  # Radius of the host lines: Half of the FWHM of the spectral resolution
            )
        else:
            raise ValueError("Invalid number of dimensions.")

        if params_init is None:
            params_init = params_init_default
        else:
            for k, v in params_init_default.items():
                if k not in params_init:
                    params_init[k] = v

        return _init_params(params_init, require_all=True)

    def _set_params_limit(self, params_limit: dict = None, ndim: int = 1) -> dict:
        """
        Setup the parameters limits by merging the user input limits with the default limits.

        Parameters
        ----------
        params_limit : dict, optional
            The user input limits for the Gaussian Process parameters.
        ndim : int, optional
            The number of dimensions for the Gaussian Process.

        Returns
        -------
        dict
            The merged parameters limits.
        """
        large_scale = 1e4
        small_scale = 1e-3

        # 1D spectrum of the host galaxy
        ## scale >= spectral resolution / 2.355
        ### Here we use a composite kernel
        ### ExpSquared - slow variation (>> spectral resolution)
        ### Matern - narrow features (~ spectral resolution)
        if ndim == 1:  # TODO: modify the default limits with a configuration file
            params_limit_default = dict(
                log_scale=np.log10(
                    [
                        [
                            self.spec_resln / 2.355,  # Limit for the ExpSquared kernel
                            self.spec_resln / 2.355,  # Limit for the Matern kernel
                        ],
                        [
                            self.spec_resln * large_scale,  # Limit for the ExpSquared kernel
                            self.spec_resln * 2,  # Limit for the Matern kernel
                        ],
                    ]
                ),
            )

        # 2D spatial profile & 1D spectrum of the host galaxy
        ## scale >= spatial resolution / 2.355, spectral resolution / 2.355
        ### here we use a single kernel and adjust the kernel within the host lines
        ## mean (i.e., deviation fromt the prior) is close to zero
        elif ndim == 2:
            params_limit_default = dict(
                log_scale=np.log10(
                    [
                        [self.spat_resln / 2.355, self.spat_resln * large_scale],
                        [self.spec_resln / 2.355, self.spec_resln * large_scale],
                    ]
                ),
                mean=[-small_scale, small_scale],
                scale_line=[self.spec_resln / 2.355 / 2, self.spec_resln * large_scale],
            )

        else:
            raise ValueError("Invalid number of dimensions.")

        if params_limit is None:
            params_limit = params_limit_default
        else:
            for key in params_limit:
                if key in params_limit_default:
                    params_limit_default.pop(key)
            params_limit[0] = {**params_limit_default, **params_limit}

        return _init_params(params_limit, require_all=False)

    ###############################################################################
    ############################ Host Galaxy Modeling #############################
    ###############################################################################

    @msgs.timer
    def _model_host_1d_opt(
        self,
        params_init: dict,
        params_limit: dict,
    ) -> dict:
        """
        Optimize the Gaussian process model of the host using jaxopt.ScipyMinimize solver.
        Only the 1D spectrum is optimized in this step.
        """

        params_limit = self._set_params_limit(params_limit, ndim=1)

        params_1d = GP(
            kernel_type="1D",
            X=self.f_host_1d.X,
            y=self.f_host_1d.y,
            yerr=self.f_host_1d.yerr,
            params_init=params_init,
            params_limit=params_limit,
            optimization=True,
        ).params

        return params_1d

    @msgs.timer
    def _model_host_2d_opt(
        self, params_init: tuple[dict, dict], params_limit: tuple[dict, dict], **kwargs
    ) -> tuple[dict, dict]:
        """
        Optimize the Gaussian process model of the host using jaxopt.ScipyMinimize solver.
        Only the 2D spatial profile is optimized in this step.
        """
        params_init_unbound = _transform_bound_to_unbound(params_init, params_limit)
        msgs.info("Optimizing the host galaxy model...")
        neg_log_prob_init = self._get_host_neg_log_probability(params=params_init_unbound, params_limit=params_limit)
        msgs.info(f"Initial negative log-probability: {neg_log_prob_init:.1f}")
        if ~np.isfinite(neg_log_prob_init):
            msgs.error("Initial log-probability is infinite.")
            msgs.info("Initial parameters:")
            _print_params(params_init)
            msgs.info("Parameter limits:")
            _print_params(params_limit)
            raise ValueError("Invalid initial parameters: please check the limits.")

        solver = jaxopt.ScipyMinimize(
            fun=partial(
                self._get_host_neg_log_probability,
                params_1d=params_init_unbound[0],
                params_limit=params_limit,
            ),
            method="SLSQP",
            **kwargs,
        )
        soln = solver.run(params_init_unbound[1])
        if soln.state.status != 0:
            msgs.warning(f"Optimization failed with status {soln.state.status}.")
        params = _transform_unbound_to_bound(soln.params, params_limit[1])
        msgs.info(f"Final negative log-probability: {soln.state.fun_val:.1f}")
        msgs.info("Final parameters:")
        _print_params(params)
        return (params_init[0], params)

    def _build_host_gp(
        self, params: tuple[dict, dict], params_limit: tuple[dict, dict] = (None, None)
    ) -> tuple[GP, GP]:
        """
        Build the Gaussian Process for the 1D host galaxy spectra and 2D host galaxy spatial profiles.

        Parameters
        ----------
        params_1d : dict
            Parameters for the 1D Gaussian Process - the 1D spectrum of the host.
        params_2d : dict
            Parameters for the 2D Gaussian Process - the spatial profile of the host.

        Returns
        -------
        tuple[GP, GP]
            GP objects for the 1D and 2D host galaxy.
        """
        params_1d, params_2d = _init_params(params)
        try:
            params_limit_1d, params_limit_2d = _init_params(params_limit, require_all=False)
        except:
            raise ValueError("Invalid parameter limits")

        f_1d = self.f_host_1d
        f_1d_mask = np.isfinite(f_1d.y)
        gp_1d = GP(
            kernel_type="1D",
            X=f_1d.X[f_1d_mask],
            y=f_1d.y[f_1d_mask],
            yerr=f_1d.yerr[f_1d_mask],
            params=params_1d,
            params_limit=params_limit_1d,
        )

        f_2d = self.dist_host_batch_2d
        f_2d_mask = np.isfinite(f_2d.y)
        gp_2d = GP(
            kernel_type="2D",
            emission_lines=self.emission_lines,
            X=f_2d.X[f_2d_mask],
            y=f_2d.y[f_2d_mask],
            yerr=f_2d.yerr[f_2d_mask],
            params=params_2d,
            params_limit=params_limit_2d,
        )

        return gp_1d, gp_2d

    def _get_host_neg_log_probability(
        self,
        params_2d: dict = None,
        params_1d: dict = None,
        params: tuple[dict, dict] = None,
        params_limit: tuple[dict, dict] = None,
    ) -> float:
        """
        Calculate the negative log probability of the host flux given the parameters.

        Parameters
        ----------
        params_2d : dict
            Parameters for the 2D Gaussian Process.
        params_1d : dict
            Parameters for the 1D Gaussian Process.
        params : tuple[dict, dict]
            A tuple of parameters for the 1D and 2D Gaussian Processes.
        params_limit : dict, optional
            Limits for the Gaussian Process parameters.

        Returns
        -------
        float
            The negative log probability of the host flux.
        """
        if params is None:
            params = (params_1d, params_2d)
        params_1d, params_2d = _init_params(_transform_unbound_to_bound(params, params_limit))

        @jax.jit
        def _neg_log_probability(
            params_1d: dict,
            params_2d: dict,
            f_X: Array,
            f_y: Array,
            f_yerr: Array,
            f_1d_X: Array,
            f_1d_y: Array,
            f_1d_yerr: Array,
            dist_2d_X: Array,
            dist_2d_y: Array,
            dist_2d_yerr: Array,
            f_mean: Array,
            emission_lines: Array,
        ) -> float:
            """
            Compute the negative log probability of the host galaxy model
            """
            gp_1d = GP(kernel_type="1D", X=f_1d_X, y=f_1d_y, yerr=f_1d_yerr, params=params_1d)
            gp_2d = GP(
                kernel_type="2D",
                X=dist_2d_X,
                y=dist_2d_y,
                yerr=dist_2d_yerr,
                params=params_2d,
                emission_lines=emission_lines,
            )
            log_prob_1d = gp_1d.log_probability(f_1d_y)
            log_prob_2d = gp_2d.log_probability(dist_2d_y)

            y_host_1d = gp_1d.predict(X_test=f_X[:, 1:])
            y_host_2d = gp_2d.predict(X_test=f_X) + f_mean
            y_host = y_host_1d * y_host_2d
            log_prob_obs = jnp.nansum(jax.scipy.stats.norm.logpdf(y_host, f_y, f_yerr))

            # jax.debug.print("{}", params_1d)
            # jax.debug.print("{}", params_2d)
            # jax.debug.print("1D log-probability: {}", log_prob_1d)
            # jax.debug.print("2D log-probability: {}", log_prob_2d)
            # jax.debug.print("Observed log-probability: {}", log_prob_obs)
            # jax.debug.print("Log posterior: {}", log_prob_1d + log_prob_2d + log_prob_obs)

            return -(log_prob_1d + log_prob_2d + log_prob_obs)

        # Only include finite values in the observation
        obs_mask = np.isfinite(self.f_host.y)
        f_1d_mask = np.isfinite(self.f_host_1d.y)
        f_2d_mask = np.isfinite(self.dist_host_batch_2d.y)

        return _neg_log_probability(
            params_1d=params_1d,
            params_2d=params_2d,
            f_X=self.f_host.X[obs_mask],
            f_y=self.f_host.y[obs_mask],
            f_yerr=self.f_host.yerr[obs_mask],
            f_1d_X=self.f_host_1d.X[f_1d_mask],
            f_1d_y=self.f_host_1d.y[f_1d_mask],
            f_1d_yerr=self.f_host_1d.yerr[f_1d_mask],
            dist_2d_X=self.dist_host_batch_2d.X[f_2d_mask],
            dist_2d_y=self.dist_host_batch_2d.y[f_2d_mask],
            dist_2d_yerr=self.dist_host_batch_2d.yerr[f_2d_mask],
            f_mean=self.f_host_prior.y[obs_mask],
            emission_lines=self.emission_lines,
        )

    def _get_pred(
        self, gp_1d: GP, gp_2d: GP, X: Array, return_var: bool = False
    ) -> tuple[Array, Array, Array] | tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
        """
        Get the predicted host galaxy flux on the given grids.

        Parameters
        ----------
        gp_1d : GP
            The 1D Gaussian Process - the 1D spectrum of the host.
        gp_2d : GP
            The 2D Gaussian Process - the spatial profile of the host.
        X : Array
            The 2D grids to make the prediction.
            Shape: (n_spat, n_spec, 2)

        Returns
        -------
        Array | tuple[Array, Array]
            The flattened, predicted host galaxy flux.
        """
        if (X.ndim != 3) or (X.shape[-1] != 2):
            raise ValueError("Invalid input grids: the shape of X must be (n_spat, n_spec, 2).")
        # Input for the 1D GP: (n_spat, 1)
        X_1d = X[0, :, 1:]
        # Input for the 2D GP: (n_spat * n_spec, 2)
        X_2d = X.reshape(-1, 2)

        # Obtain the mean and standard deviation of the mean function
        prior, prior_std = self.host_flux_prior(X_2d)

        if return_var:
            y_1d, y_1d_var = [jnp.tile(y, reps=X.shape[0]) for y in gp_1d.predict(X_test=X_1d, return_var=True)]
            y_2d, y_2d_var = gp_2d.predict(X_test=X_2d, return_var=True)

            y_2d += prior
            # y_2d_var += prior_std**2

            y = y_1d * y_2d
            y_var = y**2 * (y_1d_var / y_1d**2 + y_2d_var / y_2d**2)

            return (y_1d, y_1d_var**0.5), (y_2d, y_2d_var**0.5), (y, y_var**0.5)

        else:
            y_1d = jnp.tile(gp_1d.predict(X_test=X_1d), reps=X.shape[0])
            y_2d = gp_2d.predict(X_test=X_2d) + prior
            y = y_1d * y_2d

            return y_1d, y_2d, y

    def _get_gp_params(self) -> dict:
        """
        Get the Gaussian Process parameters.

        Returns
        -------
        dict
            The Gaussian Process parameters.
        """
        if not hasattr(self, "gp_params"):
            raise AttributeError("Please model the host galaxy first.")
        _print_params(self.gp_params)
        return self.gp_params

    ###############################################################################
    ############################## Adaptive Batching ##############################
    ###############################################################################

    def get_normalized_batch_spec(
        self, spat_batch_idx: list, spec_batch_idx: list, f_2d: SpecWrapper, f_1d_norm: SpecWrapper
    ) -> SpecWrapper:
        """
        Get the batched 2D spectrum normalized by a 1D spectrum.

        Parameters
        ----------
        spat_batch_idx : ArrayLike
            Batch indices for the spatial axis.
        spec_batch_idx : ArrayLike
            Batch indices for the spectral axis.
        f_2d : SpecWrapper, optional
            The 2D spectrum to be batched.
        f_1d_norm : SpecWrapper, optional
            The 1D spectrum to normalize the 2D spectrum.

        Returns
        -------
        SpecWrapper
            The batched 2D spectrum.
        """

        # New coordinates: mean of the batch
        shape_batch_2d = (len(spat_batch_idx), len(spec_batch_idx))
        spat_batch_2d = np.asarray([self.spat[idx].mean() for idx in spat_batch_idx])
        spec_batch_2d = np.asarray([self.spec[idx].mean() for idx in spec_batch_idx])

        # New values: mean of the batch
        values_batch_2d = np.empty(shape_batch_2d)
        values_err_batch_2d = np.empty(shape_batch_2d)

        for y, idx_spec in enumerate(spec_batch_idx):
            # Get the 1D spectrum within the spectral batch
            Y_1d = f_1d_norm.Y[idx_spec]
            Y_err_1d = f_1d_norm.Yerr[idx_spec]
            # Get the 2D spectrum within the spectral batch
            Y_2d_spec = f_2d.Y[:, idx_spec]
            Y_err_2d_spec = f_2d.Yerr[:, idx_spec]

            for x, idx_spat in enumerate(spat_batch_idx):
                # Step 1: Bin along the spatial axis
                n_spat = len(idx_spat)
                Y_2d = np.nanmean(Y_2d_spec[idx_spat], axis=0)
                Y_err_2d = (np.nanmean(Y_err_2d_spec[idx_spat] ** 2, axis=0) / n_spat) ** 0.5

                # Step 2: Estimate the flux ratios (normalized profiles)
                Y_2d_1d = Y_2d / Y_1d
                Y_err_2d_1d = Y_2d_1d * ((Y_err_2d / Y_2d) ** 2 + (Y_err_1d / Y_1d) ** 2) ** 0.5

                # Step 3: Bin along the spectral axis
                values_batch_2d[x, y] = np.nanmedian(Y_2d_1d)
                values_err_batch_2d[x, y] = np.nanmean(Y_err_2d_1d**2 / len(spec_batch_idx[y])) ** 0.5

        return SpecWrapper(
            points=(spat_batch_2d, spec_batch_2d),
            values=values_batch_2d,
            values_err=values_err_batch_2d,
        )

    def _get_spat_batches(self) -> tuple[list[list[int]], Array]:
        """
        Get the batch indices for the spatial direction.

        Returns
        -------
        list[list[int]]
            The indices of the spatial batches.
        Array
            Indicating which batches are within the host galaxy (i.e., outside the mask)
        """
        host_left = (self.spat < -self.mask_wid / 2 + self.mask_offset) & (
            self.spat > -self.host_wid / 2 + self.mask_offset
        )
        host_right = (self.spat > self.mask_wid / 2 + self.mask_offset) & (
            self.spat < self.host_wid / 2 + self.mask_offset
        )

        batch_2d = self.batch_2d
        mask = self.spat_filter["mask"]

        # On the left side of the mask
        if host_left.sum() > 0:
            # Finer binning near the aperture edge (by default)
            spat_batch_2d_left = np.array_split(np.arange(self.shape[0])[host_left], host_left.sum() // batch_2d[0])
        else:
            spat_batch_2d_left = []
        # On the right side of the mask
        if host_right.sum() > 0:
            # Finer binning near the aperture edge
            spat_batch_2d_right = np.array_split(
                np.arange(self.shape[0])[host_right][::-1], host_right.sum() // batch_2d[0]
            )[::-1]
            spat_batch_2d_right = [np.sort(idx) for idx in spat_batch_2d_right]
        else:
            spat_batch_2d_right = []
        if len(spat_batch_2d_left + spat_batch_2d_right) == 0:
            raise ValueError("No host galaxy pixels found.")
        # Within the mask
        if mask.sum() > 0:
            spat_batch_2d_sci = np.array_split(np.arange(self.shape[0])[mask], (mask).sum())
        else:
            raise ValueError("No pixels within the mask.")
        # Combine the batch indices
        spat_batch_2d_idx = spat_batch_2d_left + spat_batch_2d_sci + spat_batch_2d_right
        spat_batch_2d_idx_in_host = (jnp.arange(len(spat_batch_2d_idx)) < len(spat_batch_2d_left)) | (
            jnp.arange(len(spat_batch_2d_idx)) >= len(spat_batch_2d_left + spat_batch_2d_sci)
        )
        return spat_batch_2d_idx, spat_batch_2d_idx_in_host

    def _get_spec_batches(self, **kwargs) -> list[list[int]]:
        """
        Get the batch indices for the spectral direction.

        Returns
        -------
        list[list[int]]
            The indices of the spectral batches.
        """
        emission_lines_idx, emission_lines = self._find_host_emission(**kwargs)
        if "find_host_emission" in kwargs:
            self.emission_lines = emission_lines
            if len(emission_lines) > 0:
                msgs.info(f"Emission lines found at: {self.emission_lines}")

        emission_lines_idx = np.concatenate([[0], emission_lines_idx, [self.spec.size - 1]])
        batch_edges = []
        for i in range(len(emission_lines_idx) - 1):
            edges = self._find_batch_edges(left=emission_lines_idx[i], right=emission_lines_idx[i + 1])
            batch_edges.extend(edges)

        batch_idx = []
        for i in range(len(batch_edges) - 1):
            idx = np.arange(batch_edges[i], batch_edges[i + 1])
            if idx.size > 0:
                batch_idx.append(idx)

        return batch_idx

    @show_and_save
    def _find_host_emission(
        self,
        find_host_emission: bool = True,
        p_value: float = 0.05,
        kernel_wid: int = None,
        z: float = None,
        z_err: float = None,
    ) -> tuple[Array, Array]:
        """
        Find the edges of the host galaxy emission using the 1D spectrum.

        Parameters
        ----------
        find_host_emission : bool, optional (default: True)
            Whether to find the host galaxy emission.
        p_value : float, optional (default: 0.05)
            The p-value for emission line detection.
        kernel_wid : int, optional
            The width of the kernel for smoothing the profile.
        z : float, optional (default: 0.0)
            The redshift of the host galaxy.
        z_err : float, optional
            The error of the redshift.

        Returns
        -------
        Tuple[Array, Array]
            The indices & wavelengths of the host galaxy emission.
        """
        from scipy.signal import find_peaks
        from scipy.stats import chi2
        from astropy.stats import mad_std
        from astropy.table import Table

        from importlib import resources
        from pathlib import Path

        if not find_host_emission:
            return jnp.array([], dtype=int), jnp.array([], dtype=float)

        # Define the kernel for smoothing the standard deviation of the galaxy spatial profile
        if kernel_wid is None:
            kernel_wid = int(self.spec_resln / jnp.diff(self.spec).min())

        f_2d = np.empty_like(self.f_host.Y)
        f_2d_err = np.empty_like(self.f_host.Yerr)
        for i in range(len(self.spec)):
            # Binning the spatial profile for higher S/N and bad pixels removal
            left = max(0, i - int(kernel_wid / 2))
            right = min(len(self.spec), i + int(kernel_wid / 2) + 1)
            f_2d[:, i] = np.nanmedian(self.f_host.Y[:, left:right], axis=1)
            f_2d_err[:, i] = np.nanmedian(self.f_host.Yerr[:, left:right], axis=1) / np.sqrt(right - left)

        # 1D spectrum
        f_1d = np.nansum(f_2d, axis=0)
        # Spatial profile and the error
        prof = jnp.asarray(f_2d / f_1d)
        prof_err = jnp.asarray(f_2d_err / f_1d)

        # Continuum estimation
        f_1d_cont = np.empty_like(f_1d)
        prof_med = np.empty_like(prof)
        for i in range(len(self.spec)):
            left_wide = max(0, i - kernel_wid * 2)
            right_wide = min(len(self.spec), i + kernel_wid * 2 + 1)
            prof_med[:, i] = jnp.nanmedian(prof[:, left_wide:right_wide], axis=1)
            f_1d_cont[i] = jnp.nanmedian(f_1d[left_wide:right_wide])

        # Difference between the observed and the continuum
        f_lines = jnp.abs(f_1d - f_1d_cont)
        # Sum of the squared difference between the profile at each wavelength and the average profile (median)
        prof_diff = jnp.nanmean(((prof - prof_med) / prof_err) ** 2, axis=0) * prof.shape[0]

        # Find the emission lines
        ## Flux significantly higher than the continuum (5-sigma)
        ## Spatial profile significantly different from the median profile (chi^2 test)
        distinct_prof, _ = find_peaks(prof_diff, height=chi2.ppf(1 - p_value, prof.shape[0]), distance=kernel_wid)
        host_lines = jnp.argwhere(f_lines > mad_std(f_lines) * 5).ravel()

        emission_lines_idx = []
        for line in distinct_prof:
            host_lines_close = np.where(np.abs(host_lines - line) < kernel_wid)
            if host_lines_close[0].size > 0:
                emission_lines_idx.append(int(np.round(np.mean(host_lines[host_lines_close]))))

        # Remove duplicates
        emission_lines_idx = np.unique(emission_lines_idx)

        # Read the host emission line library
        with resources.path("hostsub_gp.data", "Emission_line_list.csv") as path:
            emission_lines_lib = Table.read(path, format="csv", comment="#")
        wv_lib = emission_lines_lib["Wavelength"].data
        weight_lib = emission_lines_lib["Weight"].data
        if z is not None:
            z_err = z_err if z_err is not None else z * 1e-1
            zs = np.linspace(z - z_err, z + z_err, 100)
        else:
            zs = np.linspace(0, 0.1, 500)  # uncertainty <~ 1 Angstrom
        # Find the redshift to match the emission lines
        ccfs = np.empty_like(zs)
        for k, _z in enumerate(zs):
            spec_lib_at_z = np.zeros_like(self.spec)
            for _wv, _weight in zip(wv_lib, weight_lib):
                spec_lib_at_z += _weight * np.exp(
                    -(((self.spec - _wv * (1 + _z)) / (self.spec_resln / 2.355 * 2)) ** 2)
                )
            ccfs[k] = (f_lines * spec_lib_at_z).sum()
        emission_lines_in_lib = wv_lib * (1 + zs[np.argmax(ccfs)])
        # Match the found emission lines with the library
        emission_lines = []
        emission_lines_idx_updated = []
        if len(emission_lines_idx) == 0:
            emission_lines = np.array([])
            emission_lines_idx_updated = np.array([])
        else:
            for line in zip(self.spec[emission_lines_idx]):
                if np.min(np.abs(emission_lines_in_lib - line)) < self.spec_resln:
                    emission_lines.append(emission_lines_in_lib[np.argmin(np.abs(emission_lines_in_lib - line))])
                    emission_lines_idx_updated.append(
                        np.interp(emission_lines[-1], self.spec, np.arange(len(self.spec)))
                    )
                    # emission_lines_idx_updated.append(np.argmin(np.abs(self.spec - emission_lines[-1])))

            emission_lines = np.unique(emission_lines)
            emission_lines_idx_updated = np.unique(emission_lines_idx_updated)

        _, ax = plt.subplots(2, 1, figsize=(20, 5), sharex=True, constrained_layout=True)
        ax[0].plot(self.spec, f_lines, color="tab:blue")
        ax[0].axhline(mad_std(f_lines) * 5, color="0.5", ls="--")
        ax[0].set_ylabel(r"$|f - f_\mathrm{cont}|$")
        ax[1].plot(self.spec, prof_diff, color="tab:blue")
        ax[1].axhline(chi2.ppf(1 - p_value, prof.shape[0]), color="0.5", ls="--")
        for line in emission_lines:
            ax[0].axvline(line, color="tab:red", ls=":")
            ax[1].axvline(line, color="tab:red", ls=":")
        ax[1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        ax[1].set_ylabel(r"$\chi^2$")
        ax[1].set_yscale("log")

        return jnp.asarray(emission_lines_idx_updated), jnp.asarray(emission_lines, dtype=float)

    def _find_batch_edges(self, left: int = None, right: int = None) -> ArrayLike:
        """
        Find the edges of the batches with adaptive sizes for the 2D spectrum.

        Parameters
        ----------
        left, right : int, optional (default: None)
            The left and right edges of the narrow lines in the spectrum.
            None means the beginning and the end of the spectrum, respectively.

        Returns
        -------
        ArrayLike
            The indices of the batch edges.
        """
        left_edge = 0
        right_edge = self.spec.size - 1

        if left is None:
            left = left_edge
        if right is None:
            right = right_edge

        min_batch_size = 2 * (int(self.spec_resln / np.diff(self.spec).min()))
        max_batch_size = self.batch_2d[1]

        def check_spectrum_length(left, right):
            if right - left < min_batch_size * 2:
                raise ValueError(f"The spectrum is too short for the batch size: Left = {left}, Right = {right}")

        # No narrow lines in the spectrum
        # Use the largest possible batch size
        if (left == left_edge) and (right == right_edge):

            check_spectrum_length(left, right)

            # The number of batches is determined such that by dividing the spectrum into n_batch (n_batch > 2) nearly equal bins,
            # the batch size is the largest possible value below max_batch_size
            n_batch = int(np.ceil((right - left) / max_batch_size))
            batch_edges = np.linspace(left, right, n_batch + 1).astype(int)

        # The left edge is the beginning of the spectrum
        # The right edge is a narrow line
        elif left == left_edge:
            right = min(right + min_batch_size / 2, right_edge)  # Ending at the right edge of the narrow line

            check_spectrum_length(left, right)

            # Batches on the right have the sizes: (2^K_max, 2^(K_max-1), ..., 2^1, 2^0, 2^0) * min_batch_size
            # These batches add up to L_right = 2^(K_max+1) * min_batch_size
            # K_max is limited by:
            # 1. 2^K_max * min_batch_size + L_right <= right - left
            # 2. 2^K_max * min_batch_size <= max_batch_size
            # n_batch_right = K_max + 2
            n_batch_right = (
                min(
                    int(np.floor(np.log2((right - left) / min_batch_size / 3))),
                    int(np.log2(max_batch_size / min_batch_size)),
                )
                + 2
            )
            batch_edges_right = (-min_batch_size * 2 ** np.arange(0, n_batch_right) + right)[::-1]

            # Batches on the left have the same sizes
            # i.e., the maximum batch size below 2^K_max * min_batch_size, which can divide the remaining spectrum nearly equally
            n_batch_left = int(
                np.ceil((batch_edges_right[0] - left) / min(2**n_batch_right * min_batch_size, max_batch_size))
            )
            batch_edges_left = np.linspace(left, batch_edges_right[0], n_batch_left + 1).astype(int)[:-1]

            batch_edges = np.concatenate([batch_edges_left, batch_edges_right])

        # The left edge is a narrow line
        # The right edge is the end of the spectrum
        elif right == right_edge:
            left = max(left - min_batch_size / 2, left_edge)

            check_spectrum_length(left, right)

            # Batches on the left have the sizes: (2^0, 2^0, 2^1, ..., 2^K_max) * min_batch_size
            # These batches add up to L_left = (2^(K_max+1) - 1) * min_batch_size
            # K_max is limited by:
            # 1. 2^K_max * min_batch_size + L_left <= right - left
            # 2. 2^K_max * min_batch_size <= max_batch_size
            # n_batch_left = K_max + 1
            n_batch_left = (
                min(
                    int(np.floor(np.log2((right - left) / min_batch_size / 3))),
                    int(np.log2(max_batch_size / min_batch_size)),
                )
                + 2
            )
            batch_edges_left = min_batch_size * 2 ** np.arange(0, n_batch_left) + left

            # Batches on the right have the same sizes
            # i.e., the maximum batch size below 2^(K_max+1) * min_batch_size, which can divide the remaining spectrum nearly equally
            n_batch_right = int(
                np.ceil((right - batch_edges_left[-1]) / min(2**n_batch_left * min_batch_size, max_batch_size))
            )
            batch_edges_right = np.linspace(batch_edges_left[-1], right, n_batch_right + 1).astype(int)[1:]

            batch_edges = np.concatenate([batch_edges_left, batch_edges_right])

        # Both edges are narrow lines
        else:
            left = max(left - min_batch_size / 2, left_edge)
            right = min(right + min_batch_size / 2, right_edge)

            if right - left <= min_batch_size * 2:
                return np.array([(left + right) / 2], dtype=int)
            elif right - left <= min_batch_size * 3:
                return np.array([np.ceil(left + min_batch_size), np.ceil(right - min_batch_size)], dtype=int)

            # Batches on the left have the sizes: (2^0, 2^0, 2^1, ..., 2^K_max) * min_batch_size
            # Batches on the right have the sizes: (2^K_max, 2^(K_max-1), ..., 2^1, 2^0, 2^0) * min_batch_size
            # These batches add up to L_left_right = 2 * (2^(K_max+1) - 1) * min_batch_size
            # K_max is limited by:
            # 1. 2^K_max * min_batch_size + L_left_right <= right - left
            # 2. 2^K_max * min_batch_size <= max_batch_size
            # n_batch_left = n_batch_right = K_max + 1
            n_batch_left = n_batch_right = (
                min(
                    int(np.floor(np.log2((right - left) / min_batch_size / 5))),
                    int(np.log2(max_batch_size / min_batch_size)),
                )
                + 2
            )
            batch_edges_left = min_batch_size * 2 ** np.arange(0, n_batch_left) + left
            batch_edges_right = (-min_batch_size * 2 ** np.arange(0, n_batch_right) + right)[::-1]

            # Batches in the middle have the same sizes
            # i.e., the maximum batch size below min(2^(K_max+1) * min_batch_size, max_batch_size), which can divide the remaining spectrum nearly equally
            n_batch_middle = int(
                np.ceil(
                    (batch_edges_right[0] - batch_edges_left[-1])
                    / min(2**n_batch_left * min_batch_size, max_batch_size)
                )
            )
            batch_edges_middle = np.linspace(batch_edges_left[-1], batch_edges_right[0], n_batch_middle + 1).astype(
                int
            )[1:-1]

            batch_edges = np.concatenate([batch_edges_left, batch_edges_middle, batch_edges_right])

        return np.asarray(np.ceil(batch_edges), dtype=int)

    ###############################################################################
    ################################# QA Plotting #################################
    ###############################################################################

    @show_and_save
    def _plot_raw(self) -> Axes:
        from scipy.interpolate import interp1d
        from astropy.stats import mad_std

        cmap_sci = plt.cm.gray
        cmap_sci.set_bad("red")

        _, ax = plt.subplots(4, 1, figsize=(20, 10), constrained_layout=True)
        # Plot the original 2D spectrum
        ax[0].imshow(
            self.f_obs.Y,
            origin="lower",
            cmap=cmap_sci,
            vmin=np.nanpercentile(self.f_obs.y, 5),
            vmax=np.nanpercentile(self.f_obs.y, 95),
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )
        ax[1].imshow(
            self.f_sky_sub.Y,
            origin="lower",
            cmap=cmap_sci,
            vmin=np.nanpercentile(self.f_sky_sub.y, 5),
            vmax=np.nanpercentile(self.f_sky_sub.y, 95),
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )
        # Plot the 2D batched spectrum
        # Convert the pixel coordinate to a uniform pseudo-spectral coordinate
        spec_to_pseudo_spec = interp1d(
            self.spec,
            np.linspace(self.spec[0], self.spec[-1], len(self.spec)),
            kind="linear",
            fill_value="extrapolate",
        )
        norm = plt.Normalize(np.nanmin(self.f_batch_2d.y), np.nanmax(self.f_batch_2d.y))
        cmap = plt.cm.get_cmap("gray") if np.nanmean(self.f_host_1d.y) > 0 else plt.cm.get_cmap("gray_r")
        cmap.set_bad("red")

        shape_batch_2d = (len(self._spat_batch_2d_idx), len(self._spec_batch_2d_idx))
        delta_spat_typical = np.diff(self.spat).mean()
        delta_spec_typical = np.diff(self.spec).mean()
        for y in range(shape_batch_2d[1]):
            spec_min = spec_to_pseudo_spec(self.spec[self._spec_batch_2d_idx[y][0]] - 0.5 * delta_spec_typical)
            spec_max = spec_to_pseudo_spec(self.spec[self._spec_batch_2d_idx[y][-1]] + 0.5 * delta_spec_typical)
            ax[2].axvline(spec_max, color="0.8", linestyle=":", lw=2, zorder=100)
            ax[1].axvline(spec_max, color="0.8", linestyle=":", lw=2, zorder=100)
            for x in range(shape_batch_2d[0]):
                spat_min = self.spat[self._spat_batch_2d_idx[x][0]] - 0.5 * delta_spat_typical
                spat_max = self.spat[self._spat_batch_2d_idx[x][-1]] + 0.5 * delta_spat_typical

                c_raw = cmap(norm(self.f_batch_2d.Y[x, y]))
                ax[2].add_patch(
                    plt.Rectangle(
                        (spec_min, spat_min),
                        spec_max - spec_min,
                        spat_max - spat_min,
                        color=c_raw,
                    )
                )

        # Plot the 1D batched spectrum
        ax[-1].plot(np.arange(len(self.spec)) + 1, self.f_host_1d.y)
        ax[-1].plot(np.arange(len(self.spec)) + 1, self.f_sky_1d.y * self.f_host.shape[0], color="0.6", ls="--")
        ax[-1].fill_between(
            np.arange(len(self.spec)) + 1,
            self.f_host_1d.y - self.f_host_1d.yerr,
            self.f_host_1d.y + self.f_host_1d.yerr,
            alpha=0.5,
        )
        ax[-1].set_ylim(
            np.nanpercentile(self.f_host_1d.y, 5) - np.nanstd(self.f_host_1d.y),
            np.nanpercentile(self.f_host_1d.y, 95) + np.nanstd(self.f_host_1d.y),
        )

        # Titles
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Global\ Background\ Subtracted}$")
        ax[2].set_title(r"$\mathrm{Batched\ 2D\ Spectrum}$")
        ax[3].set_title(r"$\mathrm{Batched\ 1D\ Spectrum}$")

        # Labels
        ax[-1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        for ax_ in ax[:-1]:
            ax_.set_aspect("auto")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax_.set_xlim(self.spec[0], self.spec[-1])
            ax_.set_xticks([])
        for ax_ in ax[:-2]:
            ax_.axhline(-self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            ax_.axhline(self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            ax_.axhline(-self.host_wid / 2 + self.mask_offset, color="crimson", linestyle="-.", lw=3)
            ax_.axhline(self.host_wid / 2 + self.mask_offset, color="crimson", linestyle="-.", lw=3)
            ax_.axhline(self.sky_region[0], color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_region[1], color="darkgreen", linestyle="-.", lw=3)
            ax_.set_ylim(self.spat[0], self.spat[-1])

        ax[-1].set_ylabel(r"$\mathrm{Counts}$")
        major_tick_size = 500 if self.spec[-1] - self.spec[0] < 4000 else 1000
        original_ticks = (
            np.arange(np.ceil(self.spec[0] / major_tick_size), np.ceil(self.spec[-1] / major_tick_size))
            * major_tick_size
        )
        original_minor_ticks = (
            np.arange(np.ceil(self.spec[0] / major_tick_size * 5), np.ceil(self.spec[-1] / major_tick_size * 5))
            * major_tick_size
            / 5
        )
        # Convert the non-uniform spectral coordinate to pixel coordinate
        spec_to_pixel = interp1d(self.spec, np.arange(len(self.spec)) + 1, kind="linear", fill_value="extrapolate")
        transformed_ticks = spec_to_pixel(original_ticks)
        transformed_minor_ticks = spec_to_pixel(original_minor_ticks)

        ax[-1].set_xticks(transformed_ticks, minor=False)
        ax[-1].set_xticks(transformed_minor_ticks, minor=True)
        ax[-1].set_xticklabels([f"${tick:.0f}$" for tick in original_ticks])
        ax[-1].set_xlim(spec_to_pixel(self.spec[0]), spec_to_pixel(self.spec[-1]))

        # Mask the SN trace in the 2D spectrum
        ax[2].fill_between(
            ax[2].get_xlim(),
            -self.mask_wid / 2 + self.mask_offset,
            self.mask_wid / 2 + self.mask_offset,
            color="w",
            zorder=100,
        )

        # Adjust the ylim of the 1D spectrum
        ax[3].set_ylim(
            np.nanmedian(self.f_host_1d.y) - 5 * np.nanstd(self.f_host_1d.y),
            np.nanmedian(self.f_host_1d.y) + 5 * np.nanstd(self.f_host_1d.y),
        )

        return ax

    @show_and_save
    def _plot_host_profile_prior(self) -> Axes:
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(figsize=(6, len(self.f_host_batch_2d.spec) / 2), constrained_layout=True, sharex=True)

        raw = self.f_batch_2d.Y
        raw_err = self.f_batch_2d.Yerr
        prior = self.f_batch_prior.Y

        offset = (np.percentile(prior, 95) - np.percentile(prior, 5)) / 2

        for k, (r, r_err, p) in enumerate(zip(raw.T, raw_err.T, prior.T)):
            # c_raw = cmap(norm(k))
            c_raw = "k"
            ax.plot(self.f_batch_2d.spat, r - offset * k, color=c_raw, alpha=0.5, ls="--")
            ax.plot(self.f_batch_2d.spat, p - offset * k, color=c_raw, lw=2)
            ax.text(
                self.mask_offset,
                -offset * k + np.nanmedian(raw[:, 0]),
                f"${self.f_batch_2d.spec[k]:.0f}$",
                ha="center",
                va="center",
                fontsize=12,
                zorder=110,
                color=c_raw,
            )
        ylim = ax.get_ylim()
        # Sometimes the errors are riduculously large
        for k, (r, r_err, p) in enumerate(zip(raw.T, raw_err.T, prior.T)):
            ax.fill_between(
                self.f_batch_2d.spat, r + r_err - offset * k, r - r_err - offset * k, color=c_raw, alpha=0.5
            )
        ax.set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax.set_ylabel(r"$\mathrm{Counts + offset}$")
        ylim = ax.get_ylim()
        ax.fill_betweenx(
            y=[ylim[0] + offset, ylim[1] - offset],
            x1=-self.mask_wid / 2 + self.mask_offset,
            x2=self.mask_wid / 2 + self.mask_offset,
            color="w",
            zorder=100,
            alpha=0.75,
        )
        ax.set_ylim(ylim)
        ax.set_yticks([])
        return ax

    @show_and_save
    def _plot_host_profile_pred(self) -> Axes:
        if not hasattr(self, "_gp_2d"):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(figsize=(6, len(self.f_host_batch_2d.spec) / 2), constrained_layout=True, sharex=True)

        raw = self.dist_host_batch_2d.Y
        raw_err = self.dist_host_batch_2d.Yerr
        pred, pred_err = [
            p.reshape(self.dist_batch_2d.shape)
            for p in self._gp_2d.predict(X_test=self.dist_batch_2d.X, return_var=True)
        ]

        offset = max((np.percentile(pred, 95) - np.percentile(pred, 5)), np.nanmedian(raw_err) * 2)

        for k, (r, err, p, perr) in enumerate(zip(raw.T, raw_err.T, pred.T, pred_err.T)):
            # c_raw = cmap(norm(k))
            c_raw = "k"
            ax.plot(self.dist_host_batch_2d.spat, r - offset * k, "--x", color=c_raw, alpha=0.5)
            ax.fill_between(
                self.dist_host_batch_2d.spat, r + err - offset * k, r - err - offset * k, color=c_raw, alpha=0.5
            )
            ax.errorbar(self.dist_batch_2d.spat, p - offset * k, yerr=perr, fmt="-o", capsize=3, color=c_raw)
            ax.text(
                self.mask_offset,
                -offset * k + np.nanmedian(raw[:, 0]),
                f"${self.dist_batch_2d.spec[k]:.0f}$",
                ha="center",
                va="center",
                fontsize=12,
                zorder=110,
                color=c_raw,
            )
            ax.axhline(-offset * k, color=c_raw, ls="--", lw=1, alpha=0.25)
        ax.set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax.set_ylabel(r"$\mathrm{2D\ profile - prior}$")
        ylim = ax.get_ylim()
        ax.fill_betweenx(
            y=[ylim[0] + offset, ylim[1] - offset],
            x1=-self.mask_wid / 2 + self.mask_offset,
            x2=self.mask_wid / 2 + self.mask_offset,
            color="w",
            zorder=100,
            alpha=0.75,
        )
        ax.set_ylim(ylim)
        ax.set_yticks([])
        return ax

    @show_and_save
    def _plot_pred(self) -> Axes:
        if not hasattr(self, "_f_pred"):
            raise ValueError("Please model the host galaxy first.")

        cmap_sci = plt.cm.gray
        cmap_sci.set_bad("red")
        cmap_res = plt.cm.RdBu_r
        cmap_res.set_bad("0.5")

        source_params = dict(
            origin="lower",
            cmap=cmap_sci,
            aspect="auto",
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )
        f_res_Y = self.f_sky_sub.Y - self._f_pred.reshape(-1, self.shape[1])
        residual_params = dict(
            origin="lower",
            cmap=cmap_res,
            aspect="auto",
            vmin=np.nanmedian(self.f_host.yerr) * -1,
            vmax=np.nanmedian(self.f_host.yerr) * 1,
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )

        _, ax = plt.subplots(5, 1, figsize=(20, 12.5), sharex=True, sharey=True, constrained_layout=True)
        ax[0].imshow(
            self.f_sky_sub.Y,
            vmin=np.nanpercentile(self.f_sky_sub.y, 1),
            vmax=np.nanpercentile(self.f_sky_sub.y, 99),
            **source_params,
        )
        flux_is_positive = np.sign(np.median(self._f_1d_pred))
        ax[1].imshow(
            self._f_1d_pred.reshape(-1, self.shape[1]) * flux_is_positive,
            vmin=np.percentile(self._f_1d_pred * flux_is_positive, 1),
            vmax=np.percentile(self._f_1d_pred * flux_is_positive, 99),
            **source_params,
        )
        ax[2].imshow(
            self._f_2d_pred.reshape(-1, self.shape[1]) * flux_is_positive,
            vmin=np.percentile(self._f_2d_pred * flux_is_positive, 1),
            vmax=np.percentile(self._f_2d_pred * flux_is_positive, 99),
            **source_params,
        )
        ax[3].imshow(
            self._f_pred.reshape(-1, self.shape[1]),
            vmin=np.nanpercentile(self.f_sky_sub.y, 1),
            vmax=np.nanpercentile(self.f_sky_sub.y, 99),
            **source_params,
        )
        ax[-1].imshow(f_res_Y, **residual_params)
        for ax_ in ax:
            ax_.axhline(-self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            ax_.axhline(self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            # ax_.axhline(-self.host_wid / 2, color="crimson", linestyle="-.", lw=3)
            # ax_.axhline(self.host_wid / 2, color="crimson", linestyle="-.", lw=3)
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Model\ (1D)}$")
        ax[2].set_title(r"$\mathrm{Model\ (2D)}$")
        ax[3].set_title(r"$\mathrm{Model}$")
        ax[-1].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[-1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        ax[-1].set_ylim(-self.host_wid / 2 + self.mask_offset, self.host_wid / 2 + self.mask_offset)

        return ax
