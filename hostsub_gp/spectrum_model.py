# hostsub_gp/spectrum_model.py

__all__ = ["SpecModel"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

from tinygp import GaussianProcess

from ._plt import plt, MultipleLocator
from ._par import _transform_unbound_to_bound, _transform_bound_to_unbound, _init_params, _print_params
from .gp import GP
from .host_model import HostProfile

from typing import Callable
from jax._src.typing import ArrayLike, Array
from matplotlib.axes import Axes

import warnings


class SpecWrapper:
    """A wrapper for the 1D and 2D spectra."""

    def __init__(
        self, points: ArrayLike | tuple[ArrayLike, ArrayLike], values: ArrayLike, values_err: ArrayLike = None
    ):
        # Loading the coordinates
        # Input = spatial and spectral axes of the 2D spectrum
        if isinstance(points, tuple):
            self.spat, self.spec = jnp.array(points[0]), jnp.array(points[1])
            self.spec_img, self.spat_img = jnp.meshgrid(self.spec, self.spat)
            self.X = jnp.stack([self.spat_img.ravel(), self.spec_img.ravel()], axis=-1)
        # Input = spectral axis of the 1D spectrum
        else:
            if points.ndim != 1:
                raise ValueError("Invalid shape of the input coordinates.")
            self.spec = self.spec_img = jnp.array(points)
            self.X = self.spec[:, None]

        # Loading the values and errors
        if not (((values.ndim == 1) | (values.ndim == 2)) & (values.shape == self.spec_img.shape)):
            raise ValueError("Invalid shape of the input values.")
        if values_err is not None:
            if values.shape != values_err.shape:
                raise ValueError("Values and errors shape mismatch.")
        self.Y = jnp.array(values)
        self.Yerr = jnp.ones_like(values) if values_err is None else jnp.array(values_err)
        self.Y = jnp.where(jnp.isfinite(self.Yerr), self.Y, np.nan)
        self.Yerr = jnp.where(jnp.isfinite(self.Yerr), self.Yerr, np.nan)

        self.shape = self.Y.shape

        # Flatten the values and errors for GP
        if self.Y.ndim == 1:
            self.y = self.Y.copy()
            self.yerr = self.Yerr.copy()
        elif self.Y.ndim == 2:
            self.y = self.Y.ravel()
            self.yerr = self.Yerr.ravel()
        else:
            raise ValueError("Y shape error")

    def marginalize(self, mask: bool = None, margin_type: str = "mean", weights: str = None) -> "SpecWrapper":
        """
        Marginalize the 2D spectrum along the spatial axis to obtain the 1D spectrum.

        Parameters
        ----------
        mask : bool, optional
            Mask certain pixels in the marginalization.
        margin_type : str, optional
            Type of the marginalization: mean or sum. Default is mean.
        weights : str, optional
            Weights for the marginalization: None, ivar, or snr. Default is None.
            None: no weights
            ivar: inverse variance
            snr: signal-to-noise ratio squared

        Returns
        -------
        SpecWrapper
            The marginalized 1D spectrum.
        """
        if mask is None:
            mask = jnp.ones(self.shape[0], dtype=bool)
        elif mask.ndim != 1:
            raise ValueError("Invalid shape of the mask.")
        if mask.shape[0] != self.shape[0]:
            raise ValueError("Mask shape mismatch.")

        if weights is None:
            w = jnp.ones_like(self.Y[mask, :])
        elif weights == "ivar":
            w = self.Yerr[mask, :] ** -2
        elif weights == "snr":
            w = (self.Y[mask, :] / self.Yerr[mask, :]) ** 2
        else:
            raise ValueError("Invalid weights.")
        mean_value = jnp.nanmean(self.Y[mask, :] * w, axis=0) / jnp.nanmean(w, axis=0)
        mean_value_err = (
            (jnp.nanmean((self.Yerr[mask, :] * w) ** 2, axis=0) * mask.sum()) / (jnp.nanmean(w) * mask.sum()) ** 2
        ) ** 0.5

        if margin_type == "mean":
            return SpecWrapper(points=self.spec, values=mean_value, values_err=mean_value_err)
        elif margin_type == "sum":
            return SpecWrapper(points=self.spec, values=mean_value * mask.sum(), values_err=mean_value_err * mask.sum())


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
    mask_wid : float
        The width of the mask to mask the source trace (in spat_resln/seeing).
    sky_wid : float
        The width of the sky region (in spat_resln/seeing).
    batch_2d : tuple
        The batch size for modeling the slowly varying host profiles.
    f_obs : SpecWrapper
        The 2D spectrum of the observed data.
    f_sky : SpecWrapper
        The 1D spectrum of the global sky background (mean of the sky region).
    f_sky_sub : SpecWrapper
        The 2D spectrum of the sky-subtracted data.
    f_host : SpecWrapper
        The 2D spectrum of the sky-subtracted host galaxy (outside the mask).
    f_host_1d : SpecWrapper
        The 1D spectrum of the sky-subtracted host galaxy (outside the mask).
    f_host_batch_2d : SpecWrapper
        The batched 2D spectrum (batch size = batch_2d) of the sky-subtracted host galaxy (outside the mask).
    host : ArrayLike
        The mask of the host galaxy pixels.
    sky : ArrayLike
        The mask of the sky pixels.

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
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        mask_offset: float = 0.0,  # offset of the mask center (when the SN is not at the center)
        sky_wid: float = 10.0,  # sky region
        batch_2d: tuple[int, int] = (2, 64),  # batch size for modeling slowing varying host profiles
        show: bool = False,
        save: str = None,
    ):
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

        self.spat, self.spec = spat, spec
        self.shape = (len(spat), len(spec))

        # The 2D grids for the raw data
        self.f_obs = SpecWrapper(
            points=(spat, spec),
            values=dat,
            values_err=dat_err,
        )
        print(f"Loading the 2D spectrum with the shape: {self.f_obs.shape}")

        # The width of the sky region
        # Adjust the sky width to the nearest integer multiple of the pixel scale
        # Add 0.5 so the sky boundary is at the edge of the pixel
        self.sky_wid = (jnp.round(sky_wid * spat_resln / 2 / pixel_scale) * 2 + 1) * pixel_scale
        print(f"Sky width: {self.sky_wid:.2f} arcsec = {self.sky_wid / pixel_scale:.0f} pixels")

        # The global sky region (|spat| > sky_wid / 2)
        sky_left = spat < -self.sky_wid / 2
        sky_right = spat > self.sky_wid / 2
        self.sky = sky_left | sky_right
        if np.nansum(self.sky) / self.sky.ravel().size < 0.1:
            warnings.warn(r"Sky region is < 10% of the overall pixels.")

        # Estimate the global sky background (sky + host): mean of the sky region along the spectral direction
        print(f"Estimating the global sky background")
        self.f_sky = SpecWrapper(
            points=(spat[self.sky], spec),
            values=self.f_obs.Y[self.sky, :],
            values_err=self.f_obs.Yerr[self.sky, :],
        )
        self.f_sky_1d = self.f_sky.marginalize(margin_type="mean")
        self.f_sky_sub = SpecWrapper(
            points=(spat, spec),
            values=self.f_obs.Y - np.tile(self.f_sky_1d.Y, reps=(len(spat), 1)),
            values_err=np.sqrt(self.f_obs.Yerr**2 + np.tile(self.f_sky_1d.Yerr, reps=(len(spat), 1)) ** 2),
        )

        # Mask the trace from the source (|spat| < mask_wid / 2)
        # Adjust the mask width to the nearest integer multiple of the pixel scale
        # Add 0.5 so the mask boundary is at the edge of the pixel
        self.mask_wid = (jnp.round(mask_wid * spat_resln / 2 / pixel_scale) * 2 + 1) * pixel_scale
        self.mask_offset = jnp.round(mask_offset / pixel_scale) * pixel_scale
        print(
            f"Masking the source trace with the width: {self.mask_wid:.2f} arcsec = {self.mask_wid / pixel_scale:.0f} pixels"
        )
        if sky_wid <= mask_wid:
            raise ValueError("sky_wid should be larger than mask_wid")
        host_left = self.spat < -self.mask_wid / 2 + self.mask_offset
        host_right = self.spat > self.mask_wid / 2 + self.mask_offset
        self.host = host_left | host_right
        self.f_host = SpecWrapper(
            points=(self.spat[self.host], spec),
            values=self.f_sky_sub.Y[self.host, :],
            values_err=self.f_sky_sub.Yerr[self.host, :],
        )

        # The 1D grids for the sky-subtracted host galaxy spectra: sum along the spatial direction outside the mask
        # Central wavelength in each row: spec
        # Total flux in each row: weighted sum of the flux in each row
        print(f"Obtaining the sky-subtracted 1D galaxy spectrum (outside the mask)")
        self.f_host_1d = self.f_host.marginalize(margin_type="sum")

        # The batched 2D grids for the normalized host galaxy spatial profiles
        self.batch_2d = batch_2d
        print(f"Batching the 2D galaxy spectrum (outside the mask) with the size: {batch_2d}")
        # Spatial batch (only for the host galaxy pixels outside the mask)
        if host_left.sum() > 0:
            self._spat_batch_2d_idx_left = np.array_split(
                np.arange(self.shape[0])[host_left], host_left.sum() // batch_2d[0]
            )
        else:
            self._spat_batch_2d_idx_left = []
        if host_right.sum() > 0:
            self._spat_batch_2d_idx_right = np.array_split(
                np.arange(self.shape[0])[host_right], host_right.sum() // batch_2d[0]
            )
        else:
            self._spat_batch_2d_idx_right = []
        if len(self._spat_batch_2d_idx_left + self._spat_batch_2d_idx_right) == 0:
            raise ValueError("No host galaxy pixels found.")
        if (~self.host).sum() > 0:
            self._spat_batch_2d_idx_sci = np.array_split(np.arange(self.shape[0])[~self.host], (~self.host).sum())
        else:
            raise ValueError("No pixels within the mask.")
        self._spat_batch_2d_idx = (
            self._spat_batch_2d_idx_left + self._spat_batch_2d_idx_sci + self._spat_batch_2d_idx_right
        )
        host_batch_2d = (jnp.arange(len(self._spat_batch_2d_idx)) < len(self._spat_batch_2d_idx_left)) | (
            jnp.arange(len(self._spat_batch_2d_idx)) >= len(self._spat_batch_2d_idx_left + self._spat_batch_2d_idx_sci)
        )

        # Spectral batch
        self._spec_batch_2d_idx = self._get_spec_batches(show=show, save=save.replace(".pdf", "_host_emission.pdf"))

        # New coordinates: mean of the batch
        shape_batch_2d = (len(self._spat_batch_2d_idx), len(self._spec_batch_2d_idx))
        spat_batch_2d = jnp.asarray([self.spat[idx].mean() for idx in self._spat_batch_2d_idx])
        spec_batch_2d = jnp.asarray([self.spec[idx].mean() for idx in self._spec_batch_2d_idx])

        # New values: mean of the batch
        values_batch_2d = np.empty(shape_batch_2d)
        values_err_batch_2d = np.empty(shape_batch_2d)
        for x in range(shape_batch_2d[0]):
            for y in range(shape_batch_2d[1]):
                values_batch_2d[x, y] = np.nanmean(
                    (self.f_sky_sub.Y / self.f_host_1d.Y)[self._spat_batch_2d_idx[x], :][:, self._spec_batch_2d_idx[y]]
                )
                values_err_batch_2d[x, y] = np.sqrt(
                    np.nanmean(
                        (self.f_sky_sub.Yerr / self.f_host_1d.Y)[self._spat_batch_2d_idx[x], :][
                            :, self._spec_batch_2d_idx[y]
                        ]
                        ** 2
                    )
                    * (len(self._spat_batch_2d_idx[x]) * len(self._spec_batch_2d_idx[y]))
                ) / (len(self._spat_batch_2d_idx[x]) * len(self._spec_batch_2d_idx[y]))

        self.f_host_batch_2d = SpecWrapper(
            points=(spat_batch_2d[host_batch_2d], spec_batch_2d),
            values=values_batch_2d[host_batch_2d, :],
            values_err=values_err_batch_2d[host_batch_2d, :],
        )
        self.f_batch_2d = SpecWrapper(
            points=(spat_batch_2d, spec_batch_2d),
            values=values_batch_2d,
            values_err=values_err_batch_2d,
        )
        print("Batched 2D galaxy spectrum:", self.f_host_batch_2d.shape)

        self._plot_raw()
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save, bbox_inches="tight")

    def model_host_prior(self, **kwargs):
        """
        Build the prior of the host galaxy using Gaussian Process regression.
        """
        host_prof = HostProfile(spec2d=self)
        self.host_flux_prior = host_prof.model_host_profile_prior(**kwargs)

    def model_host(
        self,
        params_init: tuple[dict, dict] | list[dict] = None,
        params_limit: tuple[dict, dict] | list[dict] = None,
        optimization: bool = False,
        sampling: bool = False,
        optimization_kwargs: dict = {},
        sampling_kwargs: dict = {},
    ):
        """
        Model the host galaxy using Gaussian Process regression.

        Parameters
        ----------
        params_init : dict
            Initial parameters for optimization.
        optimization : bool, optional (default: False)
            Whether to optimize the model with the jaxopt.ScipyMinimize solver.
        sampling : bool, optional (default: False)
            Whether to sample the model with numpyro.
        """

        # Make sure the host flux prior is built
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please build the host flux prior first.")

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
            # Fitting the 1D spectrum of the host galaxy
            print("Round 1: Fitting the 1D spectrum of the host galaxy")

            params_limit[0] = self._set_params_limit(params_limit[0], ndim=1)

            params_1d = GP(
                X=self.f_host_1d.X,
                y=self.f_host_1d.y,
                yerr=self.f_host_1d.yerr,
                params_init=params_init[0],
                params_limit=params_limit[0],
                kernel_type="composite",
                optimization=True,
            ).params

            # Update the initial parameters with the 1D results
            params_init[0] = params_1d
            # Update the limits for parameters with the 1D results (within +/- 0.1%)
            params_limit[0] = _init_params(
                {
                    key: (
                        params_1d[key] - 1e-3 * np.abs(params_1d[key]),
                        params_1d[key] + 1e-3 * np.abs(params_1d[key]),
                    )
                    for key in params_1d
                },
                require_all=False,
            )

            # Fitting the 2D spatial profile & 1D spectrum of the host galaxy jointly
            print("Round 2: Fitting the 2D spatial profile & 1D spectrum of the host galaxy jointly")

            params_limit[1] = self._set_params_limit(params_limit[1], ndim=2)

            self.gp_params = self._model_host_optimization(
                params_init=tuple(params_init),
                params_limit=tuple(params_limit),
                **optimization_kwargs,
            )

        if sampling:
            self.inf_data = self._model_host_sampling(params_init=params_init, **sampling_kwargs)
            # TODO: self.gp_params

        if not optimization and not sampling:
            self.gp_params = params_init

        self._gp_1d, self._gp_2d = self._build_host_gp(params=self.gp_params)
        # Predict the host galaxy flux outside the mask
        _, _, self._f_host_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_host.X)
        # Predict the host galaxy flux on the entire 2D grids
        self._f_1d_pred, self._f_2d_pred, self._f_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_obs.X)

    def extract_sci(self) -> Axes:  # TODO: adopt the extraction method of pypeit
        """
        Extract the science spectrum.
        """
        if not hasattr(self, "_f_pred"):
            raise AttributeError("Please model the host galaxy first.")
        # Subtract the host galaxy model
        self.f_sci_pred = SpecWrapper(
            points=(self.spat[~self.host], self.spec),
            values=(self.f_sky_sub.Y - self._f_pred.reshape(self.shape))[~self.host, :],
            values_err=self.f_sky_sub.Yerr[~self.host, :],
        )
        self.f_sci_pred_1d = SpecWrapper(
            points=self.spec,
            values=np.nanmean(self.f_sci_pred.Y * self.f_sci_pred.Yerr**-2, axis=0)
            / np.nanmean(self.f_sci_pred.Yerr**-2, axis=0)
            * (~self.host).sum(),
            values_err=np.nanmean(self.f_sci_pred.Yerr**-2, axis=0) ** -0.5,
        )
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
                log_amp=0.0,
                log_scale=(
                    np.log10(self.spec_resln),  # Spatial scale ~ seeing
                    2,  # Spectral scale ~ 100 Angstrom
                ),
                mean=0.0,
                amp_line=1.0,  # Covariance within the host lines = covariance outside the host lines
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
                        [self.spat_resln / 2.355, self.spec_resln / 2.355],
                        [self.spat_resln * large_scale, self.spec_resln * large_scale],
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

    def _model_host_optimization(
        self, params_init: tuple[dict, dict], params_limit: tuple[dict, dict], **kwargs
    ) -> dict:
        """
        Optimize the Gaussian process model of the host using jaxopt.ScipyMinimize solver.

        Parameters
        ----------
        params_init : dict
            Initial parameters for optimization.
        params_limit : dict
            Limits for the Gaussian Process parameters.

        Returns
        -------
        gp_params : dict
            The optimized parameters for the Gaussian Process model.
        """
        params_init_unbound = _transform_bound_to_unbound(params_init, params_limit)
        print("Optimizing the host galaxy model...")
        print(f"Initial negative log-probability: {self._get_host_neg_log_probability(params_init):.1f}")
        if ~np.isfinite(self._get_host_neg_log_probability(params_init)):
            print("Initial parameters:")
            _print_params(params_init)
            print("Parameter limits:")
            _print_params(params_limit)
            print("Initial unbound parameters:")
            _print_params(params_init_unbound)
            raise ValueError("Invalid initial parameters: please check the limits.")

        solver = jaxopt.ScipyMinimize(fun=self._get_host_neg_log_probability, **kwargs)
        soln = solver.run(params_init_unbound, params_limit)
        if soln.state.status != 0:
            warnings.warn(f"Optimization failed with status {soln.state.status}.")
        params = _transform_unbound_to_bound(soln.params, params_limit)
        print("Final parameters:")
        _print_params(params)
        print(f"Final negative log-probability: {soln.state.fun_val:.1f}")
        return params

    def _model_host_sampling(self, params_init: dict = None, **kwargs):
        """
        Perform host sampling using MCMC.

        Parameters
        ----------
        num_chains : int
            The number of MCMC chains to run.
        num_samples : int
            The number of samples to draw from each chain.
        num_warmup : int
            The number of warmup steps for each chain.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        samples : arviz.InferenceData
        """

        raise NotImplementedError("Sampling is not implemented yet.")

    def _build_host_gp(
        self, params: tuple[dict, dict], params_limit: tuple[dict, dict] = (None, None)
    ) -> tuple[GaussianProcess, GaussianProcess]:
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
        tuple[GaussianProcess, GaussianProcess]
            tinygp.GaussianProcess objects for the 1D and 2D host galaxy.
        """
        params_1d, params_2d = _init_params(params)
        try:
            params_limit_1d, params_limit_2d = _init_params(params_limit, require_all=False, params_type="limit")
        except:
            print(params_limit)
        gp_1d = GP(
            X=self.f_host_1d.X,
            y=self.f_host_1d.y,
            yerr=self.f_host_1d.yerr,
            params=params_1d,
            params_limit=params_limit_1d,
            kernel_type="composite",
        ).gp
        gp_2d = GP(
            X=self.f_host_batch_2d.X,
            y=self.f_host_batch_2d.y,
            yerr=self.f_host_batch_2d.y,
            params=params_2d,
            params_limit=params_limit_2d,
            kernel_type="EmissionLine",
            emission_lines=self.emission_lines,
        ).gp

        return gp_1d, gp_2d

    def _get_host_neg_log_probability(
        self, params: tuple[dict, dict], params_limit: tuple[dict, dict] = (None, None)
    ) -> float:
        """
        Calculate the negative log probability of the host flux given the parameters.

        Parameters
        ----------
        params : tuple[dict, dict]
            A tuple of parameters for the 1D and 2D Gaussian Processes.
        params_limit : dict, optional
            Limits for the Gaussian Process parameters.

        Returns
        -------
        float
            The negative log probability of the host flux.
        """
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
            f_2d_X: Array,
            f_2d_y: Array,
            f_2d_yerr: Array,
            f_2d_mean: Array,
            f_mean: Array,
            emission_lines: Array,
        ) -> float:
            """
            Compute the negative log probability of the host galaxy model
            """
            gp_1d = GP(X=f_1d_X, yerr=f_1d_yerr, params=params_1d, kernel_type="composite").gp
            gp_2d = GP(
                X=f_2d_X,
                yerr=f_2d_yerr,
                params=params_2d,
                kernel_type="EmissionLine",
                emission_lines=emission_lines,
            ).gp
            log_prob_1d = gp_1d.log_probability(f_1d_y)
            log_prob_2d = gp_2d.log_probability(f_2d_y)

            y_host_1d = gp_1d.predict(y=f_1d_y, X_test=f_X[:, 1][:, None])
            y_host_2d = gp_2d.predict(y=f_2d_y - f_2d_mean, X_test=f_X) + f_mean
            y_host = y_host_1d * y_host_2d
            log_prob_obs = jnp.sum(jax.scipy.stats.norm.logpdf(y_host, f_y, f_yerr))

            # jax.debug.print("1D log-probability: {}", log_prob_1d)
            # jax.debug.print("2D log-probability: {}", log_prob_2d)
            # jax.debug.print("Observed log-probability: {}", log_prob_obs)

            return -(log_prob_1d + log_prob_2d + log_prob_obs)

        # Only include finite values in the observation
        mask_obs = np.isfinite(self.f_host.y)

        return _neg_log_probability(
            params_1d=params_1d,
            params_2d=params_2d,
            f_X=self.f_host.X[mask_obs],
            f_y=self.f_host.y[mask_obs],
            f_yerr=self.f_host.yerr[mask_obs],
            f_1d_X=self.f_host_1d.X,
            f_1d_y=self.f_host_1d.y,
            f_1d_yerr=self.f_host_1d.yerr,
            f_2d_X=self.f_host_batch_2d.X,
            f_2d_y=self.f_host_batch_2d.y,
            f_2d_yerr=self.f_host_batch_2d.yerr,
            f_2d_mean=self.host_flux_prior(self.f_host_batch_2d.X),
            f_mean=self.host_flux_prior(self.f_host.X[mask_obs]),
            emission_lines=self.emission_lines,
        )

    def _get_pred(self, gp_1d: GaussianProcess, gp_2d: GaussianProcess, X: Array) -> Array:
        """
        Get the predicted host galaxy flux on the given grids.

        Parameters
        ----------
        gp_1d : GaussianProcess
            The 1D Gaussian Process - the 1D spectrum of the host.
        gp_2d : GaussianProcess
            The 2D Gaussian Process - the spatial profile of the host.
        X : Array
            The 2D grids to make the prediction.

        Returns
        -------
        Array
            The predicted host galaxy flux.
        """
        y_1d = gp_1d.predict(y=self.f_host_1d.y, X_test=X[:, 1][:, None])
        y_2d = gp_2d.predict(y=self.f_host_batch_2d.y - self.host_flux_prior(gp_2d.X), X_test=X) + self.host_flux_prior(
            X
        )

        return y_1d, y_2d, y_1d * y_2d

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
    ############################# Adaptive Batch Size #############################
    ###############################################################################

    def _get_spec_batches(self, **kwargs) -> list[list[int]]:
        """
        Get the batch indices for the spectral direction.

        Returns
        -------
        list[list[int]]
            The indices of the spectral batches.
        """
        host_emission = self._find_host_emission(**kwargs)
        self.emission_lines = self.spec[host_emission]
        print(f"Emission lines found at: {self.emission_lines}")

        host_emission = np.concatenate([[0], host_emission, [self.spec.size - 1]])
        batch_edges = []
        for i in range(len(host_emission) - 1):
            batch_edges.extend(self._find_batch_edges(left=host_emission[i], right=host_emission[i + 1]))

        batch_idx = []
        for i in range(len(batch_edges) - 1):
            batch_idx.append(np.arange(batch_edges[i], batch_edges[i + 1]))

        return batch_idx

    def _find_host_emission(self, p_value: float = 1e-8, kernel_wid: int = None, **kwargs) -> Array:
        """
        Find the edges of the host galaxy emission using the 1D spectrum.

        Parameters
        ----------
        p_value : float, optional (default: 1e-5)
            The p-value for emission line detection.
        kernel_wid : int, optional (default: None)
            The width of the kernel for smoothing the profile.

        Returns
        -------
        Array
            The indices of the host galaxy emission.
        """
        from scipy.signal import find_peaks
        from scipy.stats import chi2
        from astropy.stats import mad_std

        # Define the kernel for smoothing the standard deviation of the galaxy spatial profile
        if kernel_wid is None:
            kernel_wid = int(self.spec_resln / jnp.diff(self.spec).min()) + 1

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
        distinct_prof, _ = find_peaks(prof_diff, height=chi2.ppf(1 - p_value, prof.shape[0]))
        host_lines = jnp.argwhere(f_lines > mad_std(f_lines) * 5).ravel()

        emission_lines = []
        for line in distinct_prof:
            host_lines_close = np.where(np.abs(host_lines - line) < kernel_wid)
            if host_lines_close[0].size > 0:
                emission_lines.append(int(np.mean(host_lines[host_lines_close])))
        
        # Remove duplicates
        emission_lines = np.unique(emission_lines)

        _, ax = plt.subplots(2, 1, figsize=(20, 5), sharex=True, constrained_layout=True)
        ax[0].plot(self.spec, f_lines, color="tab:blue")
        ax[0].axhline(mad_std(f_lines) * 5, color="0.5", ls="--")
        ax[0].set_ylabel(r"$|f - f_\mathrm{cont}|$")
        ax[1].plot(self.spec, prof_diff, color="tab:blue")
        ax[1].axhline(chi2.ppf(1 - p_value, prof.shape[0]), color="0.5", ls="--")
        for line in emission_lines:
            ax[0].axvline(self.spec[line], color="tab:red", ls=":")
            ax[1].axvline(self.spec[line], color="tab:red", ls=":")
        ax[1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        ax[1].set_ylabel(r"$\chi^2$")
        ax[1].set_yscale("log")

        save = kwargs.get("save", None)
        show = kwargs.get("show", False)
        if save is not None:
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()

        return jnp.asarray(emission_lines, dtype=int)

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

        min_batch_size = 2 * (int(self.spec_resln / np.diff(self.spec).min()) + 1) + 1
        max_batch_size = self.batch_2d[1]

        def check_spectrum_length(left, right):
            if right - left < min_batch_size * 2:
                raise ValueError("The spectrum is too short for the batch size")

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

            check_spectrum_length(left, right)

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

        return np.asarray(batch_edges, dtype=int)

    ###############################################################################
    ################################# QA Plotting #################################
    ###############################################################################

    def _plot_raw(self) -> Axes:
        from scipy.interpolate import interp1d

        _, ax = plt.subplots(4, 1, figsize=(20, 10), constrained_layout=True)
        # Plot the original 2D spectrum
        ax[0].imshow(
            self.f_obs.Y,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_obs.y, 1),
            vmax=np.nanpercentile(self.f_obs.y, 99),
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )
        ax[1].imshow(
            self.f_sky_sub.Y,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_sky_sub.y, 1),
            vmax=np.nanpercentile(self.f_sky_sub.y, 99),
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
        norm = plt.Normalize(self.f_batch_2d.y.min(), self.f_batch_2d.y.max())
        cmap = plt.cm.get_cmap("gray") if np.mean(self.f_host_1d.y) > 0 else plt.cm.get_cmap("gray_r")

        shape_batch_2d = (len(self._spat_batch_2d_idx), len(self._spec_batch_2d_idx))
        delta_spat_typical = np.diff(self.spat).mean()
        delta_spec_typical = np.diff(self.spec).mean()
        for y in range(shape_batch_2d[1]):
            spec_min = spec_to_pseudo_spec(self.spec[self._spec_batch_2d_idx[y][0]] - 0.5 * delta_spec_typical)
            spec_max = spec_to_pseudo_spec(self.spec[self._spec_batch_2d_idx[y][-1]] + 0.5 * delta_spec_typical)
            ax[2].axvline(spec_max, color="0.8", linestyle=":", lw=2, zorder=100)
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
        ax[-1].fill_between(
            np.arange(len(self.spec)) + 1,
            self.f_host_1d.y - self.f_host_1d.yerr,
            self.f_host_1d.y + self.f_host_1d.yerr,
            alpha=0.5,
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
            ax_.set_ylim(self.spat[0], self.spat[-1])
            ax_.set_xticks([])
        for ax_ in ax[:-2]:
            ax_.axhline(-self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            ax_.axhline(self.mask_wid / 2 + self.mask_offset, color="w", linestyle="--", lw=3)
            ax_.axhline(-self.sky_wid / 2, color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_wid / 2, color="darkgreen", linestyle="-.", lw=3)

        ax[-1].set_ylabel(r"$\mathrm{Counts}$")
        major_tick_size = 500 if self.spec[-1] - self.spec[0] < 4000 else 5000
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

        return ax

    def _plot_host_profile_prior(self) -> Axes:
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(figsize=(6, len(self.f_host_batch_2d.spec) / 3), constrained_layout=True, sharex=True)
        # norm = plt.Normalize(0, len(self.f_batch_2d.spec))
        # cmap = plt.cm.get_cmap("gray")

        raw = self.f_batch_2d.Y
        prior = self.host_flux_prior(self.f_batch_2d.X).reshape(self.f_batch_2d.shape)

        offset = (prior.max() - prior.min()) / 3

        for k, (r, p) in enumerate(zip(raw.T, prior.T)):
            # c_raw = cmap(norm(k))
            c_raw = "k"
            ax.plot(self.f_batch_2d.spat, r - offset * k, color=c_raw, alpha=0.5, ls="--")
            ax.plot(self.f_batch_2d.spat, p - offset * k, color=c_raw, lw=2)
            ax.text(
                self.mask_offset,
                -offset * (k - 1),
                f"${self.f_batch_2d.spec[k]:.0f}$",
                ha="center",
                va="center",
                fontsize=12,
                zorder=110,
                color=c_raw,
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

    def _plot_host_profile_pred(self) -> Axes:
        if not hasattr(self, "_gp_2d"):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(figsize=(6, len(self.f_host_batch_2d.spec) / 3), constrained_layout=True, sharex=True)
        # norm = plt.Normalize(0, len(self.f_batch_2d.spec))
        # cmap = plt.cm.get_cmap("coolwarm")

        raw = self.f_batch_2d.Y - self.host_flux_prior(self.f_batch_2d.X).reshape(self.f_batch_2d.shape)
        pred = (
            self._gp_2d.predict(
                y=self.f_host_batch_2d.y - self.host_flux_prior(self._gp_2d.X), X_test=self.f_batch_2d.X
            )
        ).reshape(self.f_batch_2d.shape)

        offset = (pred.max() - pred.min()) / 3

        for k, (r, p) in enumerate(zip(raw.T, pred.T)):
            # c_raw = cmap(norm(k))
            c_raw = "k"
            ax.plot(self.f_batch_2d.spat, r - offset * k, color=c_raw, ls="--")
            ax.plot(self.f_batch_2d.spat, p - offset * k, color=c_raw, lw=2)
            ax.text(
                self.mask_offset,
                -offset * k,
                f"${self.f_batch_2d.spec[k]:.0f}$",
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

    def _plot_pred(self) -> Axes:
        if not (hasattr(self, "_f_host_pred") and hasattr(self, "_f_pred")):
            raise ValueError("Please model the host galaxy first.")

        source_params = dict(
            origin="lower",
            cmap="gray",
            aspect="auto",
            extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
        )
        f_res_Y = self.f_sky_sub.Y - self._f_pred.reshape(-1, self.shape[1])
        residual_params = dict(
            origin="lower",
            cmap="RdBu_r",
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
        ax[1].imshow(self._f_1d_pred.reshape(-1, self.shape[1]) * flux_is_positive, **source_params)
        ax[2].imshow(self._f_2d_pred.reshape(-1, self.shape[1]) * flux_is_positive, **source_params)
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
            ax_.axhline(-self.sky_wid / 2, color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_wid / 2, color="darkgreen", linestyle="-.", lw=3)
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Model\ (1D)}$")
        ax[2].set_title(r"$\mathrm{Model\ (2D)}$")
        ax[3].set_title(r"$\mathrm{Model}$")
        ax[-1].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[-1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")

        return ax
