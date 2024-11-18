# hostsub_gp/spectrum_model.py

__all__ = ["SpecModel"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

from tinygp import GaussianProcess

from ._plt_config import plt
from ._gp import _transform_unbound_to_bound, _transform_bound_to_unbound, _init_params, _print_params
from .gp import GP
from .host_model import HostProfile

from typing import Callable
from jax._src.typing import ArrayLike, Array

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
            jnp.nanmean((self.Yerr[mask, :] * w) ** 2, axis=0) / (jnp.nanmean(w) * mask.sum()) ** 2
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
        The instrumental pixel scale of the 2D spectrum (arcsec per pixel).
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
    sky_wid : tuple[float, float]
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
        sky_wid: tuple[float, float] = (5.0, 5.0),  # sky region
        batch_2d: tuple[int, int] = (2, 64),  # batch size for modeling slowing varying host profiles
        show: bool = False,
    ):
        self.pixel_scale = pixel_scale
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.slit_wid = slit_wid
        if slit_len is None:
            self.slit_len = spat.max() - spat.min()
        else:
            self.slit_len = slit_len
        self.position_angle = position_angle
        self.spat_resln = spat_resln
        self.spec_resln = spec_resln
        self.mask_wid = mask_wid
        self.sky_wid = sky_wid

        self.spat, self.spec = spat, spec
        self.shape = (len(spat), len(spec))

        # The 2D grids for the raw data
        self.f_obs = SpecWrapper(
            points=(spat, spec),
            values=dat,
            values_err=dat_err,
        )
        print(f"Loading the 2D spectrum with the shape: {self.f_obs.shape}")

        # The global sky region (spat < -sky_wid[0] * seeing) or (spat > sky_wid[1] * seeing)
        sky_left = spat < -self.spat_resln * self.sky_wid[0]
        sky_right = spat > self.spat_resln * self.sky_wid[1]
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

        # Mask the trace from the source (|spat| < seeing * mask_wid)
        if min(sky_wid) <= mask_wid:
            raise ValueError("sky_wid should be larger than mask_wid")
        host_left = self.spat < -self.mask_wid * self.spat_resln
        host_right = self.spat > self.mask_wid * self.spat_resln
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
        # TODO: adative batch size on the spectral direction
        self.batch_2d = batch_2d
        print(f"Batching the 2D galaxy spectrum (outside the mask) with the size: {batch_2d}")
        # Spatial batch (only for the host galaxy pixels outside the mask)
        if host_left.sum() > 0:
            spat_batch_2d_idx_left = np.array_split(np.arange(self.shape[0])[host_left], host_left.sum() // batch_2d[0])
        else:
            spat_batch_2d_idx_left = []
        if host_right.sum() > 0:
            spat_batch_2d_idx_right = np.array_split(
                np.arange(self.shape[0])[host_right], host_right.sum() // batch_2d[0]
            )
        else:
            spat_batch_2d_idx_right = []
        spat_batch_2d_idx = spat_batch_2d_idx_left + spat_batch_2d_idx_right
        if len(spat_batch_2d_idx) == 0:
            raise ValueError("No host galaxy pixels found.")
        # Spectral batch
        spec_batch_2d_idx = np.array_split(np.arange(self.shape[1]), self.shape[1] // batch_2d[1])

        # New coordinates: mean of the batch
        shape_batch_2d = (len(spat_batch_2d_idx), len(spec_batch_2d_idx))
        spat_batch_2d = jnp.asarray([self.spat[idx].mean() for idx in spat_batch_2d_idx])
        spec_batch_2d = jnp.asarray([self.spec[idx].mean() for idx in spec_batch_2d_idx])

        # New values: mean of the batch
        values_batch_2d = np.empty(shape_batch_2d)
        values_err_batch_2d = np.empty(shape_batch_2d)
        for x in range(shape_batch_2d[0]):
            for y in range(shape_batch_2d[1]):
                values_batch_2d[x, y] = np.nanmean(
                    (self.f_sky_sub.Y / self.f_host_1d.Y)[spat_batch_2d_idx[x], :][:, spec_batch_2d_idx[y]]
                )
                values_err_batch_2d[x, y] = np.nanmean(
                    (self.f_sky_sub.Yerr / self.f_host_1d.Y)[spat_batch_2d_idx[x], :][:, spec_batch_2d_idx[y]]
                )
        self.f_host_batch_2d = SpecWrapper(
            points=(spat_batch_2d, spec_batch_2d),
            values=values_batch_2d,
            values_err=values_err_batch_2d,
        )
        print("Batched 2D galaxy spectrum:", self.f_host_batch_2d.shape)

        if show:
            self._plot_raw()

    def model_host_prior(self, flts: str | list = "griz", show: bool = True):
        """
        Build the prior of the host galaxy using Gaussian Process regression.

        Parameters
        ----------
        imgs : list
            Names of the fits files of host galaxy images.
        flts : list
            Filters of the host galaxy images.
        """
        host_prof = HostProfile(flts=flts, spec2d=self)
        self.host_flux_prior = host_prof.model_host_profile_prior(show=show)

    def model_host(
        self,
        params_init: tuple[dict, dict] | list[dict],
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
        params_init = _init_params(params_init)
        if params_limit is None:
            params_limit = [None, None]
        else:
            params_limit = _init_params(params_limit, require_all=False, params_type="limit")

        # Default limits for the Gaussian Process parameters
        # TODO: modify the default limits with a configuration file
        # 1D spectrum of the host galaxy
        ## scale >= spectral resolution / 2.355
        ### here we use a composite kernel
        ### ExpSquared - slow variation (>> spectral resolution)
        ### Matern - narrow features (~ spectral resolution)
        params_limit_1d_default = _init_params(
            dict(
                log_scale=np.log10(
                    [[self.spec_resln / 2.355, self.spec_resln / 2.355], [self.spec_resln * 1e3, self.spec_resln * 2]]
                ),
            ),
            require_all=False,
            params_type="limit",
        )
        # 2D spatial profile & 1D spectrum of the host galaxy
        ## scale >= spatial resolution, spectral resolution / 2.355
        ## mean (i.e., deviation fromt the prior) is close to zero
        params_limit_2d_default = _init_params(
            dict(
                log_scale=np.log10([[self.spat_resln, self.spec_resln / 2.355], [1e5, 1e5]]),
                mean=[-1e-3, 1e-3],
            ),
            require_all=False,
            params_type="limit",
        )

        if optimization:
            # Fitting the 1D spectrum of the host galaxy
            print("Round 1: Fitting the 1D spectrum of the host galaxy")

            if params_limit[0] is None:
                params_limit[0] = params_limit_1d_default
            else:
                for key in params_limit[0]:
                    if key in params_limit_1d_default:
                        params_limit_1d_default.pop(key)
                params_limit[0] = {**params_limit_1d_default, **params_limit[0]}

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
                params_type="limit",
            )

            # Fitting the 2D spatial profile & 1D spectrum of the host galaxy jointly
            print("Round 2: Fitting the 2D spatial profile & 1D spectrum of the host galaxy jointly")

            if params_limit[1] is None:
                params_limit[1] = params_limit_2d_default
            else:
                for key in params_limit[1]:
                    if key in params_limit_2d_default:
                        params_limit_2d_default.pop(key)
                params_limit[1] = {**params_limit_2d_default, **params_limit[1]}

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
        self._f_host_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_host.X)
        # Predict the host galaxy flux on the entire 2D grids
        self._f_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_obs.X)

    def extract_sci(self):
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(self.f_sci_pred_1d.X, self.f_sci_pred_1d.y, color="tab:blue")
        ax.axhline(0, color="k", ls="--")
        ax.set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        ax.set_ylabel(r"$\mathrm{Counts}$")

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
        print("Initial parameters:")
        _print_params(params_init)
        print(f"Initial negative log-probability: {self._get_host_neg_log_probability(params_init):.1f}")
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
            kernel_type="ExpSquared",
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
        ) -> float:
            """
            Compute the negative log probability of the host galaxy model
            """
            gp_1d = GP(X=f_1d_X, yerr=f_1d_yerr, params=params_1d, kernel_type="composite").gp
            gp_2d = GP(X=f_2d_X, yerr=f_2d_yerr, params=params_2d, kernel_type="ExpSquared").gp
            log_prob_1d = gp_1d.log_probability(f_1d_y)
            log_prob_2d = gp_2d.log_probability(f_2d_y)

            y_host_1d = gp_1d.predict(y=f_1d_y, X_test=f_X[:, 1][:, None])
            y_host_2d = gp_2d.predict(y=f_2d_y - f_2d_mean, X_test=f_X) + f_mean
            y_host = y_host_1d * y_host_2d
            log_prob_obs = jnp.sum(jax.scipy.stats.norm.logpdf(y_host, f_y, f_yerr))

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
        y_host_1d = gp_1d.predict(y=self.f_host_1d.y, X_test=X[:, 1][:, None])
        y_host_2d = gp_2d.predict(
            y=self.f_host_batch_2d.y - self.host_flux_prior(gp_2d.X), X_test=X
        ) + self.host_flux_prior(X)

        return y_host_1d * y_host_2d

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

    ############################ QA Plotting ############################
    def _plot_raw(self):
        _, ax = plt.subplots(4, 1, figsize=(20, 10), constrained_layout=True, sharex=True)
        # Plot the original 2D spectrum
        ax[0].imshow(
            self.f_obs.Y,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_obs.y, 1),
            vmax=np.nanpercentile(self.f_obs.y, 99),
            extent=[self.spec.min(), self.spec.max(), self.spat.min(), self.spat.max()],
        )
        ax[1].imshow(
            self.f_sky_sub.Y,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_sky_sub.y, 1),
            vmax=np.nanpercentile(self.f_sky_sub.y, 99),
            extent=[self.spec.min(), self.spec.max(), self.spat.min(), self.spat.max()],
        )
        # Plot the 2D batched spectrum
        batch_size = (
            (self.spat[1] - self.spat[0]) * self.batch_2d[0],
            (self.spec[1] - self.spec[0]) * self.batch_2d[1],
        )
        norm = plt.Normalize(self.f_host_batch_2d.y.min(), self.f_host_batch_2d.y.max())
        cmap = plt.cm.get_cmap("gray")
        for k in range(len(self.f_host_batch_2d.X[:, 0])):
            c_raw = cmap(norm(self.f_host_batch_2d.y[k]))
            ax[2].add_patch(
                plt.Rectangle(
                    (
                        self.f_host_batch_2d.X[k, 1] - batch_size[1] / 2,
                        self.f_host_batch_2d.X[k, 0] - batch_size[0] / 2,
                    ),
                    batch_size[1],
                    batch_size[0],
                    color=c_raw,
                )
            )

        # Plot the 1D batched spectrum
        ax[-1].plot(self.f_host_1d.X.ravel(), self.f_host_1d.y)

        # Titles
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Global\ Background\ Subtracted}$")
        ax[2].set_title(r"$\mathrm{Batched\ 2D\ Spectrum}$")
        ax[3].set_title(r"$\mathrm{Batched\ 1D\ Spectrum}$")

        # Labels
        ax[-1].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        for ax_ in ax[:-1]:
            ax_.axhline(-self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            ax_.axhline(self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.set_aspect("auto")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax_.set_xlim(self.spec.min(), self.spec.max())
            ax_.set_ylim(self.spat.min(), self.spat.max())
        ax[-1].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_batch_pred(self):
        if not (hasattr(self, "_gp_1d") and hasattr(self, "_gp_2d")):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(5, 1, figsize=(20, 12.5), constrained_layout=True, sharex=True)
        # Plot the 2D batched spectrum
        batch_size = (
            (self.spat[1] - self.spat[0]) * self.batch_2d[0],
            (self.spec[1] - self.spec[0]) * self.batch_2d[1],
        )
        norm = plt.Normalize(self.f_host_batch_2d.y.min(), self.f_host_batch_2d.y.max())
        norm_residual = plt.Normalize(-1e-2, 1e-2)
        cmap = plt.cm.get_cmap("gray")
        cmap_residual = plt.cm.get_cmap("RdBu_r")
        pred_1d = self._gp_1d.predict(y=self.f_host_1d.y, X_test=self._gp_1d.X)
        pred_2d = self._gp_2d.predict(
            y=self.f_host_batch_2d.y - self.host_flux_prior(self._gp_2d.X), X_test=self._gp_2d.X
        ) + self.host_flux_prior(self._gp_2d.X)
        for k in range(len(self.f_host_batch_2d.X[:, 0])):
            c_raw = cmap(norm(self.f_host_batch_2d.y[k]))
            c_model = cmap(norm(pred_2d[k]))
            c_residual = cmap_residual(norm_residual(self.f_host_batch_2d.y[k] - pred_2d[k]))
            ax[0].add_patch(
                plt.Rectangle(
                    (
                        self.f_host_batch_2d.X[k, 1] - batch_size[1] / 2,
                        self.f_host_batch_2d.X[k, 0] - batch_size[0] / 2,
                    ),
                    batch_size[1],
                    batch_size[0],
                    color=c_raw,
                )
            )
            ax[1].add_patch(
                plt.Rectangle(
                    (
                        self.f_host_batch_2d.X[k, 1] - batch_size[1] / 2,
                        self.f_host_batch_2d.X[k, 0] - batch_size[0] / 2,
                    ),
                    batch_size[1],
                    batch_size[0],
                    color=c_model,
                )
            )
            ax[2].add_patch(
                plt.Rectangle(
                    (
                        self.f_host_batch_2d.X[k, 1] - batch_size[1] / 2,
                        self.f_host_batch_2d.X[k, 0] - batch_size[0] / 2,
                    ),
                    batch_size[1],
                    batch_size[0],
                    color=c_residual,
                )
            )

        # Plot the 1D batched spectrum
        ax[3].plot(self.f_host_1d.spec, self.f_host_1d.y)
        ax[3].fill_between(
            self.f_host_1d.spec,
            self.f_host_1d.y - self.f_host_1d.yerr,
            self.f_host_1d.y + self.f_host_1d.yerr,
            color="tab:blue",
            alpha=0.3,
        )
        ax[3].plot(self.f_host_1d.spec, pred_1d, "--k", lw=2)
        ax[4].plot(self.f_host_1d.spec, self.f_host_1d.y - pred_1d)
        ax[4].fill_between(
            self.f_host_1d.spec,
            -self.f_host_1d.yerr,
            +self.f_host_1d.yerr,
            color="tab:blue",
            alpha=0.3,
        )

        # Titles
        ax[0].set_title(r"$\mathrm{2D\ Spectrum}$")
        ax[1].set_title(r"$\mathrm{Model}$")
        ax[2].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[3].set_title(r"$\mathrm{1D\ Spectrum}$")
        ax[4].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")

        # Labels
        ax[4].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        for ax_ in ax[:3]:
            # ax_.axhline(-self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            # ax_.axhline(self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.set_aspect("auto")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax_.set_xlim(self.spec.min(), self.spec.max())
            ax_.set_ylim(self.spat.min(), self.spat.max())
        ax[3].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_pred(self):
        if not (hasattr(self, "_f_host_pred") and hasattr(self, "_f_pred")):
            raise ValueError("Please model the host galaxy first.")

        source_params = dict(
            origin="lower",
            cmap="gray",
            aspect="auto",
            vmin=np.nanpercentile(self.f_sky_sub.y, 1),
            vmax=np.nanpercentile(self.f_sky_sub.y, 99),
            extent=[self.spec.min(), self.spec.max(), self.spat.min(), self.spat.max()],
        )
        f_res_Y = self.f_sky_sub.Y - self._f_pred.reshape(-1, self.shape[1])
        residual_params = dict(
            origin="lower",
            cmap="RdBu_r",
            aspect="auto",
            vmin=np.nanmedian(self.f_host.yerr) * -1,
            vmax=np.nanmedian(self.f_host.yerr) * 1,
            extent=[self.spec.min(), self.spec.max(), self.spat.min(), self.spat.max()],
        )

        _, ax = plt.subplots(3, 1, figsize=(20, 7.5), sharex=True, sharey=True, constrained_layout=True)
        ax[0].imshow(self.f_sky_sub.Y, **source_params)
        ax[1].imshow(self._f_pred.reshape(-1, self.shape[1]), **source_params)
        ax[2].imshow(f_res_Y, **residual_params)
        for ax_ in ax:
            ax_.axhline(-self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            ax_.axhline(self.mask_wid * self.spat_resln, color="w", linestyle="--", lw=3)
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="darkgreen", linestyle="-.", lw=3)
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Model}$")
        ax[2].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[2].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")

        plt.show()
