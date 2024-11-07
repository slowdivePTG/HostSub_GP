# hostsub_gp/spec.py

__all__ = ["Spec2D"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt
from functools import partial

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from tinygp import GaussianProcess

from ._plt_config import plt
from .gp import _gp
from .host_prof import HostProfile

from typing import Callable
from jax._src.typing import ArrayLike, Array

import warnings


class SpecWrapper:
    def __init__(
        self, points: ArrayLike | tuple[ArrayLike, ArrayLike], values: ArrayLike, values_err: ArrayLike = None
    ):
        # Loading the coordinates
        # Input = spatial and spectral axes of the 2D spectrum
        if isinstance(points, tuple):
            self.spat, self.spec = points
            self.spec_img, self.spat_img = jnp.meshgrid(self.spec, self.spat)
            self.X = jnp.stack([self.spat_img.ravel(), self.spec_img.ravel()], axis=-1)
        # Input = spectral axis of the 1D spectrum
        else:
            if points.ndim != 1:
                raise ValueError("Invalid shape of the input coordinates.")
            self.spec = self.spec_img = jnp.asarray(points)
            self.X = self.spec[:, None]

        # Loading the values and errors
        if not (((values.ndim == 1) | (values.ndim == 2)) & (values.shape == self.spec_img.shape)):
            raise ValueError("Invalid shape of the input values.")
        if values_err is not None:
            if values.shape != values_err.shape:
                raise ValueError("Values and errors shape mismatch.")
        self.Y = jnp.asarray(values)
        self.Yerr = jnp.ones_like(values) if values_err is None else jnp.asarray(values_err)

        self.shape = self.Y.shape

        # Flatten the values and errors for GP
        if self.Y.ndim == 1:
            self.y = self.Y
            self.yerr = self.Yerr
        elif self.Y.ndim == 2:
            self.y = self.Y.ravel()
            self.yerr = self.Yerr.ravel()
        else:
            raise ValueError("Y shape error")


class Spec2DBase:
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
        slit_len: float = 10.0,  # arcsec
        position_angle: float = None,  # degree
        spat_resln: float = 1.0,  # arcsec, FWHM/seeing
        spec_resln: float = 7.5,  # LRIS, 1'' slit
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        sky_wid: tuple = (5.0, 5.0),  # sky region
        batch_2d: tuple = (2, 64),  # batch size for modeling slowing varying host profiles
        show: bool = False,
    ):
        self.pixel_scale = pixel_scale
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.slit_wid = slit_wid
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
        self.f_sky_obs = SpecWrapper(
            points=(spat[self.sky], spec),
            values=self.f_obs.Y[self.sky, :],
            values_err=self.f_obs.Yerr[self.sky, :],
        )
        self.f_sky_obs_1d = SpecWrapper(
            points=spec,
            values=np.nanmean(self.f_sky_obs.Y * self.f_sky_obs.Yerr**-2, axis=0)
            / np.nanmean(self.f_sky_obs.Yerr**-2, axis=0),
            values_err=np.nanmean(self.f_sky_obs.Yerr**-2, axis=0) ** -0.5,
        )
        # Model the global sky background using 1D GP
        # The scale should be larger than the spectral resolution
        # _gp_sky = _gp(
        #     X=self.f_sky_obs_1d.X,
        #     y=self.f_sky_obs_1d.y,
        #     yerr=self.f_sky_obs_1d.yerr,
        #     params_init=dict(log_jitter=-2.0, log_amp=3.0, log_scale=np.log10(self.spec_resln), mean=0.0),
        #     params_limits=dict(log_scale=np.log10([0.85 * self.spec_resln, 2.5 * self.spec_resln])),
        #     optimization=True,
        #     verbose=True,
        # )
        # self._gp_sky_params, self._gp_sky = _gp_sky.params, _gp_sky.gp
        # Subtract the global sky background
        # self.f_sky_sub = SpecWrapper(
        #     points=(spat, spec),
        #     values=self.f_obs.Y
        #     - jnp.tile(self._gp_sky.predict(y=self.f_sky_obs_1d.y, X_test=self.f_sky_obs_1d.X), reps=(len(spat), 1)),
        #     values_err=jnp.sqrt(self.f_obs.Yerr**2 + jnp.tile(self.f_sky_obs_1d.yerr, reps=(len(self.spat), 1)) ** 2),
        # )
        self.f_sky_sub = SpecWrapper(
            points=(spat, spec),
            values=self.f_obs.Y - np.tile(self.f_sky_obs_1d.Y, reps=(len(spat), 1)),
            values_err=np.sqrt(self.f_obs.Yerr**2 + np.tile(self.f_sky_obs_1d.Yerr, reps=(len(spat), 1)) ** 2),
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
        self.f_host_1d = SpecWrapper(
            points=spec,
            values=np.nanmean(self.f_host.Y * self.f_host.Yerr**-2, axis=0)
            / np.nanmean(self.f_host.Yerr**-2, axis=0)
            * self.host.sum(),
            values_err=np.nanmean(self.f_host.Yerr**-2, axis=0) ** -0.5 * np.sqrt(self.host.sum()),
        )

        # The batched 2D grids for the normalized host galaxy spatial profiles
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

    def build_host_prior(self, imgs: list = [], flts: list = [], **kwargs) -> None:
        """
        Build the prior of the host galaxy using Gaussian Process regression.

        Parameters
        ----------
        imgs : list
            Names of the fits files of host galaxy images.
        flts : list
            Filters of the host galaxy images.
        """
        host_prof = HostProfile(imgs=imgs, flts=flts, spec2d=self)
        self.host_flux_prior = host_prof.model_host_profile_prior(optimization=True, **kwargs)

    def model_host(
        self,
        params_init: dict,
        params_fix: dict = {},
        optimization: bool = False,
        sampling: bool = False,
        optimization_kwargs: dict = {},
        sampling_kwargs: dict = {},
    ) -> None:
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
        num_chains : int, optional (default: 1)
            number of MCMC chains to run.
        num_samples : int, optional (default: 1000)
            number of samples to draw from each chain.
        num_warmup : int, optional (default: 1000)
            number of warmup steps for each chain.
        """

        # Make sure the host flux prior is built
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please build the host flux prior first.")

        if optimization:
            self.gp_params = {
                **self._model_host_optimization(params_init=params_init, params_fix=params_fix, **optimization_kwargs),
                **params_fix,
            }

        if sampling:
            self.inf_data = self._model_host_sampling(params_init=params_init, **sampling_kwargs)
            # TODO: self.gp_params

        if not optimization and not sampling:
            self.gp_params = params_init

        params_1d, params_2d = _split_params(self.gp_params)
        self._gp_1d, self._gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
        # Predict the host galaxy flux outside the mask
        self._f_host_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_host.X)
        # Predict the host galaxy flux on the entire 2D grids
        self._f_pred = self._get_pred(self._gp_1d, self._gp_2d, self.f_obs.X)

    def _model_host_optimization(
        self, params_init: dict, verbose: bool = True, params_fix: dict = {}, **kwargs
    ) -> dict:
        """
        Optimize the Gaussian process model of the host using jaxopt.ScipyMinimize solver.

        Parameters
        ----------
        params_init : dict
            Initial parameters for optimization.
        verbose : bool, optional (default: False)
            Whether to print the optimization status.

        Returns
        -------
        gp_params : dict
            The optimized parameters for the Gaussian Process model.
        """
        solver = jaxopt.ScipyMinimize(fun=self._get_host_neg_log_probability, **kwargs)
        for key in params_fix.keys():
            params_init.pop(key, None)
        soln = solver.run(
            # self,
            params_init,
            f_host=self.f_host,
            f_host_1d=self.f_host_1d,
            f_host_batch_2d=self.f_host_batch_2d,
            host_flux_prior=self.host_flux_prior,
            params_fix=params_fix,
        )
        if soln.state.status != 0:
            warnings.warn(f"Optimization failed with status {soln.state.status}.")
        if verbose:
            print(f"Final parameters: {soln.params}")
        return soln.params

    def _model_host_sampling(self, params_init: dict = None, **kwargs) -> any:
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

        # def numpyro_model():
        #     # Priors
        #     params_1d = dict(
        #         log_jitter=numpyro.sample("log_jitter_1d", dist.Normal(0.0, 1.0)),
        #         log_amp=numpyro.sample("log_amp_1d", dist.Uniform(-10.0, 10.0)),
        #         log_scale=numpyro.sample("log_spec_scale_1d", dist.Normal(0.0, 1.0)),
        #         mean=numpyro.sample("mean_1d", dist.Uniform(0, np.max(self.f_1d))),
        #     )
        #     params_2d = dict(
        #         log_jitter=numpyro.sample("log_jitter_2d", dist.Normal(-2.0, 1.0)),
        #         log_amp=numpyro.sample("log_amp_2d", dist.Normal(-3.0, 1.0)),
        #         log_scale=jnp.asarray(
        #             [
        #                 numpyro.sample("log_spat_scale_2d", dist.Normal(0.0, 1.0)),
        #                 numpyro.sample("log_spec_scale_2d", dist.Normal(3.0, 1.0)),
        #             ]
        #         ),
        #         mean=numpyro.sample("mean_2d", dist.Uniform(-1.0 / self.shape[0], 1.0 / self.shape[0])),
        #     )
        #     gp_1d, gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
        #     numpyro.sample("y_1d", gp_1d.numpyro_dist(), obs=self.f_1d)
        #     numpyro.sample("y_2d", gp_2d.numpyro_dist(), obs=self.f_batch_2d.ravel())

        #     # Likelihood
        #     y_host = numpyro.deterministic("y_host", self._get_pred(gp_1d, gp_2d, self.X_host))
        #     numpyro.sample("y_host_obs", dist.Normal(y_host, noise), obs=self.f_host.ravel())

        # init_strategy = None if params_init == {} else numpyro.infer.init_to_value(values=params_init)
        # nuts_kernel = NUTS(numpyro_model, init_strategy=init_strategy)
        # mcmc = MCMC(nuts_kernel, **kwargs)
        # mcmc.run(jax.random.PRNGKey(0))
        # results = mcmc.get_samples()

        return results

    def _build_host_gp(self, params_1d: dict, params_2d: dict) -> tuple[GaussianProcess, GaussianProcess]:
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
        gp_1d = _gp(X=self.f_host_1d.X, y=self.f_host_1d.y, params=params_1d).gp
        gp_2d = _gp(X=self.f_host_batch_2d.X, y=self.f_host_batch_2d.y, params=params_2d).gp

        return gp_1d, gp_2d

    # def _get_host_neg_log_probability(self, params: dict, **kwargs) -> float:
    #     """
    #     Calculate the negative log probability of the host flux given the parameters.
    #     Not JIT-compiled.

    #     Parameters
    #     ----------
    #     params : dict
    #         A dictionary of parameters.

    #     Returns
    #     -------
    #     float
    #         The negative log probability of the host flux.
    #     """
    #     params_fix = kwargs.get("params_fix", {})
    #     return _get_host_neg_log_probability(
    #         params,
    #         X_1d=self.X_1d,
    #         X_2d=self.X_batch_2d,
    #         X_obs=self.X_host,
    #         y_1d=self.f_1d,
    #         y_2d=self.f_batch_2d.ravel(),
    #         y_obs=self.f_host.ravel(),
    #         y_2d_mean=self.host_flux_prior(self.X_batch_2d),
    #         y_obs_mean=self.host_flux_prior(self.X_host),
    #         params_fix=params_fix,
    #     )
    @partial(jax.jit, static_argnums=(0,))
    def _get_host_neg_log_probability(
        self,
        params: dict,
        *,
        f_host: SpecWrapper,
        f_host_1d: SpecWrapper,
        f_host_batch_2d: SpecWrapper,
        host_flux_prior: Callable,
        params_fix: dict = {},
    ) -> float:
        """
        Compute the negative log probability of the host galaxy model
        """
        params = {**params, **params_fix}
        params_1d, params_2d = _split_params(params)
        gp_1d = _gp(X=f_host_1d.X, yerr=f_host_1d.yerr, params=params_1d).gp
        gp_2d = _gp(X=f_host_batch_2d.X, yerr=f_host_batch_2d.yerr, params=params_2d).gp
        log_prob_1d = gp_1d.log_probability(f_host_1d.y)
        log_prob_2d = gp_2d.log_probability(f_host_batch_2d.y)

        y_2d_mean = host_flux_prior(f_host_batch_2d.X)
        y_obs_mean = host_flux_prior(f_host_1d.X)

        y_host_1d = gp_1d.predict(y=f_host_1d.y, X_test=f_host_1d.X)
        y_host_2d = gp_2d.predict(y=f_host_batch_2d.y - y_2d_mean, X_test=f_host.X) + y_obs_mean
        y_host = y_host_1d * y_host_2d
        log_prob_obs = jnp.nansum(dist.Normal(y_host, f_host.Yerr).log_prob(f_host.Y))

        return -(log_prob_1d + log_prob_2d + log_prob_obs)

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
        assert hasattr(self, "gp_params"), "Please model the host galaxy first."
        print("Gaussian Process parameters:")
        print("1D:")
        print("Amp:", 10 ** self.gp_params.get("log_amp_1d"))
        print("Scale:", 10 ** self.gp_params.get("log_spec_scale_1d"))
        print("Jitter:", 10 ** self.gp_params.get("log_jitter_1d"))
        print("Mean:", self.gp_params.get("mean_1d"))
        print("2D:")
        print("Amp:", 10 ** self.gp_params.get("log_amp_2d"))
        print("Spat Scale:", 10 ** self.gp_params.get("log_spat_scale_2d"))
        print("Spec Scale:", 10 ** self.gp_params.get("log_spec_scale_2d"))
        print("Jitter:", 10 ** self.gp_params.get("log_jitter_2d"))
        print("Mean:", self.gp_params.get("mean_2d"))
        return self.gp_params

    ############################ QA Plotting ############################
    def _plot_raw(self) -> None:
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
            ax_.axhline(self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(-self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.set_aspect("auto")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax_.set_xlim(self.spec.min(), self.spec.max())
            ax_.set_ylim(self.spat.min(), self.spat.max())
        ax[-1].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_batch_pred(self) -> None:
        if not (hasattr(self, "_gp_1d") and hasattr(self, "_gp_2d")):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(5, 1, figsize=(20, 12.5), constrained_layout=True, sharex=True)
        # Plot the 2D batched spectrum
        batch_size = (
            (self.spat[1, -1] - self.spat[0, -1]) * self.batch_2d[0],
            (self.spec[-1, 1] - self.spec[-1, 0]) * self.batch_2d[1],
        )
        norm = plt.Normalize(self.f_host_batch_2d.y.min(), self.f_host_batch_2d.y.max())
        norm_residual = plt.Normalize(-1e-2, 1e-2)
        cmap = plt.cm.get_cmap("gray")
        cmap_residual = plt.cm.get_cmap("RdBu_r")
        pred_1d = self._gp_1d.predict(y=self.f_1d, X_test=self._gp_1d.X)
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
        ax[3].plot(self.f_host_1d.spec, pred_1d, "--k", lw=2)
        ax[4].plot(self.f_host_1d.spec, self.f_host_1d.y - pred_1d)

        # Titles
        ax[0].set_title(r"$\mathrm{2D\ Spectrum}$")
        ax[1].set_title(r"$\mathrm{Model}$")
        ax[2].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[3].set_title(r"$\mathrm{1D\ Spectrum}$")
        ax[4].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")

        # Labels
        ax[4].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")
        for ax_ in ax[:3]:
            ax_.axhline(self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(-self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.set_aspect("auto")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax_.set_xlim(self.spec.min(), self.spec.max())
            ax_.set_ylim(self.spat.min(), self.spat.max())
        ax[3].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_pred(self) -> None:
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
        residual_params = dict(
            origin="lower",
            cmap="RdBu_r",
            aspect="auto",
            vmin=-3,  # TODO: change the hard-coded value
            vmax=3,
            extent=[self.spec.min(), self.spec.max(), self.spat.min(), self.spat.max()],
        )

        _, ax = plt.subplots(3, 1, figsize=(20, 7.5), sharex=True, sharey=True, constrained_layout=True)
        ax[0].imshow(self.f_sky_sub.Y, **source_params)
        ax[1].imshow(self._f_pred.reshape(-1, self.shape[1]), **source_params)
        ax[2].imshow(self.f_sky_sub.Y - self._f_pred.reshape(-1, self.shape[1]), **residual_params)
        for ax_ in ax:
            ax_.axhline(-self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(self.mask_wid * self.spat_resln, color="tab:red", linestyle="--")
            ax_.axhline(-self.sky_wid[0] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.axhline(self.sky_wid[1] * self.spat_resln, color="tab:blue", linestyle="-.")
            ax_.set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[0].set_title(r"$\mathrm{Source}$")
        ax[1].set_title(r"$\mathrm{Model}$")
        ax[2].set_title(r"$\mathrm{Residual} = \mathrm{Source} - \mathrm{Model}$")
        ax[2].set_xlabel(r"$\mathrm{Spec\ [\AA]}$")

        plt.show()


class Spec2D(Spec2DBase):
    def __init__(
        self,
        dat: ArrayLike,  # 2D spectrum (spatial x spectral)
        dat_err: ArrayLike = None,  # 2D error spectrum
        *,
        spat_img: ArrayLike = None,  # spatial grids
        spec_img: ArrayLike = None,  # spectral grids
        pixel_scale: float = None,  # arcsec/pixel
        center_ra: float = None,  # RA of the center
        center_dec: float = None,  # DEC of the center
        slit_wid: float = 1.0,  # arcsec
        slit_len: float = 10.0,  # arcsec
        position_angle: float = None,  # degree
        spat_resln: float = 1.0,  # arcsec, FWHM/seeing
        spec_resln: float = 7.5,  # LRIS, 1'' slit
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        sky_wid: tuple = (5.0, 5.0),  # sky region
        batch_2d: tuple = (2, 64),  # batch size for modeling slowing varying host profiles
        show: bool = False,
    ):

        # The 2D grids for the raw data
        self.f_obs = dat
        self.f_obs_err = dat_err if dat_err is not None else np.ones_like(dat)
        if spat_img.ndim == 1:
            assert dat.shape == (spat_img.size, spec_img.size), "spec2d shape mismatch"
            super().__init__(
                dat,
                dat_err,
                spat=spat_img,
                spec=spec_img,
                pixel_scale=pixel_scale,
                center_ra=center_ra,
                center_dec=center_dec,
                slit_wid=slit_wid,
                slit_len=slit_len,
                position_angle=position_angle,
                spat_resln=spat_resln,
                spec_resln=spec_resln,
                mask_wid=mask_wid,
                sky_wid=sky_wid,
                batch_2d=batch_2d,
                show=show,
            )
        elif spat_img.ndim == 2:
            # needs rectification
            assert dat.shape == spat_img.shape == spec_img.shape, "spec2d shape mismatch"
            self.spat_img = spat_img
            self.spec_img = spec_img

            spat_img_pseudo = np.linspace(spat_img.min(), spat_img.max(), spat_img.shape[0])
            spec_img_pseudo = np.linspace(spec_img.min(), spec_img.max(), spec_img.shape[1])

            dat_pseudo, dat_err_pseudo = rectification(
                (dat, dat_err), spat_img, spec_img, spat_img_pseudo, spec_img_pseudo
            )

            super().__init__(
                dat_pseudo,
                dat_err_pseudo,
                spat_img=spat_img_pseudo,
                spec_img=spec_img_pseudo,
                pixel_scale=pixel_scale,
                center_ra=center_ra,
                center_dec=center_dec,
                slit_wid=slit_wid,
                slit_len=slit_len,
                position_angle=position_angle,
                spat_resln=spat_resln,
                spec_resln=spec_resln,
                mask_wid=mask_wid,
                sky_wid=sky_wid,
                batch_2d=batch_2d,
                show=show,
            )

            self.f_obs_raw = dat
            self.f_obs_raw_err = dat_err if dat_err is not None else np.ones_like(dat)


@jax.jit
def rectification(
    f_values: tuple[Array, Array], *, spat_img: Array, spec_img: Array, spat: Array, spec: Array
) -> tuple[Array, Array]:
    """
    Rectify the 2D spectrum onto a grid.
    """
    raise NotImplementedError


@jax.jit
def _get_host_neg_log_probability(
    params: dict, *, X_1d, X_2d, X_obs, y_1d, y_2d, y_obs, y_2d_mean, y_obs_mean, params_fix: dict = {}
) -> float:
    """
    Compute the negative log probability of the host galaxy model
    """
    params = {**params, **params_fix}
    params_1d, params_2d = _split_params(params)
    gp_1d = _gp(X=X_1d, params=params_1d).gp
    gp_2d = _gp(X=X_2d, params=params_2d).gp
    log_prob_1d = gp_1d.log_probability(y_1d)
    log_prob_2d = gp_2d.log_probability(y_2d)

    y_host_1d = gp_1d.predict(y=y_1d, X_test=X_obs[:, 1][:, None])
    y_host_2d = gp_2d.predict(y=y_2d - y_2d_mean, X_test=X_obs) + y_obs_mean
    y_host = y_host_1d * y_host_2d
    log_prob_obs = jnp.nansum(dist.Normal(y_host, noise).log_prob(y_obs))

    return -(log_prob_1d + log_prob_2d + log_prob_obs)


def _split_params(params: dict) -> tuple[dict, dict]:
    """
    Split the parameters into 1D and 2D.
    """
    params_1d = {
        "log_amp": params.get("log_amp_1d", jnp.float64(3.0)),
        "log_scale": params.get("log_spec_scale_1d", jnp.float64(0.0)),
        "log_jitter": params.get("log_jitter_1d", jnp.float64(1e-6)),
        "mean": params.get("mean_1d", jnp.float64(0)),
    }
    params_2d = {
        "log_amp": params.get("log_amp_2d", jnp.float64(-3.0)),
        "log_scale": jnp.asarray(
            [params.get("log_spat_scale_2d", jnp.float64(0.0)), params.get("log_spec_scale_2d", jnp.float64(3.0))]
        ),
        "log_jitter": params.get("log_jitter_2d", jnp.float64(1e-6)),
        "mean": params.get("mean_2d", jnp.float64(0)),
    }

    return params_1d, params_2d
