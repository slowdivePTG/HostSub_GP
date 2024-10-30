# hostsub_gp/spec.py

__all__ = ["Spec2D"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

from tinygp import GaussianProcess

from ._plt_config import plt
from .gp import _gp
from .host_prof import HostProfile

from typing import Callable, Tuple
import warnings


class Spec2D:
    def __init__(
        self,
        dat: np.ndarray,  # 2D spectrum (spatial x spectral)
        dat_err: np.ndarray = None,  # 2D error spectrum
        *,
        coord_spat: np.ndarray = None,  # spatial grids
        coord_spec: np.ndarray = None,  # spectral grids
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
        noise: float = None,  # noise level
        batch_2d: tuple = (2, 50),  # batch size for modeling slowing varying host profiles
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

        # The 2D grids for the raw data
        self.f_obs = dat
        self.f_obs_err = dat_err if dat_err is not None else np.ones_like(dat)
        print(f"Loading the 2D spectrum with the shape: {self.f_obs.shape}")
        if coord_spat.ndim == 1:
            assert dat.shape == (coord_spat.size, coord_spec.size), "spec2d shape mismatch"
            self.coord_spec, self.coord_spat = np.meshgrid(coord_spec, coord_spat)
        elif coord_spat.ndim == 2:
            assert dat.shape == coord_spat.shape == coord_spec.shape, "spec2d shape mismatch"
            self.coord_spat = coord_spat
            self.coord_spec = coord_spec
        self.X = np.stack([self.coord_spat.ravel(), self.coord_spec.ravel()], axis=-1)
        self.shape = self.f_obs.shape

        # The global sky region (spat < -sky_wid[0] * seeing) or (spat > sky_wid[1] * seeing)
        sky_left = np.min(self.coord_spat, axis=1) < -self.spat_resln * self.sky_wid[0]
        sky_right = np.max(self.coord_spat, axis=1) > self.spat_resln * self.sky_wid[1]
        self.sky = sky_left | sky_right
        if self.sky.sum() / self.sky.ravel().size < 0.1:
            warnings.warn(r"Sky region is < 10% of the overall pixels.")
        # Estimate the noise background
        self.noise = noise if noise is not None else np.std(self.f_obs[self.sky])

        # Estimate the global sky background (sky + host): mean of the sky region along the spectral direction
        self.f_bkg = np.mean(self.f_obs[self.sky, :], axis=0)
        self.f_bkg_err = np.sqrt(np.mean(self.f_obs_err[self.sky, :]**2, axis=0))
        self.f_sky_sub = self.f_obs - np.tile(self.f_bkg, (self.shape[0], 1))
        self.f_sky_sub_err = np.sqrt(self.f_obs_err ** 2 + np.tile(self.f_bkg_err ** 2, (self.shape[0], 1)))

        # Mask the trace from the source (|spat| < seeing * mask_wid)
        assert min(sky_wid) > mask_wid, "sky_wid should be larger than mask_wid"
        host_left = np.min(self.coord_spat, axis=1) < -self.mask_wid * self.spat_resln
        host_right = np.max(self.coord_spat, axis=1) > self.mask_wid * self.spat_resln
        self.host = host_left | host_right
        self.f_host = self.f_sky_sub[self.host, :]
        self.f_host_err = self.f_sky_sub_err[self.host, :]
        self.X_host = np.stack([self.coord_spat[self.host, :].ravel(), self.coord_spec[self.host, :].ravel()], axis=-1)

        # The 1D grids for the sky-subtracted host galaxy spectra: sum along the spatial direction outside the mask
        print(f"Obtaining the sky-subtracted 1D galaxy spectrum (outside the mask)")
        self.f_1d = np.sum(self.f_sky_sub[self.host, :], axis=0)
        self.f_1d_err = np.sqrt(np.sum(self.f_sky_sub_err[self.host, :] ** 2, axis=0))
        # Central wavelength in each row: mean of the row
        self.X_1d = np.mean(self.coord_spec[self.host, :], axis=0)[:, None]

        # The batched 2D grids for the normalized host galaxy spatial profiles
        self.batch_2d = batch_2d
        print(f"Batching the 2D galaxy spectrum (outside the mask) with the size: {batch_2d}")
        spat_batch_2d_idx = np.array_split(
            np.arange(self.shape[0])[host_left], host_left.sum() // batch_2d[0]  # Left side
        ) + np.array_split(
            np.arange(self.shape[0])[host_right], host_right.sum() // batch_2d[0]  # Right side
        )
        spec_batch_2d_idx = np.array_split(np.arange(self.shape[1]), self.shape[1] // batch_2d[1])
        self.shape_batch_2d = (len(spat_batch_2d_idx), len(spec_batch_2d_idx))
        self.coord_spat_batch_2d, self.coord_spec_batch_2d = np.empty(self.shape_batch_2d), np.empty(
            self.shape_batch_2d
        )
        self.f_batch_2d = np.empty(self.shape_batch_2d)
        self.f_batch_2d_err = np.empty(self.shape_batch_2d)
        for x in range(self.shape_batch_2d[0]):
            for y in range(self.shape_batch_2d[1]):
                # New coordinates: mean of the batch
                self.coord_spat_batch_2d[x, y] = self.coord_spat[spat_batch_2d_idx[x], :][
                    :, spec_batch_2d_idx[y]
                ].mean()
                self.coord_spec_batch_2d[x, y] = self.coord_spec[spat_batch_2d_idx[x], :][
                    :, spec_batch_2d_idx[y]
                ].mean()
                # New values: mean of the batch
                self.f_batch_2d[x, y] = (self.f_sky_sub / self.f_1d)[spat_batch_2d_idx[x], :][
                    :, spec_batch_2d_idx[y]
                ].mean()
                self.f_batch_2d_err[x, y] = np.sqrt(
                    (self.f_sky_sub_err / self.f_1d)[spat_batch_2d_idx[x], :][
                        :, spec_batch_2d_idx[y]
                    ] ** 2
                ).mean()
        self.X_batch_2d = np.stack([self.coord_spat_batch_2d.ravel(), self.coord_spec_batch_2d.ravel()], axis=-1)
        print("Batched 2D galaxy spectrum:", self.f_batch_2d.shape)

        if show:
            self._plot_raw()

        assert (self.X_1d.shape[1] == 1) & (self.X_1d.shape[0] == self.f_1d.size), "1D spectrum error"
        assert (self.X_batch_2d.shape[1] == 2) & (
            self.X_batch_2d.shape[0] == self.f_batch_2d.ravel().size
        ), "2D batch error"

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
        self._f_host_pred = self._get_pred(self._gp_1d, self._gp_2d, self.X_host)
        # Predict the host galaxy flux on the entire 2D grids
        self._f_pred = self._get_pred(self._gp_1d, self._gp_2d, self.X)

    def _get_host_neg_log_probability(self, params: dict, **kwargs) -> float:
        """
        Calculate the negative log probability of the host flux given the parameters.
        Not JIT-compiled.

        Parameters
        ----------
        params : dict
            A dictionary of parameters.

        Returns
        -------
        float
            The negative log probability of the host flux.
        """
        params_fix = kwargs.get("params_fix", {})
        return _get_host_neg_log_probability(
            params,
            X_1d=self.X_1d,
            X_2d=self.X_batch_2d,
            X_obs=self.X_host,
            y_1d=self.f_1d,
            y_2d=self.f_batch_2d.ravel(),
            y_obs=self.f_host.ravel(),
            y_2d_mean=self.host_flux_prior(self.X_batch_2d),
            y_obs_mean=self.host_flux_prior(self.X_host),
            noise=self.noise,
            params_fix=params_fix,
        )

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
        solver = jaxopt.ScipyMinimize(fun=_get_host_neg_log_probability, **kwargs)
        for key in params_fix.keys():
            params_init.pop(key, None)
        soln = solver.run(
            params_init,
            X_1d=self.X_1d,
            X_2d=self.X_batch_2d,
            X_obs=self.X_host,
            y_1d=self.f_1d,
            y_2d=self.f_batch_2d.ravel(),
            y_obs=self.f_host.ravel(),
            y_2d_mean=self.host_flux_prior(self.X_batch_2d),
            y_obs_mean=self.host_flux_prior(self.X_host),
            noise=self.noise,
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

        def numpyro_model():
            # Priors
            params_1d = dict(
                log_jitter=numpyro.sample("log_jitter_1d", dist.Normal(0.0, 1.0)),
                log_amp=numpyro.sample("log_amp_1d", dist.Uniform(-10.0, 10.0)),
                log_scale=numpyro.sample("log_spec_scale_1d", dist.Normal(0.0, 1.0)),
                mean=numpyro.sample("mean_1d", dist.Uniform(0, np.max(self.f_1d))),
            )
            params_2d = dict(
                log_jitter=numpyro.sample("log_jitter_2d", dist.Normal(-2.0, 1.0)),
                log_amp=numpyro.sample("log_amp_2d", dist.Normal(-3.0, 1.0)),
                log_scale=jnp.asarray(
                    [
                        numpyro.sample("log_spat_scale_2d", dist.Normal(0.0, 1.0)),
                        numpyro.sample("log_spec_scale_2d", dist.Normal(3.0, 1.0)),
                    ]
                ),
                mean=numpyro.sample("mean_2d", dist.Uniform(-1.0 / self.shape[0], 1.0 / self.shape[0])),
            )
            gp_1d, gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
            numpyro.sample("y_1d", gp_1d.numpyro_dist(), obs=self.f_1d)
            numpyro.sample("y_2d", gp_2d.numpyro_dist(), obs=self.f_batch_2d.ravel())

            # Likelihood
            y_host = numpyro.deterministic("y_host", self._get_pred(gp_1d, gp_2d, self.X_host))
            noise = kwargs.get("noise", 1)
            numpyro.sample("y_host_obs", dist.Normal(y_host, noise), obs=self.f_host.ravel())

        init_strategy = None if params_init == {} else numpyro.infer.init_to_value(values=params_init)
        nuts_kernel = NUTS(numpyro_model, init_strategy=init_strategy)
        mcmc = MCMC(nuts_kernel, **kwargs)
        mcmc.run(jax.random.PRNGKey(0))
        results = mcmc.get_samples()

        return results

    def _build_host_gp(self, params_1d: dict, params_2d: dict) -> Tuple[GaussianProcess, GaussianProcess]:
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
        Tuple[GaussianProcess, GaussianProcess]
            tinygp.GaussianProcess objects for the 1D and 2D host galaxy.
        """
        gp_1d = _gp(X=self.X_1d, y=self.f_1d, params=params_1d).gp
        gp_2d = _gp(X=self.X_batch_2d, y=self.f_batch_2d.ravel(), params=params_2d).gp

        return gp_1d, gp_2d

    def _get_pred(self, gp_1d: GaussianProcess, gp_2d: GaussianProcess, X: jax.Array) -> jax.Array:
        """
        Get the predicted host galaxy flux on the given grids.

        Parameters
        ----------
        gp_1d : GaussianProcess
            The 1D Gaussian Process - the 1D spectrum of the host.
        gp_2d : GaussianProcess
            The 2D Gaussian Process - the spatial profile of the host.
        X : jax.Array
            The 2D grids to make the prediction.

        Returns
        -------
        jax.Array
            The predicted host galaxy flux.
        """
        y_host_1d = gp_1d.predict(y=self.f_1d, X_test=X[:, 1][:, None])
        y_host_2d = gp_2d.predict(
            y=self.f_batch_2d.ravel() - self.host_flux_prior(gp_2d.X), X_test=X
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
            self.f_obs,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_obs, 5),
            vmax=np.nanpercentile(self.f_obs, 99),
            extent=[self.coord_spec.min(), self.coord_spec.max(), self.coord_spat.min(), self.coord_spat.max()],
        )
        ax[1].imshow(
            self.f_sky_sub,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(self.f_sky_sub, 5),
            vmax=np.nanpercentile(self.f_sky_sub, 99),
            extent=[self.coord_spec.min(), self.coord_spec.max(), self.coord_spat.min(), self.coord_spat.max()],
        )
        # Plot the 2D batched spectrum
        batch_size = (
            (self.coord_spat[1, 0] - self.coord_spat[0, 0]) * self.batch_2d[0],
            (self.coord_spec[0, 1] - self.coord_spec[0, 0]) * self.batch_2d[1],
        )
        norm = plt.Normalize(self.f_batch_2d.min(), self.f_batch_2d.max())
        cmap = plt.cm.get_cmap("gray")
        for k in range(len(self.X_batch_2d[:, 0])):
            c_raw = cmap(norm(self.f_batch_2d.ravel()[k]))
            ax[2].add_patch(
                plt.Rectangle(
                    (self.X_batch_2d[k, 1] - batch_size[1] / 2, self.X_batch_2d[k, 0] - batch_size[0] / 2),
                    batch_size[1],
                    batch_size[0],
                    color=c_raw,
                )
            )

        # Plot the 1D batched spectrum
        ax[-1].plot(self.X_1d, self.f_1d)

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
            ax_.set_xlim(self.coord_spec.min(), self.coord_spec.max())
            ax_.set_ylim(self.coord_spat.min(), self.coord_spat.max())
        ax[-1].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_batch_pred(self) -> None:
        if not (hasattr(self, "_gp_1d") and hasattr(self, "_gp_2d")):
            raise ValueError("Please model the host galaxy first.")
        _, ax = plt.subplots(5, 1, figsize=(20, 12.5), constrained_layout=True, sharex=True)
        # Plot the 2D batched spectrum
        batch_size = (
            (self.coord_spat[1, 0] - self.coord_spat[0, 0]) * self.batch_2d[0],
            (self.coord_spec[0, 1] - self.coord_spec[0, 0]) * self.batch_2d[1],
        )
        norm = plt.Normalize(self.f_batch_2d.min(), self.f_batch_2d.max())
        norm_residual = plt.Normalize(-1e-2, 1e-2)
        cmap = plt.cm.get_cmap("gray")
        cmap_residual = plt.cm.get_cmap("RdBu_r")
        pred_1d = self._gp_1d.predict(y=self.f_1d, X_test=self._gp_1d.X)
        pred_2d = self._gp_2d.predict(
            y=self.f_batch_2d.ravel() - self.host_flux_prior(self._gp_2d.X), X_test=self._gp_2d.X
        ) + self.host_flux_prior(self._gp_2d.X)
        for k in range(len(self.X_batch_2d[:, 0])):
            c_raw = cmap(norm(self.f_batch_2d.ravel()[k]))
            c_model = cmap(norm(pred_2d[k]))
            c_residual = cmap_residual(norm_residual(self.f_batch_2d.ravel()[k] - pred_2d[k]))
            ax[0].add_patch(
                plt.Rectangle(
                    (self.X_batch_2d[k, 1] - batch_size[1] / 2, self.X_batch_2d[k, 0] - batch_size[0] / 2),
                    batch_size[1],
                    batch_size[0],
                    color=c_raw,
                )
            )
            ax[1].add_patch(
                plt.Rectangle(
                    (self.X_batch_2d[k, 1] - batch_size[1] / 2, self.X_batch_2d[k, 0] - batch_size[0] / 2),
                    batch_size[1],
                    batch_size[0],
                    color=c_model,
                )
            )
            ax[2].add_patch(
                plt.Rectangle(
                    (self.X_batch_2d[k, 1] - batch_size[1] / 2, self.X_batch_2d[k, 0] - batch_size[0] / 2),
                    batch_size[1],
                    batch_size[0],
                    color=c_residual,
                )
            )

        # Plot the 1D batched spectrum
        ax[3].plot(self.X_1d, self.f_1d)
        ax[3].plot(self.X_1d, pred_1d, "--k", lw=2)
        ax[4].plot(self.X_1d, self.f_1d - pred_1d)

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
            ax_.set_xlim(self.coord_spec.min(), self.coord_spec.max())
            ax_.set_ylim(self.coord_spat.min(), self.coord_spat.max())
        ax[3].set_ylabel(r"$\mathrm{Counts}$")

        plt.show()

    def _plot_host_pred(self) -> None:
        if not (hasattr(self, "_f_host_pred") and hasattr(self, "_f_pred")):
            raise ValueError("Please model the host galaxy first.")

        source_params = dict(
            origin="lower",
            cmap="gray",
            aspect="auto",
            vmin=np.nanpercentile(self.f_sky_sub, 5),
            vmax=np.nanpercentile(self.f_sky_sub, 99),
            extent=[self.coord_spec.min(), self.coord_spec.max(), self.coord_spat.min(), self.coord_spat.max()],
        )
        residual_params = dict(
            origin="lower",
            cmap="RdBu_r",
            aspect="auto",
            vmin=-3 * self.noise,
            vmax=3 * self.noise,
            extent=[self.coord_spec.min(), self.coord_spec.max(), self.coord_spat.min(), self.coord_spat.max()],
        )

        _, ax = plt.subplots(3, 1, figsize=(20, 7.5), sharex=True, sharey=True, constrained_layout=True)
        ax[0].imshow(self.f_sky_sub, **source_params)
        ax[1].imshow(self._f_pred.reshape(-1, self.shape[1]), **source_params)
        ax[2].imshow(self.f_sky_sub - self._f_pred.reshape(-1, self.shape[1]), **residual_params)
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


@jax.jit
def _get_host_neg_log_probability(
    params: dict, *, X_1d, X_2d, X_obs, y_1d, y_2d, y_obs, y_2d_mean, y_obs_mean, noise, params_fix: dict = {}
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
    log_prob_obs = dist.Normal(y_host, noise).log_prob(y_obs).sum()

    return -(log_prob_1d + log_prob_2d + log_prob_obs)


def _split_params(params: dict) -> Tuple[dict, dict]:
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
