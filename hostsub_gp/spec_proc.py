# hostsub_gp/spec_proc.py

__all__ = ["Spec2D"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxopt

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from tinygp import GaussianProcess

from ._plt_config import plt
from .gp import _gp
from .host_prof import HostProfile

from typing import Callable, Tuple

import numpyro
import arviz as az


class Spec2D:
    def __init__(
        self,
        spec2d: np.ndarray,  # 2D spectrum (spatial x spectral)
        spat: np.ndarray,  # spatial grids
        spec: np.ndarray,  # spectral grids
        center_ra: float = None,  # RA of the center
        center_dec: float = None,  # DEC of the center
        slit_wid: float = 1.0,  # arcsec
        slit_len: float = 10.0,  # arcsec
        position_angle: float = None,  # degree
        spat_resln: float = 1.0,  # arcsec, FWHM/seeing
        spec_resln: float = 7.5,  # LRIS, 1'' slit
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        noise: float = 1.0,  # noise level
        batch_1d: int = 1,  # batch size for modeling 1D host spectra
        batch_2d: tuple = (2, 50),  # batch size for modeling slowing varying host profiles
        show: bool = False,
    ):
        assert spec2d.shape == (spat.size, spec.size), "spec2d shape mismatch"
        self.spec2d = spec2d
        self.spat = spat
        self.spec = spec
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.slit_wid = slit_wid
        self.slit_len = slit_len
        self.position_angle = position_angle
        self.spat_resln = spat_resln
        self.spec_resln = spec_resln
        self.mask_wid = mask_wid

        self.noise = noise

        self.batch_1d = batch_1d
        self.batch_2d = batch_2d

        # The 2D grids for the raw data
        print(f"Loading the 2D spectrum with the shape: {self.spec2d.shape}")
        spec_grid2d, spat_grid2d = np.meshgrid(self.spec, self.spat)
        self.X = np.stack([spat_grid2d.ravel(), spec_grid2d.ravel()], axis=-1)

        # Mask the trace from the source (|spat| < seeing * mask_wid)
        self.mask = np.abs(self.spat) < self.spat_resln * self.mask_wid
        mask_2d = np.tile(self.mask, (self.spec.size, 1)).T
        self.spec2d_host = self.spec2d[~mask_2d]
        self.X_host = self.X[~mask_2d.ravel()]

        # The batched 1D grids for the host galaxy spectra (i.e., outside the mask)
        self.spec1d_host = self.spec2d[~self.mask].sum(axis=0)
        print(f"Batching the 1D galaxy spectrum (outside the mask) with the size: {batch_1d}")
        spec_batch_1d_idx = np.array_split(np.arange(self.spec.size), self.spec.size // batch_1d)
        spec_batch_1d = [self.spec[spec_batch] for spec_batch in spec_batch_1d_idx]
        # New central wavelength in each batch: mean of the batch
        self.X_batch_1d = np.array([spec_batch.mean() for spec_batch in spec_batch_1d])
        # New values: mean of the batch
        self.spec1d_batch_1d = np.array([self.spec1d_host[spec_batch].mean() for spec_batch in spec_batch_1d_idx])
        print("Batched 1D galaxy spectrum:", self.spec1d_batch_1d.shape)

        # The batched 2D grids for the normalized host galaxy spatial profiles
        print(f"Batching the 2D galaxy spectrum (outside the mask) with the size: {batch_2d}")
        host_left = ~self.mask & (self.spat < 0)
        host_right = ~self.mask & (self.spat > 0)
        spat_batch_2d_idx = np.array_split(
            np.arange(self.spat.size)[host_left], host_left.sum() // batch_2d[0]  # Left side
        ) + np.array_split(
            np.arange(self.spat.size)[host_right], host_right.sum() // batch_2d[0]  # Right side
        )
        spec_batch_2d_idx = np.array_split(np.arange(self.spec.size), self.spec.size // batch_2d[1])
        spat_batch_2d = [self.spat[spat_batch] for spat_batch in spat_batch_2d_idx]
        spec_batch_2d = [self.spec[spec_batch] for spec_batch in spec_batch_2d_idx]
        # New coordinates: mean of the batch
        self.spat_batch_2d = np.array([spat_batch.mean() for spat_batch in spat_batch_2d])
        self.spec_batch_2d = np.array([spec_batch.mean() for spec_batch in spec_batch_2d])
        spec_batch_2d_grid2d, spat_batch_2d_grid2d = np.meshgrid(self.spec_batch_2d, self.spat_batch_2d)
        self.X_batch_2d = np.stack([spat_batch_2d_grid2d.ravel(), spec_batch_2d_grid2d.ravel()], axis=-1)
        # New values: mean of the batch
        spec2d_batch_2d = np.array(
            [
                [self.spec2d[spat_batch, :][:, spec_batch].mean() for spec_batch in spec_batch_2d_idx]
                for spat_batch in spat_batch_2d_idx
            ]
        )
        self.spec2d_batch_2d = (
            spec2d_batch_2d / jnp.sum(spec2d_batch_2d, axis=0)[None:,] * (len(spat_batch_2d_idx) / len(self.spat))
        )
        print("Batched 2D galaxy spectrum:", self.spec2d_batch_2d.shape)

        if show:
            _, ax = plt.subplots(3, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
            # Plot the original 2D spectrum
            ax[0].imshow(
                self.spec2d,
                origin="lower",
                cmap="gray",
                vmin=np.nanpercentile(self.spec2d, 5),
                vmax=np.nanpercentile(self.spec2d, 99),
                extent=[self.spec[0], self.spec[-1], self.spat[0], self.spat[-1]],
            )
            ax[0].axhline(-self.mask_wid * self.spat_resln, color="red", linestyle="--")
            ax[0].axhline(self.mask_wid * self.spat_resln, color="red", linestyle="--")
            # Plot the 2D batched spectrum
            ax[1].imshow(
                self.spec2d_batch_2d,
                origin="lower",
                cmap="gray",
                vmin=np.nanpercentile(self.spec2d_batch_2d, 5),
                vmax=np.nanpercentile(self.spec2d_batch_2d, 99),
                extent=[self.spec_batch_2d[0], self.spec_batch_2d[-1], self.spat_batch_2d[0], self.spat_batch_2d[-1]],
            )
            ax[1].axhline(0, color="red", linestyle="--")
            # Plot the 1D batched spectrum
            ax[2].plot(self.X_batch_1d, self.spec1d_batch_1d)
            ax[0].set_xlabel("Spec (Angstrom)")
            for ax_ in ax[:2]:
                ax_.set_aspect("auto")
                ax_.set_ylabel("Spat (arcsec)")
            ax[2].set_ylabel("Counts")
            plt.show()

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
        host_prof = HostProfile(imgs=imgs, flts=flts, spec2d=self, **kwargs)
        self.host_flux_prior = host_prof.model_host_profile_prior(optimization=True)

    def model_host(
        self,
        params_init: dict,
        optimization: bool = True,
        sampling: bool = False,
        num_chains: int = 1,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        **kwargs,
    ) -> None:
        """
        Model the host galaxy using Gaussian Process regression.

        Parameters
        ----------
        params_init : dict
            Initial parameters for optimization.
        optimization : bool, optional (default: True)
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

        assert optimization or sampling, "Please specify modeling method (optimization/sampling)."

        if optimization:
            self.gp_params = self._model_host_optimization(params_init=params_init, **kwargs)

        if sampling:
            inf_data = self._model_host_sampling(num_chains, num_samples, num_warmup, **kwargs)
            # TODO: self.gp_params

        params_1d, params_2d = _split_params(self.gp_params)
        self._gp_1d, self._gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
        self._spec2d_host_pred = self._get_host_pred(self._gp_1d, self._gp_2d)

    def _get_host_neg_log_probability(self, params: dict) -> float:
        """
        Calculate the negative log probability of the host flux given the parameters.

        Parameters
        ----------
        params : dict
            A dictionary of parameters.

        Returns
        -------
        float
            The negative log probability of the host flux.
        """
        return _get_host_neg_log_probability(
            params,
            X_1d=self.X_batch_1d,
            X_2d=self.X_batch_2d,
            X_obs=self.X_host,
            y_1d=self.spec1d_batch_1d,
            y_2d=self.spec2d_batch_2d.ravel(),
            y_obs=self.spec2d_host.ravel(),
            y_2d_mean=self.host_flux_prior(self.X_batch_2d),
            y_obs_mean=self.host_flux_prior(self.X_host),
            noise=self.noise,
        )

    def _model_host_optimization(self, params_init: dict, verbose: bool) -> dict:
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
        solver = jaxopt.ScipyMinimize(fun=_get_host_neg_log_probability)
        soln = solver.run(
            params_init,
            X_1d=self.X_batch_1d,
            X_2d=self.X_batch_2d,
            X_obs=self.X_host,
            y_1d=self.spec1d_batch_1d,
            y_2d=self.spec2d_batch_2d.ravel(),
            y_obs=self.spec2d_host.ravel(),
            y_2d_mean=self.host_flux_prior(self.X_batch_2d),
            y_obs_mean=self.host_flux_prior(self.X_host),
            noise=self.noise,
        )
        if verbose:
            print(f"Optimization status: {soln.state}")
            print(f"Final parameters: {soln.params}")
            # print(f"Final negative log likelihood: {soln.state.fun_val}")
        return soln.params

    def _model_host_sampling(self, num_chains: int, num_samples: int, num_warmup: int, **kwargs) -> az.InferenceData:
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
                log_jitter=numpyro.sample("log_jitter_1d", dist.HalfNormal(1e-6)),
                log_amp=numpyro.sample("log_amp_1d", dist.Normal(-3.0, 1.0)),
                log_scale=numpyro.sample("log_spec_scale_1d", dist.Normal(0.0, 1.0)),
                mean=numpyro.sample(
                    "mean_1d", dist.Uniform(-np.median(self.spec1d_batch_1d), np.median(self.spec1d_batch_1d))
                ),
            )
            params_2d = dict(
                log_jitter=numpyro.sample("log_jitter_2d", dist.HalfNormal(1e-6)),
                log_amp=numpyro.sample("log_amp_2d", dist.Normal(-3.0, 1.0)),
                log_scale=jnp.asarray(
                    [
                        numpyro.sample("log_spat_scale_2d", dist.Normal(0.0, 1.0)),
                        numpyro.sample("log_spec_scale_2d", dist.Normal(3.0, 1.0)),
                    ]
                ),
                mean=numpyro.sample("mean_2d", dist.Uniform(-1 / len(self.spat), 1 / len(self.spat))),
            )
            gp_1d, gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
            numpyro.sample("y_1d", gp_1d.numpyro_dist(), obs=self.spec1d_batch_1d)
            numpyro.sample("y_2d", gp_2d.numpyro_dist(), obs=self.spec2d_batch_2d.ravel())

            # Likelihood
            y_host = numpyro.deterministic("y_host", self._get_host_pred(gp_1d, gp_2d))
            noise = kwargs.get("noise", 1)
            numpyro.sample("y_host_obs", dist.Normal(y_host, noise), obs=self.spec2d_host.ravel())

        nuts_kernel = NUTS(numpyro_model, target_accept_prob=0.9)
        mcmc = MCMC(
            nuts_kernel,
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
            progress_bar=True,
            **kwargs,
        )
        mcmc.run(jax.random.PRNGKey(0))
        results = az.convert_to_inference_data(mcmc.get_samples())

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
        gp_1d = _gp(X=self.X_batch_1d[:, None], y=self.spec1d_batch_1d, params=params_1d).gp
        gp_2d = _gp(X=self.X_batch_2d, y=self.spec2d_batch_2d.ravel(), params=params_2d).gp

        return gp_1d, gp_2d

    def _get_host_pred(self, gp_1d: GaussianProcess, gp_2d: GaussianProcess) -> jax.Array:
        """
        Get the predicted host galaxy flux.

        Parameters
        ----------
        gp_1d : GaussianProcess
            The 1D Gaussian Process - the 1D spectrum of the host.
        gp_2d : GaussianProcess
            The 2D Gaussian Process - the spatial profile of the host.

        Returns
        -------
        jax.Array
            The predicted host galaxy flux.
        """
        y_host_1d = gp_1d.predict(y=self.spec1d_batch_1d, X_test=self.X_host[:, 1][:, None])
        y_host_2d = gp_2d.predict(
            y=self.spec2d_batch_2d.ravel() - self.host_flux_prior(gp_2d.X), X_test=self.X_host
        ) + self.host_flux_prior(self.X_host)

        return y_host_1d * y_host_2d


@jax.jit
def _get_host_neg_log_probability(params, X_1d, X_2d, X_obs, y_1d, y_2d, y_obs, y_2d_mean, y_obs_mean, noise) -> float:
    """
    Compute the negative log probability of the host galaxy model
    """
    params_1d, params_2d = _split_params(params)
    gp_1d = _gp(X=X_1d[:, None], y=y_1d, params=params_1d).gp
    gp_2d = _gp(X=X_2d, y=y_2d, params=params_2d).gp
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
