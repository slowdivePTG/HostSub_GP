# hostsub_gp/spec_proc.py

__all__ = ["Spec2D"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from tinygp import GaussianProcess, kernels, transforms

from ._plt_config import plt
from .gp import gp
from .host_prof import HostProfile

from typing import Callable, Tuple

import numpyro


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

        self.batch_1d = batch_1d
        self.batch_2d = batch_2d

        # TODO: "Check the spat and spec axes - seems to be inconsistent with the axes in host_prof.py"

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
        """
        host_prof = HostProfile(imgs=imgs, flts=flts, spec2d=self, **kwargs)
        self.host_flux_prior = host_prof.model_host_profile_prior(optimization=True)

    def model_host(self, num_chains: int = 1, num_samples: int = 1000, num_warmup: int = 1000, **kwargs) -> None:
        """
        Model the host galaxy using Gaussian Process regression.
        """

        # Make sure the host flux prior is built
        if not hasattr(self, "host_flux_prior"):
            raise ValueError("Please build the host flux prior first.")

        def numpyro_model():
            # Priors
            params_1d = dict(
                jitter=numpyro.sample("jitter_1d", dist.HalfNormal((1e-2 * self.spec1d_batch_1d.mean()) ** 2)),
                log_amp=numpyro.sample("log_amp_1d", dist.Normal(-3.0, 1.0)),
                log_spec_scale=numpyro.sample("log_scale_1d", dist.Normal(0.0, 1.0)),
            )
            params_2d = dict(
                jitter=numpyro.sample("jitter_2d", dist.HalfNormal(1e-6)),
                log_amp=numpyro.sample("log_amp_2d", dist.Normal(-3.0, 1.0)),
                log_spat_scale=numpyro.sample("log_spat_scale_2d", dist.Normal(0.0, 1.0)),
                log_spec_scale=numpyro.sample("log_spec_scale_2d", dist.Normal(3.0, 1.0)),
                mean=numpyro.sample("mean_2d", dist.Normal(0.0, 1 / len(self.spat))),
            )
            gp_1d, gp_2d = self._build_host_gp(params_1d=params_1d, params_2d=params_2d)
            numpyro.sample("y_1d", gp_1d.numpyro_dist(), obs=self.spec1d_batch_1d)
            numpyro.sample("y_2d", gp_2d.numpyro_dist(), obs=self.spec2d_batch_2d.ravel())

            # Likelihood
            y_host_1d = gp_1d.predict(y=self.spec1d_batch_1d, X_test=self.X_host[:, 1][:, None])
            y_host_2d = gp_2d.predict(
                y=self.spec2d_batch_2d.ravel() - self.host_flux_prior(gp_2d.X), X_test=self.X_host
            ) + self.host_flux_prior(self.X_host)
            y_host_2d_norm = y_host_2d / jnp.sum(y_host_2d, axis=0)[None, :]
            y_host = numpyro.deterministic("y_host", y_host_1d * y_host_2d_norm)
            noise = numpyro.sample("noise", dist.HalfNormal(1e2))
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
        self.mcmc_samples = mcmc.get_samples()

    def _build_host_gp(self, params_1d: dict = {}, params_2d: dict = {}) -> Tuple[GaussianProcess, GaussianProcess]:
        """
        Build the Gaussian Process for the 1D host galaxy spectra and 2D host galaxy spatial profiles.
        """
        mean_1d = params_1d.get("mean", jnp.float64(0))
        jitter_1d = params_1d.get("jitter", jnp.float64(1e-6))
        log_amp_1d = params_1d.get("log_amp", jnp.float64(3.0))
        log_spec_scale_1d = params_1d.get("log_spec_scale", jnp.log10(self.spec_resln))
        kernel_1d = 10**log_amp_1d * transforms.Linear(10 ** (-log_spec_scale_1d), kernels.ExpSquared())
        gp_1d = GaussianProcess(kernel=kernel_1d, X=self.X_batch_1d[:, None], diag=jitter_1d, mean=mean_1d)

        mean_2d = params_2d.get("mean", jnp.float64(0))
        jitter_2d = params_2d.get("jitter", jnp.float64(1e-6))
        log_amp_2d = params_2d.get("log_amp", jnp.float64(-3.0))
        log_spat_scale_2d = params_2d.get("log_spat_scale", jnp.log10(self.spec_resln * self.batch_2d[0]))
        log_spec_scale_2d = params_2d.get("log_spec_scale", jnp.log10(self.spat_resln * self.batch_2d[1]))
        kernel_2d = 10**log_amp_2d * transforms.Linear(
            10 ** (-jnp.asarray([log_spat_scale_2d, log_spec_scale_2d])), kernels.ExpSquared()
        )
        gp_2d = GaussianProcess(kernel=kernel_2d, X=self.X_batch_2d, diag=jitter_2d, mean=mean_2d)

        return gp_1d, gp_2d
