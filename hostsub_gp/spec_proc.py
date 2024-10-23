# hostsub_gp/spec_proc.py

__all__ = ["Spec2D"]

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ._plt_config import plt
from .gp import gp
from .host_prof import HostProfile


class Spec2D:
    def __init__(
        self,
        spec2d: np.ndarray,  # 2D spectrum (spatial x spectral)
        spat: np.ndarray,  # spatial grids
        spec: np.ndarray,  # spectral grids
        slit_wid: float = 1.0,  # arcsec
        slit_len: float = 10.0,  # arcsec
        spat_resln: float = 1.0,  # arcsec, FWHM/seeing
        spec_resln: float = 7.5,  # LRIS, 1'' slit
        mask_wid: float = 2.0,  # in seeing, mask the trace of the source
        batch_1d: int = 2,  # batch size for modeling 1D host spectra
        batch_2d: tuple = (2, 50),  # batch size for modeling slowing varying host profiles
        show: bool = False,
    ):
        assert spec2d.shape == (spat.size, spec.size), "spec2d shape mismatch"
        self.spec2d = spec2d
        self.spat = spat
        self.spec = spec
        self.slit_wid = slit_wid
        self.slit_len = slit_len
        self.spat_resln = spat_resln
        self.spec_resln = spec_resln
        self.mask_wid = mask_wid

        # The 2D grids for the raw data
        print(f"Loading the 2D spectrum with the shape: {self.spec2d.shape}")
        spat_grid2d, spec_grid2d = np.meshgrid(self.spat, self.spec)
        self.X = np.stack([spat_grid2d.ravel(), spec_grid2d.ravel()], axis=-1)
        # Mask the trace from the source (|spat| < seeing * mask_wid)
        self.mask = np.abs(self.spat) < self.spat_resln * self.mask_wid
        mask_2d = np.tile(self.mask, (self.spec.size, 1)).T
        self.X_host = self.X[~mask_2d.ravel()]

        # The batched 1D grids for the host galaxy spectra (i.e., outside the mask)
        print(f"Batching the 1D galaxy spectrum (outside the mask) with the size: {batch_1d}")
        spec_batch_1d_idx = np.array_split(np.arange(self.spec.size), self.spec.size // batch_1d)
        spec_batch_1d = [self.spec[spec_batch] for spec_batch in spec_batch_1d_idx]
        # new central wavelength in each batch: mean of the batch
        self.spec_batch_1d = np.array([spec_batch.mean() for spec_batch in spec_batch_1d])
        # new values: mean of the batch
        self.spec1d_batch_1d = np.array(
            [self.spec2d[:, spec_batch][~self.mask].mean() for spec_batch in spec_batch_1d_idx]
        )
        print("Batched 1D galaxy spectrum:", self.spec1d_batch_1d.shape)

        # The batched 2D grids for the host galaxy spatial profiles
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
        # new coordinates: mean of the batch
        self.spat_batch_2d = np.array([spat_batch.mean() for spat_batch in spat_batch_2d])
        self.spec_batch_2d = np.array([spec_batch.mean() for spec_batch in spec_batch_2d])
        spat_batch_2d_grid2d, spec_batch_2d_grid2d = np.meshgrid(self.spat_batch_2d, self.spec_batch_2d)
        self.X_batch_2d = np.stack([spat_batch_2d_grid2d.ravel(), spec_batch_2d_grid2d.ravel()], axis=-1)
        # new values: mean of the batch
        self.spec2d_batch_2d = np.array(
            [
                [self.spec2d[spat_batch, :][:, spec_batch].mean() for spec_batch in spec_batch_2d_idx]
                for spat_batch in spat_batch_2d_idx
            ]
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
                vmin=np.nanpercentile(self.spec2d, 5),
                vmax=np.nanpercentile(self.spec2d, 99),
                extent=[self.spec_batch_2d[0], self.spec_batch_2d[-1], self.spat_batch_2d[0], self.spat_batch_2d[-1]],
            )
            ax[1].axhline(0, color="red", linestyle="--")
            # Plot the 1D batched spectrum
            ax[2].plot(self.spec_batch_1d, self.spec1d_batch_1d)
            ax[0].set_xlabel("Spec (Angstrom)")
            for ax_ in ax[:2]:
                ax_.set_aspect("auto")
                ax_.set_ylabel("Spat (arcsec)")
            ax[2].set_ylabel("Counts")
            plt.show()

    def build_host_prior(self) -> None:
        """
        Build the prior of the host galaxy using Gaussian Process regression.
        """
        host_prof = HostProfile(
            imgs=None,
            flts=None,
            slit_params={"slit_wid": self.slit_wid, "slit_len": self.slit_len},
            center_ra=None,
            center_dec=None,
            show=False,
        )
        self.host_flux_prior = host_prof.model_host_profile_prior(
            spat=self.spat, spec=self.spec, optimization=True
        )