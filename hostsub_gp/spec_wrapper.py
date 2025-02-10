# hostsub_gp/spec_wrapper.py
import numpy as np

import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)

from ._utils import msgs

from jax._src.typing import ArrayLike, Array


class SpecWrapper:
    """A wrapper for the 1D and 2D spectra."""

    def __init__(
        self, points: ArrayLike | tuple[ArrayLike, ArrayLike], values: ArrayLike, values_err: ArrayLike = None
    ):
        """
        Initialize the SpecWrapper object.

        Parameters
        ----------
        points : ArrayLike | tuple[ArrayLike, ArrayLike]
            The coordinates of the spectrum.
        values : ArrayLike
            The values of the spectrum.
        values_err : ArrayLike, optional
            The errors of the values.
        """
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
        Y = jnp.array(values)
        if values_err is None:
            Yerr = jnp.zeros_like(Y)
            msgs.warning("No error is provided. Assuming the errors are zeros.")
        else:
            Yerr = jnp.array(values_err)
        self.Y = jnp.where(jnp.isfinite(Yerr), Y, np.nan)
        self.Yerr = jnp.where(jnp.isfinite(Yerr), Yerr, np.nan)

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

    def sigma_clip(
        self, sigma: float = 5.0, clip_cr: bool = False, batch_idx: ArrayLike | tuple[ArrayLike, ArrayLike] = None
    ) -> "SpecWrapper":
        """
        Sigma clipping for the spectrum.

        Parameters
        ----------
        sigma : float, optional
            Sigma clipping threshold. Default is 5.
        clip_cr : bool, optional
            Whether to clip cosmic rays only (i.e., positive outliers). Default is False.
        batch_idx : list | tuple[list, list], optional
            Batch indices for sigma clipping. Default is None.

        Returns
        -------
        SpecWrapper
            The clipped spectrum.
        """

        def clip(Y: Array, Yerr: Array) -> tuple[Array, Array]:
            """
            Sigma clipping for a batch of the spectrum.
            """
            from astropy.stats import mad_std

            Y_meds = np.nanmedian(Y)
            Y_stds = mad_std(Y[np.isfinite(Y)])

            if clip_cr:  # Only remove positive outliers
                sigma_mask = (Y - Y_meds) <= (sigma * Y_stds)
            else:  # Remove both positive and negative outliers
                sigma_mask = (np.abs(Y - Y_meds) <= (sigma * Y_stds)) & np.isfinite(Y)

            Y_clipped = np.where(sigma_mask, Y, jnp.nan)
            Yerr_clipped = np.where(sigma_mask, Yerr, jnp.nan)
            return Y_clipped, Yerr_clipped

        Y_target = np.array(self.Y)
        masked_init = ~np.isfinite(self.Y)
        Yerr_target = np.array(self.Yerr)

        if batch_idx is None:
            if self.Y.ndim == 1:
                # Calculate the means and standard deviations over the entire spectrum
                batch_idx = ([np.arange(self.shape[0])],)
            else:
                # Calculate the means and standard deviations at each wavelength (for all spatial pixels)
                batch_idx = ([np.arange(self.shape[0])], [np.array(i) for i in np.arange(self.shape[1])])

        if self.Y.ndim == 1:
            for spec_idx in batch_idx[0]:
                Y_target[spec_idx], Yerr_target[spec_idx] = clip(self.Y[spec_idx], self.Yerr[spec_idx])
        else:
            for spat_idx in batch_idx[0]:
                for spec_idx in batch_idx[1]:
                    if (spat_idx.ndim == 1) & (spec_idx.ndim == 1):
                        # Both spat_idx and spec_idx are lists
                        Y_target[np.ix_(spat_idx, spec_idx)], Yerr_target[np.ix_(spat_idx, spec_idx)] = clip(
                            self.Y[spat_idx, :][:, spec_idx], self.Yerr[spat_idx, :][:, spec_idx]
                        )
                    else:
                        # Either spat_idx or spec_idx is a scalar
                        Y_target[spat_idx, spec_idx], Yerr_target[spat_idx, spec_idx] = clip(
                            self.Y[spat_idx, :][:, spec_idx], self.Yerr[spat_idx, :][:, spec_idx]
                        )

        masked_final = ~np.isfinite(Y_target)
        msgs.info(f"Sigma clipped {masked_final.sum() - masked_init.sum()} pixels")

        return SpecWrapper(points=(self.spat, self.spec), values=Y_target, values_err=Yerr_target)

    def fill_nan(self) -> "SpecWrapper":
        """
        Fill the NaN values in the spectrum by interpolation.
        """
        from scipy.interpolate import griddata

        Y_masked = np.ma.masked_invalid(self.Y)
        Y_err_masked = np.ma.masked_invalid(self.Yerr)
        valid = ~Y_masked.mask

        x, y = np.indices(self.shape)

        # Interpolate
        Y_filled = griddata((x[valid], y[valid]), Y_masked[valid], (x, y), method="linear")
        Y_err_filled = griddata((x[valid], y[valid]), Y_err_masked[valid], (x, y), method="linear")
        msgs.info(f"Filled {np.sum(~valid)} NaN pixels")

        return SpecWrapper(points=(self.spat, self.spec), values=Y_filled, values_err=Y_err_filled)

    def marginalize(
        self, margin_type: str = "mean", weights: str | ArrayLike = None, sigma_clip: float = 5.0
    ) -> "SpecWrapper":
        """
        Marginalize the 2D spectrum along the spatial axis to obtain the 1D spectrum.

        Parameters
        ----------
        margin_type : str, optional
            Type of the marginalization: mean or sum. Default is mean.
        weights : str, optional
            Weights for the marginalization: None, ivar, snr, or an array of the weights. Default is None.
            None: no weights
            ivar: inverse variance
            snr: signal-to-noise ratio squared
        sigma_clip : float, optional
            Sigma clipping threshold for the marginalization. Default is 5.

        Returns
        -------
        SpecWrapper
            The marginalized 1D spectrum.
        """
        if (weights is None) or jnp.all(self.Yerr == 0):
            w = jnp.ones_like(self.Y)
        elif isinstance(weights, (ArrayLike, Array)):
            if weights.ndim < self.Y.ndim:
                # Broadcasting the weights to the same shape as the spectrum
                weights = jnp.tile(weights[:, None], reps=self.Y.shape[1])
            if weights.shape != self.Y.shape:
                raise ValueError(
                    f"Input weights shape {weights.shape} does not match the spectrum shape {self.Y.shape}"
                )
            w = jnp.array(weights)
        elif weights == "ivar":
            w = self.Yerr**-2
        elif weights == "snr":
            w = (self.Y / self.Yerr) ** 2
        else:
            raise ValueError("Invalid weights.")

        # Calculate the overall means and standard deviations
        Y_meds = np.nanmedian(self.Y, axis=0)
        Y_stds = np.nanstd(self.Y, axis=0, ddof=1)

        # Create the mask for sigma clipping
        # Broadcasting to compare each column with its own mean and std
        deviations = np.abs(self.Y - Y_meds[None, :])
        sigma_masks = deviations <= (sigma_clip * Y_stds[None, :])
        valid_masks = np.isfinite(self.Y)
        combined_mask = sigma_masks & valid_masks

        # Calculate weighted means
        weights = np.where(combined_mask, w, 0)
        weighted_values = np.where(combined_mask, self.Y * weights, 0)

        mean_value = np.sum(weighted_values, axis=0) / np.sum(weights, axis=0)

        # Calculate errors
        weighted_errors = np.where(combined_mask, (self.Yerr * weights) ** 2, 0)
        mean_value_err = np.sqrt(np.sum(weighted_errors, axis=0) / np.sum(weights, axis=0) ** 2)

        if margin_type == "mean":
            return SpecWrapper(points=self.spec, values=mean_value, values_err=mean_value_err)
        elif margin_type == "sum":
            return SpecWrapper(
                points=self.spec, values=mean_value * self.shape[0], values_err=mean_value_err * self.shape[0]
            )

    def subtract(self, other: "SpecWrapper") -> "SpecWrapper":
        """
        Subtract another spectrum.

        Parameters
        ----------
        other : SpecWrapper
            The other spectrum to be subtracted.

        Returns
        -------
        SpecWrapper
            The subtracted spectrum.
        """
        if ((len(other.shape) == 1) & (other.shape[-1] != self.shape[-1])) | (
            (len(other.shape) == 2) & (other.shape != self.shape)
        ):
            raise ValueError("Shape mismatch.")
        return SpecWrapper(
            points=(self.spat, self.spec),
            values=self.Y - other.Y,
            values_err=(self.Yerr**2 + other.Yerr**2) ** 0.5,
        )

    def apply_spatial_filter(self, spat_filter: ArrayLike) -> "SpecWrapper":
        """
        Creates a new spectrum by applying a spatial mask to the current spectrum.

        Parameters
        ----------
        spat_filter : ArrayLike
            Filter to apply to the spatial axis.

        Returns
        -------
        SpecWrapper
        A new spectrum containing only the data points selected by the mask
        """
        return SpecWrapper(
            points=(self.spat[spat_filter], self.spec),
            values=self.Y[spat_filter],
            values_err=self.Yerr[spat_filter],
        )

    def convolve(self, kernel: ArrayLike) -> "SpecWrapper":
        """
        Convolve the spectrum with a kernel.

        Parameters
        ----------
        kernel : ArrayLike
            The convolution kernel.

        Returns
        -------
        SpecWrapper
            The convolved spectrum.
        """
        raise NotImplementedError("Convolution is not implemented yet.")