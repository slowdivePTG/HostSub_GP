# hostsub_gp/spec_wrapper.py
# import numpy as np

import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)

from functools import partial
from ._utils import msgs

from jax._src.typing import ArrayLike, Array


class SpecWrapper:
    """A wrapper for the 1D and 2D spectra."""

    def __init__(
        self, points: ArrayLike | tuple[ArrayLike, ArrayLike], values: ArrayLike = None, values_err: ArrayLike = None
    ):
        """
        Initialize the SpecWrapper object.

        Parameters
        ----------
        points : ArrayLike | tuple[ArrayLike, ArrayLike]
            The coordinates of the spectrum.
        values : ArrayLike, optional
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

        self.shape = self.spec_img.shape

        # Loading the values and errors
        if values is None:
            # No values are provided, only coordinates are loaded
            self.Y = self.Yerr = None
            self.y = self.yerr = None
        else:
            if not (((values.ndim == 1) | (values.ndim == 2)) & (values.shape == self.spec_img.shape)):
                raise ValueError("Invalid shape of the input values.")
            if values_err is not None:
                if values.shape != values_err.shape:
                    raise ValueError("Values and errors shape mismatch.")
            Y = jnp.array(values)
            if values_err is None:
                Yerr = jnp.zeros_like(Y)
                # msgs.warning("No error is provided. Assuming the errors are zeros.")
            else:
                Yerr = jnp.array(values_err)
            self.Y = jnp.where(jnp.isfinite(Yerr), Y, jnp.nan)
            self.Yerr = jnp.where(jnp.isfinite(Yerr), Yerr, jnp.nan)

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

        if self.Y is None:
            raise ValueError("sigma_clip requires non-empty spectra.")

        def clip(Y: Array, Yerr: Array) -> tuple[Array, Array]:
            """
            Sigma clipping for a batch of the spectrum.
            """
            from astropy.stats import mad_std

            Y_meds = jnp.nanmedian(Y)
            Y_stds = mad_std(Y[jnp.isfinite(Y)])

            if clip_cr:  # Only remove positive outliers
                sigma_mask = (Y - Y_meds) <= (sigma * Y_stds)
            else:  # Remove both positive and negative outliers
                sigma_mask = (jnp.abs(Y - Y_meds) <= (sigma * Y_stds)) & jnp.isfinite(Y)

            Y_clipped = jnp.where(sigma_mask, Y, jnp.nan)
            Yerr_clipped = jnp.where(sigma_mask, Yerr, jnp.nan)
            return Y_clipped, Yerr_clipped

        Y_target = jnp.array(self.Y)
        masked_init = ~jnp.isfinite(self.Y)
        Yerr_target = jnp.array(self.Yerr)

        if batch_idx is None:
            if self.Y.ndim == 1:
                # Calculate the means and standard deviations over the entire spectrum
                batch_idx = ([jnp.arange(self.shape[0])],)
            else:
                # Calculate the means and standard deviations at each wavelength (for all spatial pixels)
                batch_idx = ([jnp.arange(self.shape[0])], [jnp.array(i) for i in jnp.arange(self.shape[1])])

        if self.Y.ndim == 1:
            for spec_idx in batch_idx[0]:
                Y_target = Y_target.at[spec_idx].set(clip(self.Y[spec_idx], self.Yerr[spec_idx])[0])
                Yerr_target = Yerr_target.at[spec_idx].set(clip(self.Y[spec_idx], self.Yerr[spec_idx])[1])
        else:
            for spat_idx in batch_idx[0]:
                for spec_idx in batch_idx[1]:
                    if (spat_idx.ndim == 1) & (spec_idx.ndim == 1):
                        # Both spat_idx and spec_idx are lists
                        Y_clip, Yerr_clip = clip(self.Y[spat_idx, :][:, spec_idx], self.Yerr[spat_idx, :][:, spec_idx])
                        Y_target = Y_target.at[jnp.ix_(spat_idx, spec_idx)].set(Y_clip)
                        Yerr_target = Yerr_target.at[jnp.ix_(spat_idx, spec_idx)].set(Yerr_clip)
                    else:
                        # Either spat_idx or spec_idx is a scalar
                        Y_clip, Yerr_clip = clip(self.Y[spat_idx, :][:, spec_idx], self.Yerr[spat_idx, :][:, spec_idx])
                        Y_target = Y_target.at[(spat_idx, spec_idx)].set(Y_clip)
                        Yerr_target = Yerr_target.at[(spat_idx, spec_idx)].set(Yerr_clip)

        masked_final = ~jnp.isfinite(Y_target)
        msgs.info(f"Sigma clipped {masked_final.sum() - masked_init.sum()} pixels")

        return SpecWrapper(points=(self.spat, self.spec), values=Y_target, values_err=Yerr_target)

    def fill_nan(self) -> "SpecWrapper":
        """
        Fill the NaN values in the spectrum by interpolation.
        """
        from scipy.interpolate import griddata

        if self.Y is None:
            raise ValueError("Filling NaN requires non-empty spectra.")

        Y = jnp.array(self.Y)
        Yerr = jnp.array(self.Yerr)
        valid = ~jnp.isnan(Y)

        if jnp.all(valid):
            return self

        x, y = jnp.indices(self.shape)

        # Interpolate
        Y_filled = griddata((x[valid], y[valid]), Y[valid], (x, y), method="linear")
        Y_err_filled = griddata((x[valid], y[valid]), Yerr[valid], (x, y), method="linear")
        filled_count = jnp.sum(~valid)

        # Ideally use a logging mechanism here
        msgs.info(f"Filled {filled_count} NaN pixels")

        return SpecWrapper(points=(self.spat, self.spec), values=Y_filled, values_err=Y_err_filled)

    def marginalize(
        self, margin_type: str = "mean", weights: str | ArrayLike = None, sigma_clip: float = 5.0, nan_threshold: float = 0.1
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
        nan_threshold : float, optional
            Threshold for the fraction of NaN values in a column. Columns with NaN fraction > nan_threshold will be masked. Default is 0.1.

        Returns
        -------
        SpecWrapper
            The marginalized 1D spectrum.
        """
        if self.Y is None:
            raise ValueError("Marginalizing requires non-empty spectra.")

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
        Y_meds = jnp.nanmedian(self.Y, axis=0)
        Y_stds = jnp.nanstd(self.Y, axis=0, ddof=1)

        # Create the mask for sigma clipping
        # Broadcasting to compare each column with its own mean and std
        deviations = jnp.abs(self.Y - Y_meds[None, :])
        sigma_masks = deviations <= (sigma_clip * Y_stds[None, :])
        valid_masks = jnp.isfinite(self.Y)
        combined_mask = sigma_masks & valid_masks

        # Calculate weighted means
        weights = jnp.where(combined_mask, w, 0)
        weighted_values = jnp.where(combined_mask, self.Y * weights, 0)

        mean_value = jnp.sum(weighted_values, axis=0) / jnp.sum(weights, axis=0)

        # Calculate errors
        weighted_errors = jnp.where(combined_mask, (self.Yerr * weights) ** 2, 0)
        mean_value_err = jnp.sqrt(jnp.sum(weighted_errors, axis=0) / jnp.sum(weights, axis=0) ** 2)

        # Mask columns with NaN fraction > nan_threshold
        nan_fraction = jnp.sum(~jnp.isfinite(self.Y), axis=0) / self.shape[0]
        mean_value = jnp.where(nan_fraction > nan_threshold, jnp.nan, mean_value)
        mean_value_err = jnp.where(nan_fraction > nan_threshold, jnp.nan, mean_value_err)

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
        if self.Y is None:
            raise ValueError("Subtraction requires non-empty spectra.")

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

    def convolve(self, kernel_wid: float | ArrayLike) -> "SpecWrapper":
        """
        Convolve the spectrum with a kernel.

        Parameters
        ----------
        kernel_wid : float | ArrayLike
            The width (FWHM in pixel) of the Gaussian kernel

        Returns
        -------
        SpecWrapper
            The seeing-matched spectrum.
        """
        if self.Y is None:
            raise ValueError("Convolution requires non-empty spectra.")

        def gaussian_filter(y: Array, sigma: Array) -> Array:
            """
            Vectorized 1D Gaussian filter
            """

            @partial(jax.jit, static_argnums=(2,))
            def jax_gaussian_filter1d(y: Array, sigma: float, max_kernel_size=30) -> Array:
                """
                JAX implementation of 1D Gaussian filter
                """
                # Create Gaussian kernel
                radius = max_kernel_size // 2
                x = jnp.arange(-radius, radius + 1)
                kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
                kernel = kernel / jnp.sum(kernel)

                # Pad and convolve
                pad_width = len(kernel) // 2
                y_padded = jnp.pad(y, (pad_width, pad_width), mode="constant")
                return jnp.convolve(y_padded, kernel, mode="valid")

            return jax.vmap(jax_gaussian_filter1d, in_axes=(1, 0), out_axes=1)(y, sigma)

        if isinstance(kernel_wid, float):
            kernel_sigma = jnp.ones_like(self.spec) * kernel_wid / 2.355
        elif isinstance(kernel_wid, ArrayLike):
            if kernel_wid.size != self.spec.size:
                raise ValueError(
                    f"The length of the kernel_wid array ({kernel_wid.size}) does not match the spectrum ({self.spec.size})."
                )
            kernel_sigma = kernel_wid / 2.355
        else:
            raise ValueError("Invalid kernel width.")

        Y_conv = gaussian_filter(self.Y, kernel_sigma)
        Yerr_conv = gaussian_filter(self.Yerr, kernel_sigma)

        return SpecWrapper(points=(self.spat, self.spec), values=Y_conv, values_err=Yerr_conv)
