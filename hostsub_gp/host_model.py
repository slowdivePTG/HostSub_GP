# hostsub_gp/host_model.py

__all__ = ["HostProfile"]

import numpy as np

import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)


from .gp import GP
from .host_image import PS1Image, SDSSImage, LSImage
from .host_image import ImageProduct
from ._utils import plt, msgs
from ._utils._plt import show_and_save

from typing import Callable, Optional, Literal, Type, Any
from jax._src.typing import Array
from .spec_base import SpecModelP


class HostProfile:
    def __init__(
        self,
        filters: str | list[str],
        wv_eff: list[float],
        spat_slit: list[Array],
        counts_slit: list[Array],
        counts_err_slit: list[Array],
        spec_model: Optional[SpecModelP] = None,
        slit_len: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ):
        """
        Estimate the host galaxy spatial profile from the 2D spectrum.

        Parameters
        ----------
        flts : str
            Filters to load the images.
        wv_eff : list[float]
            Effective wavelengths.
        spat_slit : list[Array]
            Spatial coordinates along the slit.
        counts_slit : list[Array]
            Counts along the slit.
        counts_err_slit : list[Array]
            Errors of the counts along the slit.
        spec_model : SpecModelP, optional
            SpecModel object.
        slit_len : float, optional
            Slit length in arcsec.
        pixel_scale : float, optional
            Pixel scale in arcsec.
        """
        self.filters = filters
        self.wv_eff = wv_eff

        prof_slit, prof_err_slit = [], []

        if spec_model is not None:
            slit_len = (
                spec_model.spat_edges["slit"][1] - spec_model.spat_edges["slit"][0]
            )
            pixel_scale = spec_model.pixel_scale
            mask_offset = spec_model.mask_offset
        elif slit_len is None or pixel_scale is None:
            raise ValueError("Slit length and pixel scale are required")
        else:
            mask_offset = 0

        for k in range(len(self.filters)):
            if spec_model is not None:
                host_left = (
                    spec_model.spat_edges["host"][0],
                    spec_model.spat_edges["mask"][0],
                )
                host_right = (
                    spec_model.spat_edges["mask"][1],
                    spec_model.spat_edges["host"][1],
                )

                sky_left = (
                    spec_model.spat_edges["slit"][0],
                    max(
                        spec_model.spat_edges["sky"][0],
                        spec_model.spat_edges["slit"][0],
                    ),
                )
                sky_right = (
                    min(
                        spec_model.spat_edges["sky"][1],
                        spec_model.spat_edges["slit"][1],
                    ),
                    spec_model.spat_edges["slit"][1],
                )

                xi = counts_slit[k]
                xi_err = counts_err_slit[k]
                xi_sky_mean = (
                    bound_sum(spat_slit[k], xi, x_bound=sky_left)
                    + bound_sum(spat_slit[k], xi, x_bound=sky_right)
                ) / ((sky_left[1] - sky_left[0]) + (sky_right[1] - sky_right[0]))
                xi_host_mean = (
                    bound_sum(spat_slit[k], xi, x_bound=host_left)
                    + bound_sum(spat_slit[k], xi, x_bound=host_right)
                ) / ((host_left[1] - host_left[0]) + (host_right[1] - host_right[0]))
                prof_slit.append(
                    (xi - xi_sky_mean)
                    / (xi_host_mean - xi_sky_mean)
                    # / (spec_model.host_wid - spec_model.mask_wid)
                    / ((host_left[1] - host_left[0]) + (host_right[1] - host_right[0]))
                    * pixel_scale
                )
                prof_err_slit.append(
                    xi_err
                    / (xi_host_mean - xi_sky_mean)
                    # /(spec_model.host_wid - spec_model.mask_wid)
                    / ((host_left[1] - host_left[0]) + (host_right[1] - host_right[0]))
                    * pixel_scale
                )

            else:  # No mask
                xi = counts_slit[k] / np.sum(counts_slit[k]) / pixel_scale
                xi_err = counts_err_slit[k] / np.sum(counts_slit[k]) / pixel_scale
                prof_slit.append(xi)
                prof_err_slit.append(xi_err)

        # trim the slit
        if spec_model is not None:
            self.host_wid = (
                spec_model.spat_edges["host"][1] - spec_model.spat_edges["host"][0]
            )  # Host width in pixels
        else:
            self.host_wid = slit_len  # Host width in pixels - if not specified, using the slit length

        host_idx = [
            np.argwhere(
                np.abs(spat_slit[k] - mask_offset) <= np.ceil(self.host_wid / 2)
            ).ravel()
            for k in range(len(self.filters))
        ]

        self.prof_slit = [prof_slit[k][host_idx[k]] for k in range(len(self.filters))]
        self.prof_err_slit = [
            prof_err_slit[k][host_idx[k]] for k in range(len(self.filters))
        ]
        self.spat_slit = [spat_slit[k][host_idx[k]] for k in range(len(self.filters))]
        self.wv_slit = [
            np.ones_like(host_idx[k]) * self.wv_eff[k] for k in range(len(self.filters))
        ]
        self.prof = jnp.concatenate(self.prof_slit)
        self.prof_err = jnp.concatenate(self.prof_err_slit)
        self.X = jnp.stack(
            [jnp.concatenate(self.spat_slit), jnp.concatenate(self.wv_slit)], axis=-1
        )

    @staticmethod
    def _suppress_fitsfixed_warning(func):
        """
        Decorator to suppress FITSFixedWarning in the decorated function.
        """
        import warnings
        from functools import wraps
        from astropy.wcs import FITSFixedWarning

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Suppress FITSFixedWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FITSFixedWarning)
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    @_suppress_fitsfixed_warning
    def load_archival_images(
        spec_model: Optional[SpecModelP] = None,
        center_ra: Optional[float] = None,
        center_dec: Optional[float] = None,
        slit_len: Optional[float] = None,
        slit_wid: float = 1.0,
        position_angle: Optional[float] = None,
        filters: Optional[str | list] = None,
        survey: Literal["PS1", "LS", "any"] = "PS1",
    ) -> list[ImageProduct]:
        """
        Load, rotate, and resample archival images from PS1, LS, and SDSS
        """
        from operator import attrgetter

        if spec_model is not None:
            center_ra = spec_model.center_ra
            center_dec = spec_model.center_dec
            slit_len = (
                spec_model.spat_edges["slit"][1] - spec_model.spat_edges["slit"][0]
            )
            slit_wid = spec_model.slit_wid
            position_angle = spec_model.position_angle
        else:
            if center_ra is None or center_dec is None:
                raise ValueError("Coordinates are required")
            if position_angle is None:
                raise ValueError("Position angle is required")

        if filters is None:
            # Load all filters
            filters = "grizy"

        # Data
        img_product_list = []

        def _load_images(
            image_class: Type[SDSSImage | LSImage | PS1Image],
            filters: str,
            center_ra: float,
            center_dec: float,
        ) -> bool:
            """
            Load images from the specified image class.
            """
            image = image_class(ra=center_ra, dec=center_dec, filters=filters)
            image.get_cutout()
            imgs, headers = image.load()

            img_products = []

            for k in range(len(image.filters)):
                img, header = imgs[k], headers[k]
                flt = image.filters[k]
                wv_eff = image.wv_eff_dict[flt]
                img_products.append(
                    ImageProduct(
                        center_ra=center_ra,
                        center_dec=center_dec,
                        slit_len=slit_len,
                        slit_wid=slit_wid,
                        position_angle=position_angle,
                        img=img,
                        header=header,
                        flt=flt,
                        wv_eff=wv_eff,
                    )
                )

            # No images found
            if len(img_products) == 0:
                return False

            img_product_list.extend(img_products)

            return True

        # Try to load SDSS u-band image
        _load_images(
            SDSSImage,
            "".join(flt for flt in "u" if flt in filters),
            center_ra,
            center_dec,
        )

        # Try LS images first if requested
        if survey == "LS" or survey == "any":
            _ls_loaded = _load_images(
                LSImage,
                "".join(flt for flt in "griz" if flt in filters),
                center_ra,
                center_dec,
            )
            if not _ls_loaded:
                raise ValueError("No LS images found")
        if (survey == "PS1") or ((survey == "any") and not _ls_loaded):
            _ps1_loaded = _load_images(
                PS1Image,
                "".join(flt for flt in "grizy" if flt in filters),
                center_ra,
                center_dec,
            )
            if not _ps1_loaded:
                raise ValueError("No PS1 images found")

        # TODO: Load acquisition images (optional)

        return sorted(img_product_list, key=attrgetter("wv_eff"))

    @classmethod
    @_suppress_fitsfixed_warning
    def from_archival(
        cls,
        img_products: list[ImageProduct],
        spec_model: Optional[SpecModelP] = None,
        slit_len: Optional[float] = None,
        slit_wid: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        dseeing: Optional[float] = None,
        alpha: Optional[float] = 0.2,
        verbose: bool = False,
    ):
        """
        Load archival images from PS1 and SDSS and estimate the host galaxy spatial profile.

        Parameters
        ----------
        spec_model : any, optional
            SpecModel object.
        center_ra : float, optional
            Right ascension of the object.
        center_dec : float, optional
            Declination of the object.
        slit_len : float, optional
            Slit length in arcsec.
        slit_wid : float, optional
            Slit width in arcsec.
        pixel_scale : float, optional
            Pixel scale in arcsec of the 2D spectrum (not the archival images)
        position_angle : float, optional
            Position angle of the slit.
        filters : str or list, optional
            Filters to load the images.
        survey : str, optional
            Survey to use for loading images. Options are 'PS1' or 'LS'.
        dseeing : float, optional
        alpha: float, optional, default = 0.2
        """
        from scipy.ndimage import gaussian_filter

        if spec_model is not None:
            slit_wid = spec_model.slit_wid
            slit_len = (
                spec_model.spat_edges["slit"][1] - spec_model.spat_edges["slit"][0]
            )
            pixel_scale = spec_model.pixel_scale
        else:
            assert slit_len is not None and slit_wid is not None
            assert pixel_scale is not None

        # Seeing correction
        if dseeing is None:
            dseeing = 0.0
        else:
            dseeing = np.abs(dseeing)  # By default, dseeing is negative
        assert isinstance(dseeing, float), "dseeing is not properly assigned"

        # Spatial coordinates along the slit
        spat_slit = [img_product.spat_slit for img_product in img_products]
        # Counts along the slit
        counts_slit, counts_err_slit = [], []

        for img_product in img_products:
            if dseeing > 0:
                # Convolve the images with a Gaussian kernel
                assert spec_model is not None, (
                    "SpecModel is required for dseeing correction"
                )
                dseeing_wv = dseeing * (
                    img_product.wv_eff / spec_model.spec.mean()
                ) ** (-alpha)
                img = gaussian_filter(
                    img_product.img, sigma=dseeing_wv / spec_model.pixel_scale / 2.355
                )
                if verbose:
                    msgs.info(
                        f"Convolving {img_product.flt} with a {dseeing_wv:.2f} arcsec kernel "
                        + f"(sigma = {dseeing_wv / spec_model.pixel_scale / 2.355:.2f} pixels)"
                    )
            else:
                img = img_product.img

            counts_slit.append(
                bound_mean_img(
                    img_product.spat_slit_wid,
                    img,
                    x_bound=(-slit_wid / 2, slit_wid / 2),
                )
            )
            # Estimate the error: standard deviation of the residuals (count at each pixel - average count)
            img_slit = img_product.img[
                :, np.abs(img_product.spat_slit_wid) < slit_wid / 2
            ]
            # slit_wid_pix = int(np.round(slit_wid / img_product.pixel_scale))
            err1 = np.nanstd(img_slit, axis=1, ddof=1) #/ np.sqrt(slit_wid_pix)
            # Smooth the error: convolution with a boxcar filter
            noise_smooth_kernel = 3
            if noise_smooth_kernel is not None:
                err1 = (np.convolve(err1**2, np.ones(noise_smooth_kernel) / noise_smooth_kernel, mode="same")) ** 0.5
            
            err2 = np.ones_like(counts_slit[-1]) * img_product.err
            counts_err_slit.append(np.where(err1 < err2, err1, err2))

        flts = [img_product.flt for img_product in img_products]
        wv_effs = [img_product.wv_eff for img_product in img_products]

        return cls(
            filters=flts,
            wv_eff=wv_effs,
            spat_slit=spat_slit,
            counts_slit=counts_slit,
            counts_err_slit=counts_err_slit,
            spec_model=spec_model,
            slit_len=slit_len,
            pixel_scale=pixel_scale,
        )

    @msgs.timer
    def model_host_profile_prior(
        self, spat_resln: float = 1.0, params_init: Optional[dict] = None, **kwargs
    ) -> Callable[[Any], Array | tuple[Array, Array]]:
        """
        Model the host galaxy spatial profile using Gaussian Process regression.

        Parameters
        ----------
        spat_resln : float, optional
            Spatial resolution (seeing) in arcsec.

        Returns
        -------
        host_prior : Callable[[Array], tuple[Array, Array]]
            A function that returns the mean and variance of the host profile.
        """

        if params_init is not None:
            assert "log_scale" in params_init, "log_scale is required in params_init"
            assert "log_amp" in params_init, "log_amp is required in params_init"
            assert "mean" in params_init, "mean is required in params_init"

        # No prior photometric data
        if len(self.filters) == 0:

            def host_prior_flat(
                x: Array,
            ) -> tuple[Array, Array]:  # constant, variance = 0
                return jnp.array(1 / self.host_wid, dtype=jnp.float32), jnp.array(
                    0, dtype=jnp.float32
                )

            self._gp_params = {}
            host_prior = host_prior_flat

        # Single band - no wavelength dependence
        elif len(self.filters) == 1:
            if params_init is None:
                params_init = dict(
                    log_amp=np.float64(-3),
                    log_scale=np.log10(spat_resln),
                    mean=np.float64(1 / self.host_wid),
                )

            params_limit = dict(log_scale=np.log10([spat_resln / 2.355, np.inf]))
            gp_host_prior = GP(
                kernel_type="HostProfie",
                X=self.X[:, :1],  # Spatial coordinate only
                y=self.prof,
                yerr=self.prof_err,
                params_init=params_init,
                params_limit=params_limit,
                optimization=True,
            )
            self._gp_params = gp_host_prior.params

            def host_prior_single(x: Array) -> tuple[Array, Array]:
                return gp_host_prior.predict(X_test=x[:, :1], return_var=True)

            host_prior = host_prior_single

        # Multiple bands - wavelength dependence
        else:
            if params_init is None:
                params_init = dict(
                    log_amp=np.ones((2, 2)) * -2,
                    log_scale=np.log10([[spat_resln, spat_resln], [1e5, 1e4]]),
                    mean=np.float64(1 / self.host_wid),
                )

            params_limit = dict(
                log_scale=np.log10(
                    [
                        [
                            [spat_resln / 2.355, spat_resln / 2.355],
                            [1e3, 1e3],
                        ],  # lower bound
                        [
                            [np.inf, spat_resln * 2],
                            [np.inf, np.inf],
                        ],  # upper bound
                    ]
                )
            )
            gp_host_prior = GP(
                kernel_type="HostProfile",
                X=self.X,
                y=self.prof,
                yerr=self.prof_err,
                params_init=params_init,
                params_limit=params_limit,
                optimization=True,
            )
            self._gp_params = gp_host_prior.params

            def host_prior_multi(x: Array) -> tuple[Array, Array]:
                return gp_host_prior.predict(X_test=x, return_var=True)

            host_prior = host_prior_multi

        self._plot_host_profile(host_prior, **kwargs)

        return host_prior

    @show_and_save
    def _plot_host_profile(self, host_prior) -> None:
        """
        Plot the host galaxy spatial profile.
        """
        from matplotlib.colors import Normalize

        _, ax = plt.subplots(
            len(self.filters),
            1,
            figsize=(6, 2 * len(self.filters)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        ax = np.atleast_1d(ax)
        cmap = plt.cm.get_cmap("coolwarm")
        norm = Normalize(vmin=0, vmax=len(self.filters) - 1)
        for k in range(len(self.filters)):
            ax[k].plot(
                self.spat_slit[k],
                self.prof_slit[k],
                label=f"{self.filters[k]}",
                color=cmap(norm(k)),
            )
            ax[k].plot(
                self.spat_slit[k],
                host_prior(jnp.stack([self.spat_slit[k], self.wv_slit[k]], axis=-1))[0],
                "--",
                color=cmap(norm(k)),
            )
            ax[k].fill_between(
                self.spat_slit[k],
                self.prof_slit[k] - self.prof_err_slit[k],
                self.prof_slit[k] + self.prof_err_slit[k],
                color=cmap(norm(k)),
                alpha=0.2,
            )
            ax[k].set_ylabel(r"$\mathrm{Profile}$")
            ax[k].text(
                0.05,
                0.8,
                f"{self.filters[k]}: {self.wv_eff[k]:.0f} Ang",
                color=cmap(norm(k)),
                transform=ax[k].transAxes,
            )
        ax[-1].set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")


def bound_sum(x: Array, y: Array, x_bound: Optional[tuple] = None) -> float:
    """
    Compute the mean values in a bounded region.
    """
    bin_size = jnp.append(x[1] - x[0], jnp.diff(x))
    if x_bound is None:
        x_bound = (x[0] - bin_size[0] / 2, x[-1] + bin_size[-1] / 2)
    if x_bound[1] <= x_bound[0]:
        return jnp.float32(0)
    # sum up all pixels that are fully contained in the region
    idx_center = (x > x_bound[0] + bin_size[0] / 2) & (
        x < x_bound[1] - bin_size[-1] / 2
    )
    sum_center = jnp.sum(y[idx_center] * bin_size[idx_center])

    # leftmost pixel that is partially contained in the region (if any)
    idx_left = jnp.where(x >= x_bound[0] - bin_size[0] / 2)[0]
    if idx_left.size > 0:
        y_left = y[idx_left[0]]
        frac_left = x[idx_left[0]] - (x_bound[0] - bin_size[0] / 2)
        sum_left = y_left * frac_left
    else:
        raise ValueError("Invalid left bound")

    # rightmost pixel that is partially contained in the region (if any)
    idx_right = jnp.where(x <= x_bound[1] + bin_size[-1] / 2)[-1]
    if idx_right.size > 0:
        y_right = y[idx_right[-1]]
        frac_right = (x_bound[1] + bin_size[-1] / 2) - x[idx_right[-1]]
        sum_right = y_right * frac_right
    else:
        raise ValueError("Invalid right bound")
    return sum_center + sum_left + sum_right


def bound_mean(x: Array, y: Array, x_bound: tuple) -> float:
    """
    Compute the sum in a bounded region.
    """
    return bound_sum(x, y, x_bound) / (x_bound[1] - x_bound[0])


def bound_mean_img(x: Array, y_img: Array, x_bound: tuple) -> Array:
    """
    Apply bound_mean to each row of the image.
    """
    return jax.vmap(lambda y: bound_mean(x, y, x_bound))(y_img)
