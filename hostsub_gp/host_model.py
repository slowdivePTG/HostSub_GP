# hostsub_gp/host_profile.py

__all__ = ["HostProfile"]

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ._plt import plt
from .gp import GP
from .host_image import PS1Image, SDSSImage
from .interp import Interp2D_Grid

from typing import Callable
from jax._src.typing import Array, ArrayLike


class HostProfile:
    def __init__(
        self,
        flts: str,
        wv_eff: list[float],
        spat_slit: list[ArrayLike],
        counts_slit: list[ArrayLike],
        counts_err_slit: list[ArrayLike],
        spec_model: any = None,
        slit_len: float = None,
        pixel_scale: float = 1.0,
    ):
        """
        Estimate the host galaxy spatial profile from the 2D spectrum.

        Parameters
        ----------
        flts : str
            Filters to load the images.
        wv_eff : list[float]
            Effective wavelengths.
        spat_slit : list[ArrayLike]
            Spatial coordinates along the slit.
        counts_slit : list[ArrayLike]
            Counts along the slit.
        counts_err_slit : list[ArrayLike]
            Errors of the counts along the slit.
        spec_model : any, optional
            SpecModel object.
        slit_len : float, optional
            Slit length in arcsec.
        pixel_scale : float, optional
            Pixel scale in arcsec.
        """
        self.flts = flts
        self.wv_eff = wv_eff

        prof_slit, prof_err_slit = [], []

        for k in range(len(self.flts)):
            if spec_model is not None:
                host_left = (-spec_model.host_wid / 2, -spec_model.mask_wid / 2 + spec_model.mask_offset)
                host_right = (spec_model.mask_wid / 2 + spec_model.mask_offset, spec_model.host_wid / 2)
                sky_left = (-spec_model.slit_len / 2, max(spec_model.sky_region[0], -spec_model.slit_len / 2))
                sky_right = (min(spec_model.slit_len / 2, spec_model.sky_region[1]), spec_model.slit_len / 2)
                xi = counts_slit[k]
                xi_err = counts_err_slit[k]
                xi_sky_mean = (
                    bound_sum(spat_slit[k], xi, x_bound=sky_left) + bound_sum(spat_slit[k], xi, x_bound=sky_right)
                ) / ((sky_left[1] - sky_left[0]) + (sky_right[1] - sky_right[0]))
                xi_host_mean = (
                    bound_sum(spat_slit[k], xi, x_bound=host_left) + bound_sum(spat_slit[k], xi, x_bound=host_right)
                ) / ((host_left[1] - host_left[0]) + (host_right[1] - host_right[0]))
                prof_slit.append(
                    (xi - xi_sky_mean)
                    / (xi_host_mean - xi_sky_mean)
                    / (spec_model.host_wid - spec_model.mask_wid)
                    * spec_model.pixel_scale
                )
                prof_err_slit.append(
                    xi_err / (xi_host_mean - xi_sky_mean) / (spec_model.host_wid - spec_model.mask_wid) * spec_model.pixel_scale
                )

            else:  # No mask
                xi = counts_slit[k] / np.sum(counts_slit[k]) / pixel_scale
                xi_err = counts_err_slit[k] / np.sum(counts_slit[k]) / pixel_scale
                prof_slit.append(xi)
                prof_err_slit.append(xi_err)

        # trim the slit
        if spec_model is not None:
            self.host_wid = spec_model.host_wid  # Host width in pixels
        else:
            self.host_wid = slit_len  # Host width in pixels - if not specified, using the slit length

        host_idx = [
            np.argwhere(np.abs(spat_slit[k]) <= np.ceil(self.host_wid / 2)).ravel() for k in range(len(self.flts))
        ]

        self.prof_slit = [prof_slit[k][host_idx[k]] for k in range(len(self.flts))]
        self.prof_err_slit = [prof_err_slit[k][host_idx[k]] for k in range(len(self.flts))]
        self.spat_slit = [spat_slit[k][host_idx[k]] for k in range(len(self.flts))]
        self.wv_slit = [np.ones_like(host_idx[k]) * self.wv_eff[k] for k in range(len(self.flts))]
        self.prof = jnp.concatenate(self.prof_slit)
        self.prof_err = jnp.concatenate(self.prof_err_slit)
        self.X = jnp.stack([jnp.concatenate(self.spat_slit), jnp.concatenate(self.wv_slit)], axis=-1)

    @classmethod
    def from_archival(
        cls,
        spec_model: any = None,
        center_ra: float = None,
        center_dec: float = None,
        slit_len: float = None,
        slit_wid: float = 1.0,
        position_angle: float = None,
        filters: str | list = None,
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
        position_angle : float, optional
            Position angle of the slit.
        filters : str or list, optional
            Filters to load the images.
        """

        if spec_model is not None:
            center_ra = spec_model.center_ra
            center_dec = spec_model.center_dec
            slit_len = spec_model.slit_len
            slit_wid = spec_model.slit_wid
            position_angle = spec_model.position_angle
        else:
            if center_ra is None or center_dec is None:
                raise ValueError("Coordinates are required")
            if position_angle is None:
                raise ValueError("Position angle is required")

        if filters is None:
            # Load all filters
            filters = "ugrizy"

        data_list, header_list = [], []
        wv_eff = []
        flts = []

        # Load SDSS images
        sdss_filters = "u"
        if len(sdss_filters) > 0:
            SDSS = SDSSImage(
                ra=center_ra,
                dec=center_dec,
                filters="".join([flt for flt in sdss_filters if flt in filters]),
                path="./sdss_cutout/",
            )
            SDSS.download()
            data_list_sdss, header_list_sdss = SDSS.load()
            data_list.extend(data_list_sdss)
            header_list.extend(header_list_sdss)
            wv_eff_sdss = np.array([SDSS.wv_eff_dict[flt] for flt in sdss_filters])
            wv_eff.extend(wv_eff_sdss)
            flts.extend(SDSS.filters)

        # Load PS1 images
        ps1_filters = "grizy"
        if len(ps1_filters) > 0:
            PS1 = PS1Image(
                ra=center_ra,
                dec=center_dec,
                filters="".join([flt for flt in ps1_filters if flt in filters]),
                path="./ps1_cutout/",
            )
            PS1.download()
            data_list_ps1, header_list_ps1 = PS1.load()
            data_list.extend(data_list_ps1)
            header_list.extend(header_list_ps1)
            wv_eff_ps1 = np.array([PS1.wv_eff_dict[flt] for flt in ps1_filters])
            wv_eff.extend(wv_eff_ps1)
            flts.extend(PS1.filters)

        # TODO: Load acquisition images (optional)

        # Order data_list and header_list by wavelength
        data_list = [data for _, data in sorted(zip(wv_eff, data_list))]
        header_list = [header for _, header in sorted(zip(wv_eff, header_list))]
        flts = [flt for _, flt in sorted(zip(wv_eff, flts))]
        wv_eff = sorted(wv_eff)

        # Spatial coordinates along the slit
        spat_slit = []
        # Counts along the slit
        counts_slit, counts_err_slit = [], []

        # Read the images and estimate the spatial profile
        for k, (data, header) in enumerate(zip(data_list, header_list)):
            # Load FITS image and WCS info
            wcs = WCS(header)

            # Get the position angle of the image cutout
            # Get the CD or PC matrix from WCS
            if wcs.wcs.has_cd():  # Check if CD matrix is present
                cd = wcs.wcs.cd
                pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600  # arcsec/pixel
                pa_img = jnp.arctan2(cd[0, 1], cd[1, 1]) + np.pi  # Arctangent of the y-x ratio
            else:  # Otherwise, use PC matrix with CDELT
                cd = wcs.wcs.pc * wcs.wcs.cdelt
                pixel_scale = wcs.wcs.cdelt[0] * 3600  # arcsec/pixel
                pa_img = jnp.arctan2(cd[0, 1], cd[1, 1])  # Arctangent of the y-x ratio

            # Define the rectangle size in pixels or arcminutes (angular size)
            slit_len_pix = slit_len / pixel_scale  # Slit length in pixels
            slit_wid_pix = slit_wid / pixel_scale  # Slit width in pixels

            # Convert RA, Dec to pixel coordinates
            coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
            center_x, center_y = wcs.world_to_pixel(coord)

            # Create a slit with the specified size/position angle
            slit_y_0, slit_x_0 = np.meshgrid(
                np.arange(-np.ceil(slit_wid_pix / 2), np.ceil(slit_wid_pix / 2)) + 0.5,
                np.arange(-np.ceil(slit_len_pix / 2), np.ceil(slit_len_pix / 2)) + 0.5,
            )

            # Obtain the pixel coordinates of the slit
            pa_slit = pa_img + np.deg2rad(position_angle) + np.pi / 2  # w.r.t. the west
            rot_matrix = np.array([[np.cos(pa_slit), -np.sin(pa_slit)], [np.sin(pa_slit), np.cos(pa_slit)]])
            slit_x_rot, slit_y_rot = np.dot(rot_matrix, np.array([slit_x_0.flatten(), slit_y_0.flatten()]))
            slit_x_rot += center_x
            slit_y_rot += center_y
            # Resample the image along the slit
            spat_slit.append((np.arange(-np.ceil(slit_len_pix / 2), np.ceil(slit_len_pix / 2)) + 0.5) * pixel_scale)
            data_slit = Interp2D_Grid(
                points=(np.arange(data.shape[1]) + 1, np.arange(data.shape[0]) + 1), values=data.T
            )(np.stack([slit_x_rot, slit_y_rot], axis=-1)).reshape(slit_x_0.shape)
            # Estimate the counts: average along the slit width
            counts_slit.append(
                np.array(
                    [
                        bound_mean(
                            (np.arange(-np.ceil(slit_wid_pix / 2), np.ceil(slit_wid_pix / 2)) + 0.5) * pixel_scale,
                            d,
                            x_bound=(-slit_wid / 2, slit_wid / 2),
                        )
                        for d in data_slit
                    ]
                )
            )
            # Estimate the error: standard deviation of the residuals (count at each pixel - average count)
            err = np.nanstd(data_slit - counts_slit[-1][:, None], axis=1)
            # Smooth the error: convolution with a boxcar filter
            err = (np.convolve(err**2, np.ones(3) / 3, mode="same")) ** 0.5
            counts_err_slit.append(err)

        return cls(
            flts=flts,
            wv_eff=wv_eff,
            spat_slit=spat_slit,
            counts_slit=counts_slit,
            counts_err_slit=counts_err_slit,
            spec_model=spec_model,
            slit_len=slit_len,
            pixel_scale=pixel_scale,
        )

    def model_host_profile_prior(self, **kwargs) -> Callable[[Array], Array]:
        """
        Model the host galaxy spatial profile using Gaussian Process regression.
        """
        # No prior photometric data
        if len(self.flts) == 0:
            host_prior = lambda _: jnp.float64(1 / self.host_wid)  # constantv
        # Single band
        elif len(self.flts) == 1:
            params = dict(
                log_amp=np.float64(-3),
                log_scale=np.float64(-0.5),
                mean=np.float64(1 / self.host_wid),
            )
            params_limit = dict(log_scale=np.log10([0.8 / 2.355, 1.5 / 2.355]))
            gp_host_prior = GP(
                X=self.X[:, 0][:, None],  # Spatial coordinate only
                y=self.prof,
                yerr=self.prof_err,
                # params=params,
                params_init=params,
                params_limit=params_limit,
                optimization=True,
            )
            host_prior = jax.jit(lambda x: gp_host_prior.gp.predict(y=self.prof, X_test=x[:, 0][:, None]))
        # Multiple bands
        else:
            params = dict(
                log_amp=np.float64(-2),
                log_scale=np.log10([1 / 2.355, 1e4]),
                mean=np.float64(1 / self.host_wid),
            )
            params_limit = dict(
                log_scale=np.log10([[0.8 / 2.355, 1e2], [1.5 / 2.355, 1e5]]),
            )
            gp_host_prior = GP(
                X=self.X,
                y=self.prof,
                yerr=self.prof_err,
                # params=params,
                params_init=params,
                params_limit=params_limit,
                optimization=True,
            )
            host_prior = jax.jit(lambda x: gp_host_prior.gp.predict(y=self.prof, X_test=x))

        # Whether to plot the host profile
        show = kwargs.get("show", False)
        # Whether to save the plot
        save = kwargs.get("save", None)
        _, ax = plt.subplots(
            len(self.flts), 1, figsize=(6, 2 * len(self.flts)), sharex=True, sharey=True, constrained_layout=True
        )
        ax = np.atleast_1d(ax)
        cmap = plt.cm.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=0, vmax=len(self.flts) - 1)
        for k in range(len(self.flts)):
            ax[k].plot(self.spat_slit[k], self.prof_slit[k], label=f"{self.flts[k]}", color=cmap(norm(k)))
            ax[k].plot(
                self.spat_slit[k],
                host_prior(jnp.stack([self.spat_slit[k], self.wv_slit[k]], axis=-1)),
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
                0.05, 0.8, f"{self.flts[k]}: {self.wv_eff[k]:.0f} Ang", color=cmap(norm(k)), transform=ax[k].transAxes
            )
        ax[-1].set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
        if save is not None:
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        return host_prior


def bound_sum(x: Array, y: Array, x_bound: tuple[float, float] = None) -> jnp.float64:
    """
    Compute the mean values in a bounded region.
    """
    bin_size = jnp.append(x[1] - x[0], jnp.diff(x))
    if x_bound is None:
        x_bound = (x[0] - bin_size[0] / 2, x[-1] + bin_size[-1] / 2)
    if x_bound[1] <= x_bound[0]:
        return jnp.float64(0)
    # sum up all pixels that are fully contained in the region
    idx_center = (x > x_bound[0] + bin_size[0] / 2) & (x < x_bound[1] - bin_size[-1] / 2)
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


def bound_mean(x: Array, y: Array, x_bound: tuple[float, float] = None) -> jnp.float64:
    """
    Compute the sum in a bounded region.
    """
    return bound_sum(x, y, x_bound) / (x_bound[1] - x_bound[0])
