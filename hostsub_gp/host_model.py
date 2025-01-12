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
from jax._src.typing import Array


class HostProfile:
    def __init__(
        self,
        # flts: str | list = None,
        # cameras: str | list = None,
        spec2d: any = None,
        center_ra: float = None,  # deg
        center_dec: float = None,  # deg
        slit_len: float = None,  # arcsec
        slit_wid: float = 1.0,  # arcsec
        position_angle: float = None,  # deg
    ):

        if spec2d is not None:
            self.center_ra = spec2d.center_ra
            self.center_dec = spec2d.center_dec
            self.slit_len = spec2d.slit_len
            self.slit_wid = spec2d.slit_wid
            self.position_angle = spec2d.position_angle
        else:
            if center_ra is None or center_dec is None:
                raise ValueError("Coordinates are required")
            if position_angle is None:
                raise ValueError("Position angle is required")
            self.center_ra = center_ra
            self.center_dec = center_dec
            self.slit_len = slit_len
            self.slit_wid = slit_wid
            self.position_angle = position_angle

        # self.flts = flts
        # if cameras is None:
        #     self.cameras = ["ps1"] * len(flts)
        # elif isinstance(cameras, str):
        #     self.cameras = [cameras] * len(flts)
        # else:
        #     if len(cameras) != len(flts):
        #         raise ValueError("cameras and flts length mismatch")
        #     self.cameras = cameras
        # if not all(cam in ["ps1", "sdss"] for cam in self.cameras):  # TODO: Add support for other cameras
        #     raise NotImplementedError("Only PS1 and SDSS images are supported")

        data_list, header_list = [], []
        wv_eff = []
        flts = []

        # Load SDSS images
        # sdss_filters = "".join([flt for cam, flt in zip(self.cameras, self.flts) if cam == "sdss"])
        sdss_filters = "u"
        if len(sdss_filters) > 0:
            SDSS = SDSSImage(ra=self.center_ra, dec=self.center_dec, filters=sdss_filters, path="./sdss_cutout/")
            SDSS.download()
            data_list_sdss, header_list_sdss = SDSS.load()
            data_list.extend(data_list_sdss)
            header_list.extend(header_list_sdss)
            wv_eff_sdss = np.array([SDSS.wv_eff_dict[flt] for flt in sdss_filters])
            wv_eff.extend(wv_eff_sdss)
            flts.extend(SDSS.filters)

        # Load PS1 images
        # ps1_filters = "".join([flt for cam, flt in zip(self.cameras, self.flts) if cam == "ps1"])
        ps1_filters = "grizy"
        if len(ps1_filters) > 0:
            PS1 = PS1Image(ra=self.center_ra, dec=self.center_dec, filters=ps1_filters, path="./ps1_cutout/")
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
        self.flts = [flt for _, flt in sorted(zip(wv_eff, flts))]
        self.wv_eff = sorted(wv_eff)

        counts_slit, counts_err_slit = [], []
        prof_slit, prof_err_slit = [], []
        spat_slit = []
        wv_slit = []

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
            slit_len_pix = self.slit_len / pixel_scale  # Slit length in pixels
            slit_wid_pix = self.slit_wid / pixel_scale  # Slit width in pixels

            # Convert RA, Dec to pixel coordinates
            coord = SkyCoord(ra=self.center_ra * u.deg, dec=self.center_dec * u.deg, frame="icrs")
            center_x, center_y = wcs.world_to_pixel(coord)

            # Create a slit with the specified size/position angle
            slit_y_0, slit_x_0 = np.meshgrid(
                np.arange(-np.ceil(slit_wid_pix / 2), np.ceil(slit_wid_pix / 2)) + 0.5,
                np.arange(-np.ceil(slit_len_pix / 2), np.ceil(slit_len_pix / 2)) + 0.5,
            )

            # Obtain the pixel coordinates of the slit
            pa_slit = pa_img + np.deg2rad(self.position_angle) + np.pi / 2  # w.r.t. the west
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
                            x_bound=(-self.slit_wid / 2, self.slit_wid / 2),
                        )
                        for d in data_slit
                    ]
                )
            )
            # Estimate the error: standard deviation of the residuals (count at each pixel - average count)
            err = np.nanstd(data_slit - counts_slit[-1][:, None], axis=1)
            # Smooth the error: convolution with a boxcar filter
            err = np.convolve(err, np.ones(5) / 5, mode="same")
            counts_err_slit.append(err)

            wv_slit.append(np.ones_like(counts_slit[-1]) * self.wv_eff[k])
            if spec2d is not None:
                host_left = (-spec2d.slit_len / 2, -spec2d.mask_wid / 2 + spec2d.mask_offset)
                host_right = (spec2d.mask_wid / 2 + spec2d.mask_offset, spec2d.slit_len / 2)
                sky_left = (-spec2d.slit_len / 2, -spec2d.sky_wid / 2)
                sky_right = (spec2d.sky_wid / 2, spec2d.slit_len / 2)
                xi = counts_slit[-1]
                xi_err = counts_err_slit[-1]
                xi_sky_mean = (
                    bound_mean(spat_slit[-1], xi, x_bound=sky_left) + bound_mean(spat_slit[-1], xi, x_bound=sky_right)
                ) / 2
                xi_host_mean = (
                    bound_mean(spat_slit[-1], xi, x_bound=host_left) + bound_mean(spat_slit[-1], xi, x_bound=host_right)
                ) / 2
                prof_slit.append(
                    (xi - xi_sky_mean)
                    / (xi_host_mean - xi_sky_mean)
                    / (spec2d.slit_len - spec2d.mask_wid)
                    * spec2d.pixel_scale
                )
                prof_err_slit.append(
                    xi_err / (xi_host_mean - xi_sky_mean) / (spec2d.slit_len - spec2d.mask_wid) * spec2d.pixel_scale
                )

            else:  # No mask
                xi = counts_slit[-1] / np.sum(counts_slit[-1]) / pixel_scale
                xi_err = counts_err_slit[-1] / np.sum(counts_slit[-1]) / pixel_scale
                prof_slit.append(xi)
                prof_err_slit.append(xi_err)

        self.prof_slit = prof_slit
        self.prof_err_slit = prof_err_slit
        self.spat_slit = spat_slit
        self.wv_slit = wv_slit
        self.prof = jnp.concatenate(prof_slit)
        self.prof_err = jnp.concatenate(prof_err_slit)
        self.X = jnp.stack([jnp.concatenate(spat_slit), jnp.concatenate(wv_slit)], axis=-1)

    def model_host_profile_prior(self, **kwargs) -> Callable[[jax.Array], jax.Array]:
        """
        Model the host galaxy spatial profile using Gaussian Process regression.
        """
        # No prior photometric data
        if len(self.flts) == 0:
            host_prior = lambda _: jnp.float64(1 / self.slit_len)  # constant
        # Single band
        elif len(self.flts) == 1:
            params = dict(
                log_amp=jnp.float64(-3),
                log_scale=jnp.float64(0.5),
                # log_jitter=jnp.float64(-6),
                mean=jnp.float64(1 / self.slit_len),
            )
            params_limit = dict(log_scale=np.log10([1e-1, 10]))
            gp_host_prior = GP(
                X=self.X[:, 0][:, None],  # Spatial coordinate only
                y=self.prof,
                yerr=self.prof_err,
                params=params,
                params_init=params,
                params_limit=params_limit,
                optimization=True,
            )
            host_prior = jax.jit(lambda x: gp_host_prior.gp.predict(y=self.prof, X_test=x[:, 0][:, None]))
        # Multiple bands
        else:
            params = dict(
                log_amp=jnp.float64(-2),
                log_scale=jnp.asarray([0.1, 5], dtype=jnp.float64),
                mean=jnp.float64(1 / self.slit_len),
            )
            params_limit = dict(
                log_scale=np.log10([[1e-1, 1e3], [1e1, 1e7]]),
            )
            gp_host_prior = GP(
                X=self.X,
                y=self.prof,
                yerr=self.prof_err,
                params=params,
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
        ax[-1].set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save, bbox_inches="tight")

        return host_prior


def bound_mean(x: Array, y: Array, x_bound: tuple[float, float] = None) -> jnp.float64:
    """
    Compute the mean values in a bounded region.
    """
    bin_size = jnp.append(x[1] - x[0], jnp.diff(x))
    if x_bound is None:
        x_bound = (x[0] - bin_size[0] / 2, x[-1] + bin_size[-1] / 2)
    # sum up all pixels that are fully contained in the region
    idx_center = (x > x_bound[0] + bin_size[0] / 2) & (x < x_bound[1] - bin_size[-1] / 2)
    sum_center = jnp.sum(y[idx_center] * bin_size[idx_center])
    # print("Pixels fully contained in the region:")
    # print(x[idx_center])

    # leftmost pixel that is partially contained in the region (if any)
    idx_left = jnp.where(x >= x_bound[0] - bin_size[0] / 2)[0]
    if idx_left.size > 0:
        y_left = y[idx_left[0]]
        frac_left = x[idx_left[0]] - (x_bound[0] - bin_size[0] / 2)
        sum_left = y_left * frac_left
    else:
        raise ValueError("Invalid left bound")
    # print("Pixel on the left edge:")
    # print(x[idx_left[0]])
    # print("Coverage:")
    # print(frac_left)

    # rightmost pixel that is partially contained in the region (if any)
    idx_right = jnp.where(x <= x_bound[1] + bin_size[-1] / 2)[-1]
    if idx_right.size > 0:
        y_right = y[idx_right[-1]]
        frac_right = (x_bound[1] + bin_size[-1] / 2) - x[idx_right[-1]]
        sum_right = y_right * frac_right
    else:
        raise ValueError("Invalid right bound")
    # print("Pixel on the right edge:")
    # print(x[idx_right[-1]])
    # print("Coverage:")
    # print(frac_right)
    return (sum_center + sum_left + sum_right) / (x_bound[1] - x_bound[0])
