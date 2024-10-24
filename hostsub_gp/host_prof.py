# hostsub_gp/host_prof.py

__all__ = ["HostProfile"]

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ._plt_config import plt
from .gp import gp

from typing import Callable

class HostProfile:
    def __init__(
        self,
        imgs: list = [],
        flts: list = [],
        spec2d: any = None,
        center_ra: float = None, # deg
        center_dec: float = None, # deg
        slit_len: float = 10.0, # arcsec
        slit_wid: float = 1.0, # arcsec
        position_angle: float = None, # deg
        show: bool = False,
    ):
        assert len(imgs) == len(flts), "imgs and flts length mismatch"
        wv_eff_dict = dict(g_ps1=4810.16, r_ps1=6155.47, i_ps1=7503.03, z_ps1=8668.36)

        self.imgs = imgs
        self.flts = flts
        self.wv_eff = np.array([wv_eff_dict[flt] for flt in flts])

        if spec2d is not None:
            self.center_ra = spec2d.center_ra
            self.center_dec = spec2d.center_dec
            self.slit_len = spec2d.slit_len
            self.slit_wid = spec2d.slit_wid
            self.position_angle = spec2d.position_angle
        else:
            self.center_ra = center_ra
            self.center_dec = center_dec
            self.slit_len = slit_len
            self.slit_wid = slit_wid
            self.position_angle = position_angle

        if len(imgs) > 0:
            assert self.center_ra is not None, "center_ra is required for image data"
            assert self.center_dec is not None, "center_dec is required for image data"
            assert self.position_angle is not None, "position_angle is required for image data"

        self.counts_slit = []

        if show:
            # Plot the image and the slit
            _, ax = plt.subplots(
                len(flts), 2, figsize=(9, 4 * len(flts)), constrained_layout=True, sharex="col"
            )  # plt.subplots(subplot_kw={'projection': wcs})
            ax[-1, 0].set_xlabel("X (pixels)")
            ax[-1, 1].set_xlabel("Spatial coordinate (arcsec)")

        for k, (img, flt) in enumerate(zip(self.imgs, self.flts)):

            # Load FITS image and WCS info
            hdulist = fits.open(img)
            data = hdulist[0].data
            header = hdulist[0].header
            wcs = WCS(header)

            # Define the rectangle size in pixels or arcminutes (angular size)
            pixel_scale = header["CDELT1"] * 3600  # Pixel scale in arcsec/pixel
            slit_len_pix = self.slit_len / pixel_scale  # Slit length in pixels
            slit_wid_pix = self.slit_wid / pixel_scale  # Slit width in pixels

            # Convert RA, Dec to pixel coordinates
            coord = SkyCoord(ra=self.center_ra * u.deg, dec=self.center_dec * u.deg, frame="icrs")
            center_x, center_y = wcs.world_to_pixel(coord)

            # Rotate the image
            data_fill = data.copy()
            data_fill[np.isnan(data_fill)] = np.nanmedian(data_fill)
            data_rot = rotate(data_fill, self.position_angle, reshape=False, order=5, cval=np.nan)

            # Obtain the new pixel coordinates of the target
            theta = np.deg2rad(self.position_angle)
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
            center_x_rot, center_y_rot = (
                np.dot(rot_matrix, np.array([center_x, center_y]) - data.shape[0] / 2) + data.shape[0] / 2
            )

            # Extract the host galaxy spatial profile
            data_slit = data_rot[
                int(np.floor(center_y_rot - slit_len_pix / 2)) - 1 : int(np.ceil(center_y_rot + slit_len_pix / 2)) - 1,
                int(np.floor(center_x_rot - slit_wid_pix / 2)) - 1 : int(np.ceil(center_x_rot + slit_wid_pix / 2)) - 1,
            ][::-1]
            counts_slit = np.mean(data_slit, axis=1)
            self.counts_slit.append(counts_slit)

            if show:
                # Plot the image and the slit
                ax[k, 0].imshow(
                    data_rot,
                    origin="lower",
                    cmap="gray",
                    vmin=np.nanpercentile(data_rot, 5),
                    vmax=np.nanpercentile(data_rot, 99),
                )
                rect = Rectangle(
                    (center_x_rot - slit_wid_pix / 2, center_y_rot - slit_len_pix / 2),
                    slit_wid_pix,
                    slit_len_pix,
                    edgecolor="red",
                    facecolor="none",
                )
                ax[k, 0].add_patch(rect)
                ax[k, 0].scatter(center_x_rot, center_y_rot, color="blue")  # Mark the center point
                ax[k, 0].set_xlim(center_x_rot - 40, center_x_rot + 40)
                ax[k, 0].set_ylim(center_y_rot - 40, center_y_rot + 40)

                # Plot the spatial profile of the galaxy
                ax[k, 1].plot(np.linspace(-self.slit_len / 2, self.slit_len / 2, counts_slit.size), counts_slit)

                ax[k, 0].set_ylabel("Y (pixels)")
                ax[k, 1].set_ylabel("Counts")
                ax[k, 0].set_title(f"{flt} Image")
                ax[k, 1].set_title(f"{flt} Profile")

                ax[k, 0].set_aspect("auto")

        if show:
            plt.show()

        if len(self.imgs) > 0:
            self.counts_slit = jnp.asarray(self.counts_slit)
            self.mean_prof_slit = jnp.mean(self.counts_slit, axis=0)
            self.prof_slit = self.counts_slit / jnp.sum(self.counts_slit, axis=1)[:, None]
            self.spat_slit = jnp.linspace(-self.slit_len / 2, self.slit_len / 2, self.prof_slit.shape[1])

    def model_host_profile_prior(self, **kwargs) -> Callable[[jax.Array], jax.Array]:
        """
        Model the host galaxy spatial profile using Gaussian Process regression.
        """
        # No prior photometric data
        if len(self.flts) == 0:
            host_prior = lambda x: jnp.float64(1 / self.slit_len) # constant
        # Single band
        elif len(self.flts) == 1:
            params = {
                "log_amp": jnp.float64(-3.0),
                "log_scale": jnp.float64(0.0),
                "jitter": jnp.float64(1e-6),
            }
            gp_host_prior = gp(
                X=self.spat_slit.T,
                y=self.mean_prof_slit,
                mean=1 / self.slit_len,
                params=params,
                params_init=params,
                **kwargs,
            )
            gp_pred = lambda x: gp_host_prior.gp.predict(y=self.mean_prof_slit, X_test=x)
            host_prior = gp_pred
        # Multiple bands
        else:
            spat_grid, wv_eff_grid = jnp.meshgrid(self.spat_slit, self.wv_eff)
            X = jnp.stack([spat_grid.ravel(), wv_eff_grid.ravel()], axis=-1)
            params = {
                "log_amp": jnp.float64(-3.0),
                "log_scale": jnp.asarray([0.0, 4.0]),
                "jitter": jnp.float64(1e-6),
            }
            gp_host_prior = gp(
                X=X,
                y=self.prof_slit.ravel(),
                mean=1 / self.slit_len,
                params=params,
                params_init=params,
                **kwargs,
            )
            gp_pred = lambda x: gp_host_prior.gp.predict(y=self.prof_slit.ravel(), X_test=x)
            host_prior = gp_pred
        return host_prior
