# hostsub_gp/host_prof.py

__all__ = ["HostProfile"]

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ._plt_config import plt
from .gp import _gp
from ._load import load_image, wv_eff_dict

from typing import Callable


class HostProfile:
    def __init__(
        self,
        imgs: list = [],
        flts: list = [],
        cameras: str | list = None,
        spec2d: any = None,
        center_ra: float = None,  # deg
        center_dec: float = None,  # deg
        slit_len: float = 10.0,  # arcsec
        slit_wid: float = 1.0,  # arcsec
        position_angle: float = None,  # deg
        show: bool = False,
    ):
        assert len(imgs) == len(flts), "imgs and flts length mismatch"

        self.imgs = imgs
        self.flts = flts
        if cameras is None:
            self.cameras = ["ps1"] * len(flts)
        elif isinstance(cameras, str):
            self.cameras = [cameras] * len(flts)
        else:
            assert len(cameras) == len(flts), "cameras and flts length mismatch"
            self.cameras = cameras
        self.wv_eff = np.array([wv_eff_dict[cam][flt] for cam, flt in zip(self.cameras, self.flts)])

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

        counts_slit, prof_slit = [], []
        spat_slit = []
        wv_slit = []

        if show:
            # Plot the image and the slit
            _, ax = plt.subplots(
                len(flts), 2, figsize=(9, 4 * len(flts)), constrained_layout=True, sharex="col"
            )  # plt.subplots(subplot_kw={'projection': wcs})
            ax[-1, 0].set_xlabel("X (pixels)")
            ax[-1, 1].set_xlabel("Spatial coordinate (arcsec)")

        for k, (img, flt, cam) in enumerate(zip(self.imgs, self.flts, self.cameras)):
            # Load FITS image and WCS info
            data, header = load_image(img, camera=cam)
            wcs = WCS(header)
            pixel_scale = header["CDELT1"] * 3600

            # Define the rectangle size in pixels or arcminutes (angular size)
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
            ]
            counts_slit.append(np.mean(data_slit, axis=1))  # Average along the slit width
            spat_slit.append(np.linspace(-self.slit_len / 2, self.slit_len / 2, counts_slit[-1].size))
            wv_slit.append(np.ones_like(counts_slit[-1]) * self.wv_eff[k])
            if spec2d is not None:
                # Mask the SN aperture
                mask = np.abs(spat_slit[-1]) < spec2d.spat_resln * spec2d.mask_wid
                # Subtract the sky background
                sky = (spat_slit[-1] < spec2d.spat_resln * spec2d.sky_wid[-1]) | (
                    spat_slit[-1] > spec2d.spat_resln * spec2d.sky_wid[0]
                )
                xi = counts_slit[-1] / np.sum(counts_slit[-1][~mask])
                mask_len = 2 * spec2d.spat_resln * spec2d.mask_wid
                sky_len = spec2d.spat_resln * (spec2d.sky_wid[-1] + spec2d.sky_wid[0])
                prof_slit.append(
                    (xi - xi[sky].mean())
                    / (1 - xi[sky].sum() * (slit_len - mask_len) / (slit_len - sky_len))
                    * (spec2d.pixel_scale / pixel_scale)
                )
            else:  # No mask
                xi = counts_slit[-1] / np.sum(counts_slit[-1])
                prof_slit.append(xi)

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
                ax[k, 1].plot(np.linspace(-self.slit_len / 2, self.slit_len / 2, counts_slit[-1].size), counts_slit[-1])

                ax[k, 0].set_ylabel("Y (pixels)")
                ax[k, 1].set_ylabel("Counts")
                ax[k, 0].set_title(f"{flt} Image")
                ax[k, 1].set_title(f"{flt} Profile")

                ax[k, 0].set_aspect("auto")

        if show:
            plt.show()

        self.prof_slit = prof_slit
        self.spat_slit = spat_slit
        self.wv_slit = wv_slit
        self.prof = jnp.concatenate(prof_slit)
        self.X = jnp.stack([jnp.concatenate(spat_slit), jnp.concatenate(wv_slit)], axis=-1)

    def model_host_profile_prior(self, show: bool = False, **kwargs) -> Callable[[jax.Array], jax.Array]:
        """
        Model the host galaxy spatial profile using Gaussian Process regression.
        """
        # No prior photometric data
        if len(self.flts) == 0:
            host_prior = lambda x: jnp.float64(1 / self.slit_len)  # constant
        # Single band
        elif len(self.flts) == 1:
            params = {
                "log_amp": jnp.float64(-3),
                "log_scale": jnp.float64(0),
                "log_jitter": jnp.float64(-6),
                "mean": jnp.float64(1 / self.slit_len),
            }
            gp_host_prior = _gp(
                X=self.X[:, 0][:, None],  # Spatial coordinate only
                y=self.prof,
                params=params,
                params_init=params,
                **kwargs,
            )
            gp_pred = lambda x: gp_host_prior.gp.predict(y=self.prof, X_test=x[:, 0][:, None])
            host_prior = jax.jit(gp_pred)
        # Multiple bands
        else:
            params = {
                "log_amp": jnp.float64(-3),
                "log_scale": jnp.asarray([0.5, 3], dtype=jnp.float64),
                "log_jitter": jnp.float64(-6),
                "mean": jnp.float64(1 / self.slit_len),
            }
            gp_host_prior = _gp(
                X=self.X,
                y=self.prof,
                params=params,
                params_init=params,
                **kwargs,
            )
            gp_pred = lambda x: gp_host_prior.gp.predict(y=self.prof, X_test=x)
            host_prior = jax.jit(gp_pred)

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            cmap = plt.cm.get_cmap("coolwarm")
            norm = plt.Normalize(vmin=0, vmax=len(self.flts) - 1)
            delta = (self.prof.max() - self.prof.min()) * 0.4
            for k in range(len(self.flts)):
                ax.plot(self.spat_slit[k], self.prof_slit[k] - k * delta, label=f"{self.flts[k]}", color=cmap(norm(k)))
                ax.plot(
                    self.spat_slit[k],
                    host_prior(jnp.stack([self.spat_slit[k], self.wv_slit[k]], axis=-1)) - k * delta,
                    "--",
                    color=cmap(norm(k)),
                )
            ax.set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
            ax.set_ylabel(r"$\mathrm{Profile + offset}$")

        return host_prior
