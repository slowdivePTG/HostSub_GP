# hostsub_gp/spectrum_proc.py

__all__ = ["SpecData"]

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from functools import partial
from jax._src.typing import ArrayLike, Array

import glob
from astropy.io import fits
from pypeit import spec2dobj, specobjs
import json

from .interp import Interp1D_Grid, Interp2D_Grid, Interp2D_RBF
from .spectrum_model import SpecModel
from .host_model import HostProfile
from ._plt import plt


class SpecData:
    """
    A class to load (from different spectrograph) and preprocess (rectification & alignment) the 2D spectrum.
    """

    def __init__(
        self,
        *,
        pixel_scale: float,
        center_ra: float,
        center_dec: float,
        slit_wid: float,
        position_angle: float,
        spat_resln: float,
        spec_resln: float,
        spat_rect: ArrayLike,
        spec_rect: ArrayLike,
        flux_rect: ArrayLike = None,
        flux_ivar_rect: ArrayLike = None,
        dist: ArrayLike = None,
        flux: ArrayLike = None,
        flux_ivar: ArrayLike = None,
        waveimg: ArrayLike = None,
        spat_padding: float = 1.0,
    ):
        self.spat_rect = jnp.asarray(spat_rect)
        self.spec_rect = jnp.asarray(spec_rect)
        self.pixel_scale = pixel_scale
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.slit_wid = slit_wid
        self.position_angle = position_angle
        self.spat_resln = spat_resln
        self.spec_resln = spec_resln

        if (flux_rect is not None) and (flux_ivar_rect is not None):
            self.flux_rect = jnp.asarray(flux_rect)
            self.flux_ivar_rect = jnp.asarray(flux_ivar_rect)
        else:
            if dist is None:
                raise ValueError("No distance array provided.")
            if (flux is None) or (flux_ivar is None) or (waveimg is None):
                raise ValueError("No flux, ivar, or wavelength solution provided.")

            offset = self.get_offset(points=jnp.stack([dist, waveimg], axis=-1), flux=flux, show=True)

            self._points = jnp.stack([dist - offset, waveimg], axis=-1)

            # valid points - not NaN/inf
            valid_flag = jnp.isfinite(self._points).all(axis=-1)
            # valid spatial range - within the slit + padding (in case the trace is not perfectly centered)
            dist_flag = (dist >= spat_rect.min() - spat_padding) & (dist <= spat_rect.max() + spat_padding)
            flag = jnp.array(valid_flag & dist_flag, dtype=bool)

            self.flux_rect, self.flux_ivar_rect = self.rectify(
                points=self._points,
                f_values=(flux, flux_ivar, flag),
                spat_rect=self.spat_rect,
                spec_rect=self.spec_rect,
            )
            self.to_caches()

    @classmethod
    def from_pypeit(
        cls,
        sci_id: str,
        obj_id: str = None,
        std_id: str = None,
        sci_dir: str = "./",
        spat_resln: float = None,
        slit_len: float = 20.0,
    ):
        """
        Load 2D spectra from PypeIt output files.

        Parameters
        ----------
        sci_id : str
            The ID of the science object.
        obj_id : str, optional (default: None)
            The object ID in the science frame.
        std_id : str, optional (default: None)
            The ID of the standard star.
        sci_dir : str, optional (default: "./")
            The directory of the science frame.
        spat_resln : float, optional (default: None)
            The spatial resolution (seeing) of the science frame.
        slit_len : float, optional (default: 20.0, in arcsec)
            The length of the slit in the spatial direction.
        """

        sci_file = glob.glob(pathname=sci_dir + f"spec2d*{sci_id}*fits")[0]
        pypeit_header = fits.getheader(sci_file)

        if pypeit_header["PYP_SPEC"] in ["keck_lris_blue", "keck_lris_red", "keck_lris_red_mark4"]:
            position_angle = pypeit_header["ROTPOSN"] + 90
            ra, dec = pypeit_header["RA"], pypeit_header["DEC"]
            binning = int(
                pypeit_header["BINNING"].split(",")[-1]
            )  # TODO: handle different binning in spatial and spectral directions
            pixel_scale = 0.135 * binning
            det = "DET02"
            slit_wid = float(pypeit_header["SLITNAME"].split("_")[-1])
            spec_resln = 7.5  # TODO: get the spectral resolution from the header
        else:
            raise NotImplementedError("Only LRIS is supported")

        if obj_id is not None:
            # If the object ID in the science frame is provided (i.e., object successfully found), use the object trace
            trace_file = glob.glob(pathname=sci_dir + f"spec1d*{sci_id}*fits")[0]
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            name_idx = trace_objs.name_indices(obj_id)
            if all(~name_idx):
                raise ValueError(f"Object {obj_id} not found in the trace file.")
            trace_obj = trace_obj[name_idx]
        elif std_id is not None:
            # If the object ID is not provided, use the standard star trace
            trace_file = glob.glob(pathname=sci_dir + f"spec1d*{std_id}*fits")[0]
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            # Find the SpecObj with the highest signal-to-noise ratio (S2N) in the SpecObjs
            argmax_snr = np.argmax([obj["S2N"] for obj in trace_objs])
            trace_obj = trace_objs[argmax_snr]

            if spat_resln is None:
                spat_resln = trace_obj["FWHM"] * pixel_scale
        else:
            raise ValueError("No spec1d file provided for identifying the trace.")

        trace_spat_pix = trace_obj["TRACE_SPAT"]  # spatial pixel of the trace

        sci2d = spec2dobj.Spec2DObj.from_file(sci_file, detname=det)

        flux = np.array(sci2d.sciimg.T)
        ivar = np.array(sci2d.ivarraw.T)
        waveimg = np.array(sci2d.waveimg.T)
        bpmmask = np.array(sci2d.bpmmask.mask.T)
        tilts = np.array(sci2d.tilts.T)

        flux[bpmmask != 0] = np.nan
        ivar[bpmmask != 0] = 0

        # Estimate the distance from the standard trace
        # For each pixel in the 2D spectrum with a certain wavelength,
        # find the corresponding spectral pixel within the trace at the same wavelength
        trace_spec = Interp2D_Grid(points=(np.arange(waveimg.shape[0]), np.arange(waveimg.shape[1])), values=waveimg)(
            np.stack([trace_spat_pix, np.arange(waveimg.shape[1])], axis=-1)
        )
        trace_spec_pix = np.where(
            tilts != 0, Interp1D_Grid(points=trace_spec, values=np.arange(len(trace_spec)))(waveimg), np.nan
        )

        # indices of the spatial and spectral pixels
        spat_pix = jnp.tile(jnp.arange(waveimg.shape[0]), (waveimg.shape[1], 1)).T
        spec_pix = jnp.tile(jnp.arange(waveimg.shape[1]), (waveimg.shape[0], 1))

        dist_spat_pix = spat_pix - trace_spat_pix
        dist_spec_pix = spec_pix - trace_spec_pix
        dist_pix = np.sqrt(dist_spat_pix**2 + dist_spec_pix**2) * np.where(dist_spat_pix > 0, 1, -1)

        slit_radius_pix = int(np.ceil(slit_len / pixel_scale / 2))
        spat_range = (-slit_radius_pix, slit_radius_pix + 1)

        return cls(
            pixel_scale=pixel_scale,
            center_ra=ra,
            center_dec=dec,
            slit_wid=slit_wid,
            position_angle=position_angle,
            spat_resln=spat_resln,
            spec_resln=spec_resln,
            flux=flux,
            flux_ivar=ivar,
            waveimg=waveimg,
            dist=dist_pix * pixel_scale,
            spat_rect=jnp.arange(*spat_range) * pixel_scale,
            spec_rect=trace_spec,
        )

    @classmethod
    def from_caches(cls, cache_path: str = ".cache.json"):
        """
        Load 2D spectra from cache files.

        Parameters
        ----------
        cache_path : str, optional (default: ".cache.json")
            The path to the cache file.
        """
        if hasattr(cls, "_cache_path"):
            cache_file = cls._cache_path
        else:
            cache_file = cache_path
        try:
            with open(cache_file, "r", encoding="UTF-8") as f:
                public_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache file {cache_file} not found.")
        return cls(**public_data)

    def to_caches(self, cache_path: str = ".cache.json"):
        """
        Save the 2D spectra to cache files.

        Parameters
        ----------
        cache_path : str, optional (default: ".cache.json")
            The path to the cache file.
        """

        public_data = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

        for key in ["spat_rect", "spec_rect", "flux_rect", "flux_ivar_rect"]:
            # Convert the JAX array (if any) to numpy array
            public_data[key] = np.array(public_data[key]).tolist()

        with open(cache_path, "w", encoding="UTF-8") as f:
            json.dump(public_data, f, indent=4)
        self._cache_path = cache_path

    def to_SpecModel(
        self,
        mask_wid: float = 1.5,
        sky_wid: tuple = (5.0, 5.0),
        spec_range: tuple[float, float] | list[float] = None,
        show: bool = False,
    ) -> SpecModel:
        """
        Convert the 2D spectra to a SpecModel object.

        Parameters
        ----------
        mask_wid : float, optional (default: 1.5, in spat_resln)
            The width of the mask (science object) region.
        sky_wid : tuple, optional (default: (5.0, 5.0), in spat_resln)
            The width of the sky region.
        spec_range : tuple or list, optional (default: None)
            The range of the spectral pixels to include.
        """
        if spec_range is None:
            spec_mask = jnp.ones_like(self.spec_rect, dtype=bool)
        else:
            spec_mask = (self.spec_rect >= spec_range[0]) & (self.spec_rect <= spec_range[1])

        return SpecModel(
            dat=self.flux_rect[:, spec_mask],
            dat_err=self.flux_ivar_rect[:, spec_mask] ** -0.5,
            spat=self.spat_rect,
            spec=self.spec_rect[spec_mask],
            pixel_scale=self.pixel_scale,
            center_ra=self.center_ra,
            center_dec=self.center_dec,
            slit_wid=self.slit_wid,
            position_angle=self.position_angle,
            spat_resln=self.spat_resln,
            spec_resln=self.spec_resln,
            mask_wid=mask_wid,
            sky_wid=sky_wid,
            show=show,
        )

    def rectify(
        self,
        points: ArrayLike,
        f_values: tuple[ArrayLike, ArrayLike, ArrayLike],
        spat_rect: ArrayLike,
        spec_rect: ArrayLike,
        batch_size: int = 8,
        padding_size: int = 1,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Rectify the 2D spectrum onto a grid.

        Parameters
        ----------
        points : ArrayLike
            The spatial and spectral pixel coordinates.
        f_values : tuple[ArrayLike, ArrayLike, ArrayLike]
            The flux, ivar, and flag values.
        spat_rect : Array
            The rectified spatial coordinates.
        spec_rect : Array
            The rectified spectral coordinates.
        batch_size : int, optional (default: 8, in pixels)
            The batch size for interpolation.
        padding_size : int, optional (default: 1, in pixels)
            The padding size for interpolation.
        """

        flux, ivar, flag = f_values

        spec_batch_idx = jnp.array_split(jnp.arange(len(spec_rect)), len(spec_rect) // batch_size)
        spec_pix_rect, spat_pix_rect = jnp.meshgrid(spec_rect, spat_rect)

        flux_rect = np.zeros((len(spat_rect), len(spec_rect)))
        flux_ivar_rect = np.zeros((len(spat_rect), len(spec_rect)))

        # Interpolate the flux row by row
        for idx_list in spec_batch_idx:
            # The range of the spectrum to interpolate
            spec_min = max(0, idx_list[0] - padding_size)
            spec_max = min(len(spec_rect), idx_list[-1] + padding_size + 1)

            flag_ = flag[:, spec_min:spec_max]
            points_ = points[:, spec_min:spec_max][flag_]
            flux_ = flux[:, spec_min:spec_max][flag_]
            ivar_ = ivar[:, spec_min:spec_max][flag_]
            query_points_ = jnp.stack([spat_pix_rect[:, idx_list].ravel(), spec_pix_rect[:, idx_list].ravel()], axis=-1)

            # Interpolate the flux with RBF
            rbf = Interp2D_RBF(
                kernel="gaussian", epsilon=1.0, n_neighbors=8, scales=(self.spat_resln / 2.355, self.spec_resln / 2.355)
            )
            rbf.fit(points=points_, values=flux_)
            flux_rect[:, idx_list] = rbf.predict(query_points=query_points_).reshape(flux_rect[:, idx_list].shape)
            rbf_ivar = Interp2D_RBF(
                kernel="gaussian", epsilon=1.0, n_neighbors=8, scales=(self.spat_resln / 2.355, self.spec_resln / 2.355)
            )
            rbf_ivar.fit(points=points_, values=ivar_)
            flux_ivar_rect[:, idx_list] = rbf_ivar.predict(query_points=query_points_).reshape(
                flux_ivar_rect[:, idx_list].shape
            )

            print(
                f"Interpolating {points_[:, 1].min():.2f} - {points_[:, 1].max():.2f} Ang ({idx_list[0]} - {idx_list[-1]})"
            )

        return flux_rect, flux_ivar_rect

    def get_offset(self, points: ArrayLike, flux: ArrayLike, show: bool = True) -> float:
        """
        Center the trace of the science object.
        """
        from scipy.stats import binned_statistic

        # spec_model = self.to_SpecModel(spec_range=spec_range, show=False)
        # spec_model.model_host_prior(flts="griz", show=False)

        host_prior = HostProfile(
            flts="griz",
            center_ra=self.center_ra,
            center_dec=self.center_dec,
            slit_wid=self.slit_wid,
            slit_len=self.spat_rect.max() - self.spat_rect.min(),
            position_angle=self.position_angle,
        ).model_host_profile_prior(show=False)
        plt.show()

        flag = (
            jnp.isfinite(points[:, :, 0])
            & jnp.isfinite(flux)
            & (flux > jnp.nanpercentile(flux, 25))
        )

        obs, _, _ = binned_statistic(
            points[:, :, 0][flag],
            flux[flag],
            statistic="mean",
            bins=len(self.spat_rect),
            range=(self.spat_rect[0] - self.pixel_scale / 2, self.spat_rect[-1] + self.pixel_scale / 2),
        )

        offset = np.arange(-1, 1 + self.pixel_scale / 5, self.pixel_scale / 5)

        def corr_coef(offset):
            dist = self.spat_rect - offset
            wv = jnp.mean(points[:, :, 1][flag] * flux[flag]) / jnp.mean(flux[flag]) * jnp.ones_like(self.spat_rect)
            prior = host_prior(jnp.stack([dist, wv], axis=-1))
            return jnp.corrcoef(prior, obs)[0, 1]

        ccf = jax.vmap(corr_coef)(offset)

        offset_0 = offset[np.argmax(ccf)]

        if show:
            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(offset, ccf)
            ax[0].set_xlabel(r"$\mathrm{SCI\ -\ STD offset\ [arcsec]}$")
            ax[0].set_ylabel(r"$\mathrm{Correlation Coefficient}$")
            ax[1].scatter(
                self.spat_rect - offset_0,
                obs / obs.max(),
                label="obs",
            )

            profile_prior = host_prior(
                jnp.stack(
                    [
                        self.spat_rect - offset_0,
                        jnp.mean(points[:, :, 1][flag] * flux[flag])
                        / jnp.mean(flux[flag])
                        * jnp.ones_like(self.spat_rect),
                    ],
                    axis=-1,
                )
            )
            ax[1].scatter(
                self.spat_rect - offset_0,
                profile_prior / profile_prior.max(),
                label="prior",
            )
            ax[1].set_xlabel(r"$\mathrm{Spat\ [pixel]}$")
            ax[1].set_ylabel(r"$\mathrm{Normalized\ Counts}$")
            ax[1].legend()
        plt.show()

        return offset_0
