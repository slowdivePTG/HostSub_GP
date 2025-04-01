# hostsub_gp/spec_proc.py

__all__ = ["SpecData"]

import numpy as np
import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)

from jax._src.typing import ArrayLike, Array
from typing import Callable

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from .interp import Interp1D_Grid, Interp2D_Grid
from .spec_model import SpecModel
from .host_model import HostProfile
from ._utils import plt, msgs
from ._utils._plt import show_and_save


class SpecData:
    """
    A class to load (from different spectrograph) and preprocess (rectification & alignment) the 2D spectrum.
    """

    def __init__(
        self,
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
        waveimg: ArrayLike = None,
        flux: ArrayLike = None,
        flux_global_sky: ArrayLike = None,
        flux_ivar: ArrayLike = None,
        sky_offset: float = None,
        to_caches: bool = False,
        cache_path: str = None,
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
        self.sky_offset = sky_offset

        if (flux_rect is not None) and (flux_ivar_rect is not None):
            self.flux_rect = jnp.asarray(flux_rect)
            self.flux_ivar_rect = jnp.asarray(flux_ivar_rect)
        else:
            if dist is None:
                raise ValueError("No distance array provided.")
            if (flux is None) or (flux_ivar is None) or (waveimg is None):
                raise ValueError("No flux, ivar, or wavelength solution provided.")

            if self.sky_offset is None:
                host_prior = HostProfile.from_archival(
                    center_ra=self.center_ra,
                    center_dec=self.center_dec,
                    slit_len=self.spat_rect.max() - self.spat_rect.min() + self.pixel_scale,
                    slit_wid=self.slit_wid,
                    position_angle=self.position_angle,
                ).model_host_profile_prior()

                self.sky_offset = self._get_offset(
                    points=jnp.stack([dist, waveimg], axis=-1), flux=flux, show=True, mask_wid=2.0, host_prior=host_prior
                )
            self._points = jnp.stack([dist - self.sky_offset, waveimg], axis=-1)

            # Rectify the 2D spectrum
            if flux_global_sky is not None:
                self.flux_rect, self.flux_ivar_rect = self._rectify(
                    points=self._points, f_values=(flux - flux_global_sky, flux_ivar), interp_method="scipy"
                )
            else:
                self.flux_rect, self.flux_ivar_rect = self._rectify(
                    points=self._points, f_values=(flux, flux_ivar), interp_method="scipy"
                )

            # Save the 2D spectra to cache files
            self.cache_path = cache_path
            if to_caches:
                self.to_fits()

    @classmethod
    def from_pypeit(
        cls,
        sci_file: str,
        raw_dir: str = None,
        obj_id: str = None,
        std_file: str = None,
        ra: float = None,
        dec: float = None,
        spat_resln: float = None,
        spat_rect: ArrayLike = None,
        spec_rect: ArrayLike = None,
        **kwargs,
    ):
        """
        Load 2D spectra from PypeIt output files.

        Parameters
        ----------
        sci_file : str
            The filename of the science object.
        obj_id : str, optional (default: None)
            The object ID in the science frame.
        std_file : str, optional (default: None)
            The filename of the standard star.
        raw_dir : str, optional (default: None)
            The directory of the raw files (not needed for LRIS data).
        ra, dec : float, optional (default: None)
            The RA and DEC of the science object.
            If not provided, the RA and DEC in the header will be used.
        spat_resln : float, optional (default: None)
            The spatial resolution (seeing) of the science frame.
        spat_rect : ArrayLike, optional (default: None)
            The spatial coordinates of the rectified 2D spectrum.
        spec_rect : ArrayLike, optional (default: None)
            The spectral coordinates of the rectified 2D spectrum.
        """
        from pypeit import spec2dobj, specobjs

        if "spec1d" in sci_file:
            spec1d_file = sci_file
            spec2d_file = sci_file.replace("spec1d", "spec2d")
        elif "spec2d" in sci_file:
            spec2d_file = sci_file
            spec1d_file = sci_file.replace("spec2d", "spec1d")
        else:
            raise ValueError("Incorrect sci_file format.")

        msgs.info(f"Loading 2D spectrum for {spec2d_file}...")
        pypeit_header = fits.getheader(spec2d_file)

        # Access the header of the raw image
        history = pypeit_header["HISTORY"]
        for i in range(len(history)):
            if "Combining frames" in history[i]:
                raw_file = history[i + 1].strip('"')
                break

        if pypeit_header["PYP_SPEC"] in ["keck_lris_blue", "keck_lris_red", "keck_lris_red_mark4"]:
            position_angle = pypeit_header["ROTPOSN"] + 90
            if ra is None or dec is None:
                # RA and Dec in the header are already in the format of degrees
                ra, dec = pypeit_header["RA"], pypeit_header["DEC"]
            binning = int(pypeit_header["BINNING"].split(",")[1])  # in the spatial direction
            pixel_scale = 0.135 * binning
            if "red_mark4" in pypeit_header["PYP_SPEC"]:
                det = "DET01"
            else:
                det = "DET02"
            slit_wid = float(pypeit_header["SLITNAME"].split("_")[-1])

        elif pypeit_header["PYP_SPEC"] == "mmt_binospec":
            if raw_dir is None:
                raise ValueError("The raw file directory is needed for Binospec data.")
            raw_file = "/".join([raw_dir, raw_file])
            raw_header = fits.getheader(raw_file, ext=1)

            # PA: parallactic angle
            # ROT: instrument rotator angle (relative to the parallactic angle)
            position_angle = raw_header["PA"] - raw_header["ROT"]
            pixel_scale = 0.24
            det = "DET02"
            if ra is None or dec is None:
                # RA and DEC in the header are in the format of 'HH:MM:SS.SS' and 'DD:MM:SS.SS'
                ra_str, dec_str = raw_header["CATRA"].strip("'"), raw_header["CATDEC"].strip("'")
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra, dec = coord.ra.deg, coord.dec.deg

            slit_wid = float(raw_header["MASK"].split("Longslit")[-1])

        else:
            raise NotImplementedError("Only LRIS and Binospec are supported")

        # If the object ID in the science frame is provided (i.e., object successfully found), use the object trace
        if obj_id is not None:
            trace_file = spec1d_file
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            name_idx = trace_objs.name_indices(obj_id)
            if all(~name_idx):
                raise ValueError(f"Object {obj_id} not found in the trace file.")
            trace_obj = trace_objs[np.where(name_idx)[0][0]]

        # If the object ID is not provided, use the standard star trace
        elif std_file is not None:
            trace_file = std_file
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            # Find the SpecObj with the highest signal-to-noise ratio (S2N) in the SpecObjs
            argmax_snr = np.argmax([obj["S2N"] for obj in trace_objs])
            trace_obj = trace_objs[argmax_snr]
        else:
            raise ValueError("No spec1d file provided for identifying the trace.")

        if spat_resln is None:
            if std_file is None:
                raise ValueError(
                    "The spatial resolution needs to be either provided in the config file or estimated from the standard."
                )
            std_objs = specobjs.SpecObjs.from_fitsfile(std_file, det=det)
            argmax_snr = np.argmax([obj["S2N"] for obj in std_objs])
            spat_resln = std_objs[argmax_snr]["FWHM"] * pixel_scale

        trace_spat_pix = trace_obj["TRACE_SPAT"]  # spatial pixel of the trace

        sci2d = spec2dobj.Spec2DObj.from_file(spec2d_file, detname=det)

        # Spectral resolution: the FWHM of the arc lines
        spec_resln = sci2d["wavesol"]["mesured_fwhm"].value[0]  # A typo in PypeIt

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
        dist = np.sqrt(dist_spat_pix**2 + dist_spec_pix**2) * np.where(dist_spat_pix > 0, 1, -1) * pixel_scale

        # Preliminary sky line removal - reduce the noise introduced in the rectification
        wave_sky = np.nanmedian(np.where(waveimg > 0, waveimg, np.nan), axis=0)
        flux_sky = np.nanmedian(np.where(waveimg > 0, flux, np.nan), axis=0)
        global_sky = Interp1D_Grid(points=wave_sky, values=flux_sky, method="cubic")(waveimg)

        # Save the 2D spectrum wavelength solution & distance from the trace to a fits file
        primary_hdu = fits.PrimaryHDU()
        hdr = primary_hdu.header
        hdr["RAWFILE"] = spec2d_file
        hdr["DET"] = det

        # Save the distance array
        hdu_dist = fits.ImageHDU(dist, name="DIST")
        hdu_dist.header["UNIT"] = "arcsec"
        hdu_dist.header["COMMENT"] = "Distance from the trace"

        # Save the wavelength solution
        hdu_waveimg = fits.ImageHDU(waveimg, name="WAVEIMG")
        hdu_waveimg.header["UNIT"] = "Angstrom"
        hdu_waveimg.header["COMMENT"] = "Wavelength solution"

        # Save the global sky background
        hdu_global_sky = fits.ImageHDU(global_sky, name="GLOBALSKY")
        hdu_global_sky.header["COMMENT"] = "Global sky background (average across the slit)"

        hdul = fits.HDUList([primary_hdu, hdu_dist, hdu_waveimg, hdu_global_sky])
        hdul.writeto(spec2d_file.replace(".fits", "_preproc.fits"), overwrite=True)

        # Remove spatial pixels outside the slit (all spat values are NaN)
        valid_spat = jnp.any(np.isfinite(dist), axis=1)
        # Remove spectral pixels outside the wavelength range (some spec values are NaN)
        valid_spec = jnp.all(np.isfinite(dist[valid_spat]), axis=1)

        dist = dist[valid_spat][valid_spec]
        waveimg = waveimg[valid_spat][valid_spec]
        flux = flux[valid_spat][valid_spec]
        ivar = ivar[valid_spat][valid_spec]
        global_sky = global_sky[valid_spat][valid_spec]

        # _, ax = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
        # ax[0].imshow(
        #     flux,
        #     origin="lower",
        #     aspect="auto",
        #     cmap="gray",
        #     vmin=np.nanpercentile(flux, 5),
        #     vmax=np.nanpercentile(flux, 95),
        # )
        # ax[0].set_title("Flux")
        # ax[1].imshow(
        #     global_sky,
        #     origin="lower",
        #     aspect="auto",
        #     cmap="gray",
        #     vmin=np.nanpercentile(flux, 5),
        #     vmax=np.nanpercentile(flux, 95),
        # )
        # ax[1].set_title("Sky Prediction")
        # ax[2].imshow(
        #     flux - global_sky,
        #     origin="lower",
        #     aspect="auto",
        #     cmap="gray",
        #     vmin=np.nanpercentile(flux - global_sky, 5),
        #     vmax=np.nanpercentile(flux - global_sky, 95),
        # )
        # ax[2].set_title("Sky Subtracted")
        # plt.show()

        if spat_rect is None:
            # Spatial coordinates: within the range of the slit
            # Remove the first and last spatial pixels to avoid the edge effects
            spat_rect = jnp.arange(dist[0].max() // pixel_scale, dist[-1].min() // pixel_scale + 1)[1:-1] * pixel_scale
            msgs.info(f"Distance from the trace: {spat_rect[0]:.2f} - {spat_rect[-1]:.2f} arcsec")

        if spec_rect is None:
            # Spectral coordinates: at the location of the trace
            # Remove the first and last spectral pixels to avoid the edge effects
            spec_rect = trace_spec[1:-1]
            msgs.info(f"Wavelength range: {spec_rect[0]:.2f} - {spec_rect[-1]:.2f} Angstrom")

        return cls(
            pixel_scale=pixel_scale,
            center_ra=ra,
            center_dec=dec,
            slit_wid=slit_wid,
            position_angle=position_angle,
            spat_resln=spat_resln,
            spec_resln=spec_resln,
            flux=flux,
            flux_global_sky=global_sky,
            flux_ivar=ivar,
            dist=dist,
            waveimg=waveimg,
            spat_rect=spat_rect,
            spec_rect=spec_rect,
            cache_path=spec2d_file.replace(".fits", "_rect.fits"),
            to_caches=True,
            **kwargs,
        )

    @classmethod
    def from_fits(cls, fits_path: str = None, **kwargs):
        """
        Load 2D spectra from cache files.

        Parameters
        ----------
        cache_path : str, optional (default: ".cache.json")
            The path to the cache file.
        """
        if fits_path is None:
            raise ValueError("No fits file provided.")

        msgs.info(f"Loading 2D spectrum from {fits_path}...")

        try:
            with fits.open(fits_path) as f:
                header = f[0].header
                data = dict(
                    pixel_scale=header["PIXSCALE"],
                    center_ra=header["CENRA"],
                    center_dec=header["CENDEC"],
                    slit_wid=header["SLITWID"],
                    position_angle=header["POSANG"],
                    spat_resln=header["SPATRESLN"],
                    spec_resln=header["SPECRESLN"],
                    spat_rect=np.array(f["SPAT"].data, dtype=np.float64),
                    spec_rect=np.array(f["SPEC"].data, dtype=np.float64),
                    flux_rect=np.array(f["FLUX"].data, dtype=np.float64),
                    flux_ivar_rect=np.array(f["IVAR"].data, dtype=np.float64),
                    sky_offset=header["SKYOFFSET"],
                    cache_path=header["SPECFILE"],
                    to_caches=False,
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Fits file {fits_path} not found.")
        return cls(**data, **kwargs)

    @classmethod
    @show_and_save
    def coadd2d(cls, spec_data_list: list["SpecData"], **kwargs):
        """
        Coadd multiple SpecData objects.

        Parameters
        ----------
        spec_data_list : list[SpecData]
            A list of SpecData objects to be coadded.
        """

        from astropy.stats import mad_std

        if len(spec_data_list) == 0:
            raise ValueError("No SpecData object provided.")

        if len(spec_data_list) == 1:
            return spec_data_list[0]

        # Check if the pixel scales are the same
        if not all(spec_data.pixel_scale == spec_data_list[0].pixel_scale for spec_data in spec_data_list):
            raise ValueError("All SpecData objects must have the same pixel scale.")

        # Check if the spatial and spectral coordinates are the same
        if not all(
            (spec_data.spat_rect == spec_data_list[0].spat_rect).all()
            and (spec_data.spec_rect == spec_data_list[0].spec_rect).all()
            for spec_data in spec_data_list
        ):
            raise ValueError("All SpecData objects must have the same spatial and spectral coordinates.")

        # Check if the flux and ivar arrays have the same shape
        if not all(
            (spec_data.flux_rect.shape == spec_data_list[0].flux_rect.shape)
            and (spec_data.flux_ivar_rect.shape == spec_data_list[0].flux_ivar_rect.shape)
            for spec_data in spec_data_list
        ):
            raise ValueError("All SpecData objects must have the same flux and ivar arrays.")

        msgs.info(f"Coadding 2D spectra from {len(spec_data_list)} objects...")

        # Coadd the flux and ivar arrays
        flux_rect_stack = jnp.stack([spec_data.flux_rect for spec_data in spec_data_list], axis=0)
        flux_ivar_rect_stack = jnp.stack([spec_data.flux_ivar_rect for spec_data in spec_data_list], axis=0)
        flux_err_rect_stack = flux_ivar_rect_stack**-0.5

        # Calculate weighted means
        valid_mask = np.isfinite(flux_rect_stack) & np.isfinite(flux_err_rect_stack)
        cr_mask = np.ones_like(flux_rect_stack, dtype=bool)
        pos_outlier = np.ones_like(flux_rect_stack, dtype=bool)
        neg_outlier = np.ones_like(flux_rect_stack, dtype=bool)
        diff = np.zeros_like(flux_rect_stack)
        if len(spec_data_list) > 1:
            # Try identifying cosmic rays with image subtraction
            for i in range(len(spec_data_list)):
                diff[i] = (flux_rect_stack[i] - jnp.nanmedian(flux_rect_stack, axis=0)) / flux_err_rect_stack[i]
                pos_outlier[i] = diff[i] > 5 * mad_std(diff[i][np.isfinite(diff[i])])
                neg_outlier[i] = diff[i] < -5 * mad_std(diff[i][np.isfinite(diff[i])])
            
            for i in range(len(spec_data_list)):
                # Mask the pixels with large positive deviations (cosmic rays)
                cr_mask[i] = ~pos_outlier[i]

                for j in range(len(spec_data_list)):
                    if i != j:
                        # Mask the pixels with large negative deviations in other frames
                        cr_mask[i] &= ~neg_outlier[j] | neg_outlier[i]
                

        w = flux_ivar_rect_stack
        weights = np.where(valid_mask & cr_mask, w, 0)
        weighted_values = np.where(valid_mask & cr_mask, flux_rect_stack * w, 0)

        flux_rect = np.sum(weighted_values, axis=0) / np.sum(weights, axis=0)

        cmap = plt.get_cmap("gray")
        cmap.set_bad("red", 1.0)
        _, ax = plt.subplots(
            len(spec_data_list) + 1,
            1,
            figsize=(12, 4 * (len(spec_data_list) + 1)),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for k in range(len(spec_data_list)):
            ax[k].imshow(
                diff[k], # - global_sky_stack[k][np.newaxis, :],
                cmap=cmap,
                origin="lower",
                aspect="auto",
                vmin=-5,
                vmax=5,
                # vmin=np.nanpercentile(flux_rect, 5),
                # vmax=np.nanpercentile(flux_rect, 95),
            )
            ax[k].set_title(f"Object {k+1} - Mean")
            ax[k].set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[-1].imshow(
            flux_rect,
            cmap=cmap,
            origin="lower",
            aspect="auto",
            vmin=np.nanpercentile(flux_rect, 5),
            vmax=np.nanpercentile(flux_rect, 95),
        )
        ax[-1].set_xlabel(r"$\mathrm{Spec\ [pixel]}$")
        ax[-1].set_ylabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[-1].set_title("Coadded")

        # Calculate errors
        weighted_errors = np.where(valid_mask & cr_mask, (flux_err_rect_stack * w) ** 2, 0)
        flux_err_rect = np.sqrt(np.sum(weighted_errors, axis=0) / np.sum(weights, axis=0) ** 2)
        flux_ivar_rect = np.where(np.isfinite(flux_err_rect), flux_err_rect**-2, 0)

        return cls(
            pixel_scale=spec_data_list[0].pixel_scale,
            center_ra=spec_data_list[0].center_ra,
            center_dec=spec_data_list[0].center_dec,
            slit_wid=spec_data_list[0].slit_wid,
            position_angle=spec_data_list[0].position_angle,
            spat_resln=spec_data_list[0].spat_resln,
            spec_resln=spec_data_list[0].spec_resln,
            spat_rect=spec_data_list[0].spat_rect,
            spec_rect=spec_data_list[0].spec_rect,
            flux_rect=flux_rect,
            flux_ivar_rect=flux_ivar_rect,
            **kwargs,
        )

    def to_fits(self):
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

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()

        # Add headers
        hdr = primary_hdu.header
        hdr["PIXSCALE"] = public_data["pixel_scale"]
        hdr["CENRA"] = public_data["center_ra"]
        hdr["CENDEC"] = public_data["center_dec"]
        hdr["SLITWID"] = public_data["slit_wid"]
        hdr["POSANG"] = public_data["position_angle"]
        hdr["SPATRESLN"] = public_data["spat_resln"]
        hdr["SPECRESLN"] = public_data["spec_resln"]
        hdr["SPECFILE"] = public_data["cache_path"]
        hdr["SKYOFFSET"] = public_data["sky_offset"]

        # Create HDUs
        spat_hdu = fits.ImageHDU(public_data["spat_rect"], name="SPAT")
        spat_hdu.header["UNIT"] = "arcsec"
        spat_hdu.header["COMMENT"] = "Spatial coordinates"
        spec_hdu = fits.ImageHDU(public_data["spec_rect"], name="SPEC")
        spec_hdu.header["UNIT"] = "Angstrom"
        spec_hdu.header["COMMENT"] = "Spectral coordinates"
        flux_hdu = fits.ImageHDU(public_data["flux_rect"], name="FLUX")
        flux_hdu.header["COMMENT"] = "Flux values"
        ivar_hdu = fits.ImageHDU(public_data["flux_ivar_rect"], name="IVAR")
        ivar_hdu.header["COMMENT"] = "Inverse variance values"

        # Create HDU list
        hdul = fits.HDUList([primary_hdu, spat_hdu, spec_hdu, flux_hdu, ivar_hdu])

        # Save to fits file
        # fits_path = self.spec2d_file.replace(".fits", "_rect.fits")
        hdul.writeto(public_data["cache_path"], overwrite=True)

    def to_SpecModel(
        self,
        # slit_len: float = None,
        # spec_range: tuple[float, float] | list[float] = None,
        **kwargs,
    ) -> SpecModel:
        """
        Convert the 2D spectra to a SpecModel object.

        Parameters
        ----------
        spec_range : tuple or list, optional (default: None)
            The range of the spectral pixels to include.
        """
        # if slit_len is None:
        #     spat_mask = jnp.ones_like(self.spat_rect, dtype=bool)
        # else:
        #     spat_mask = (self.spat_rect >= -slit_len / 2) & (self.spat_rect <= slit_len / 2)

        # if spec_range is None:
        #     spec_mask = jnp.ones_like(self.spec_rect, dtype=bool)
        # else:
        #     spec_mask = (self.spec_rect >= spec_range[0]) & (self.spec_rect <= spec_range[1])

        # Update the spectral and spatial resolutions if specified in the config file
        spec_resln_cfg = kwargs.pop("spec_resln", None)
        spat_resln_cfg = kwargs.pop("spat_resln", None)
        if spec_resln_cfg is not None:
            self.spec_resln = spec_resln_cfg
            msgs.info(f"Spectral resolution specified in the config file: setting it to {self.spec_resln:.2f} Ang.")
        if spat_resln_cfg is not None:
            self.spat_resln = spat_resln_cfg
            msgs.info(f"Spatial resolution specified in the config file: setting it to {self.spat_resln:.2f} arcsec.")

        return SpecModel(
            dat=self.flux_rect,
            dat_err=self.flux_ivar_rect**-0.5,
            spat=self.spat_rect,
            spec=self.spec_rect,
            pixel_scale=self.pixel_scale,
            center_ra=self.center_ra,
            center_dec=self.center_dec,
            slit_wid=self.slit_wid,
            position_angle=self.position_angle,
            spat_resln=self.spat_resln,
            spec_resln=self.spec_resln,
            **kwargs,
        )

    def update_pypeit_skymodel(self, spec_model: SpecModel, spec2d_file: str):
        """
        Update the sky model in the PypeIt spec2d file.
        """
        import os
        from pypeit import spec2dobj

        preproc_file = spec2d_file.replace(".fits", "_preproc.fits")
        rect_file = spec2d_file.replace(".fits", "_rect.fits")
        output_file = spec2d_file.replace(".fits", "_hostsub.fits")

        if not (os.path.exists(spec2d_file) | os.path.exists(rect_file)):
            raise FileNotFoundError("Spec2D file or rectified file not found.")

        # The new sky model will be composed of
        ###### (i) the preliminary global sky (average across the slit) removed before the rectification
        ###### (ii) the global sky (average of the sky region) subtracted after the rectification
        ###### (iii) the local sky modeled with HostSub

        # Preliminary sky line removal - reduce the noise introduced in the rectification
        ###### Set the sky model to NaN if the corresponding pixels are excluded from the fit
        with fits.open(preproc_file) as hdul_preproc:
            waveimg = jnp.array(hdul_preproc["WAVEIMG"].data, dtype=jnp.float32)
            dist = jnp.array(hdul_preproc["DIST"].data - self.sky_offset, dtype=jnp.float32)
            global_sky_pre = jnp.array(hdul_preproc["GLOBALSKY"].data, dtype=jnp.float32)
            det = hdul_preproc[0].header["DET"]

        # mask_model = (
        #     (dist >= spec_model.spat[0])
        #     & (dist <= spec_model.spat[-1])
        #     & (waveimg >= spec_model.spec[0])
        #     & (waveimg <= spec_model.spec[-1])
        # )
        # global_sky_pre = np.where(mask_model, global_sky_pre, np.nan)

        # The global sky subtracted after the rectification
        global_sky_post = Interp1D_Grid(
            points=spec_model.f_sky_1d.X.ravel(), values=spec_model.f_sky_1d.y, method="cubic"
        )(waveimg)

        # The local sky
        X = np.stack([dist.ravel(), waveimg.ravel()], axis=-1).reshape(-1, 2)
        spat_mask = (dist.ravel() >= spec_model.spat_edges["host"][0]) & (dist.ravel() <= spec_model.spat_edges["host"][-1])
        spec_mask = (waveimg.ravel() >= spec_model.spec[0]) & (waveimg.ravel() <= spec_model.spec[-1])
        local_mask = spat_mask & spec_mask
        
        raise NotImplementedError

        breakpoint()
        

        sky_host_prior, _ = spec_model.host_prior(X[local_mask])
        sky_pred, sky_pred_var = spec_model._get_pred(spec_model._gp_1d, spec_model._gp_2d, X[local_mask], return_var=True)

        sky_local = np.zeros_like(dist.ravel())
        sky_local[local_mask] = sky_pred + sky_host_prior
        sky_local = sky_local.reshape(dist.shape)

        sci2d = spec2dobj.Spec2DObj.from_file(spec2d_file, detname=det)
        # assert sci2d.sciimg.shape == sky_model.T.shape
        sci2d.skymodel = (global_sky_pre + global_sky_post + sky_local).T

        all_spec2d = spec2dobj.AllSpec2DObj()
        all_spec2d[det] = sci2d

        with fits.open(spec2d_file) as hdul:
            pri_hdr = hdul[0].header

        all_spec2d.write_to_fits(output_file, pri_hdr=pri_hdr)

    def _rectify(
        self,
        points: ArrayLike,
        f_values: tuple[ArrayLike, ArrayLike, ArrayLike],
        batch_size: int = 1024,
        interp_method: str = "rbf",
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Rectify the 2D spectrum onto a grid.

        Parameters
        ----------
        points : ArrayLike
            The spatial and spectral pixel coordinates.
        f_values : tuple[ArrayLike, ArrayLike, ArrayLike]
            The flux, ivar, and flag values.
        batch_size : int, optional (default: 1024, in pixels)
            The batch size for interpolation.
        interp_method : str, optional (default: "rbf")
            The interpolation method.
        """

        from .interp import Interp2D_RBF, Interp2D_Linear, Interp2D_Scipy

        if ~jnp.all(jnp.isfinite(points)):
            raise ValueError("Some points are NaN.")

        if interp_method not in ["rbf", "linear", "scipy"]:
            raise ValueError("Invalid interpolation method.")
        elif interp_method == "rbf":
            msgs.info("Interpolating the flux with RBF...")
        elif interp_method == "linear":
            msgs.info("Interpolating the flux with linear method...")
        else:
            msgs.info("Interpolating the flux with scipy.interpolate.griddata...")

        # Initialize the flux and ivar arrays on a semi-rectified grid
        # The spatial/spectral coordinate monitonically increases in each row/column
        flux, ivar = f_values

        spec_pix_rect, spat_pix_rect = jnp.meshgrid(self.spec_rect, self.spat_rect)
        spec_batch_idx = jnp.array_split(jnp.arange(len(self.spec_rect)), len(self.spec_rect) // batch_size + 1)

        flux_rect = np.zeros((len(self.spat_rect), len(self.spec_rect)))
        flux_ivar_rect = np.zeros((len(self.spat_rect), len(self.spec_rect)))

        scales = (self.spat_resln / 2.355, self.spec_resln / 2.355)

        if interp_method == "rbf":
            interp2d = Interp2D_RBF(kernel="gaussian", scales=scales)
            interp2d_ivar = Interp2D_RBF(kernel="gaussian", scales=scales)

        elif interp_method == "linear":
            interp2d = Interp2D_Linear(scales=scales)
            interp2d_ivar = Interp2D_Linear(scales=scales)

        elif interp_method == "scipy":
            interp2d = Interp2D_Scipy(method="linear", scales=scales)
            interp2d_ivar = Interp2D_Scipy(method="linear", scales=scales)

        # Interpolate the flux in batches
        for idx_list in spec_batch_idx:
            msgs.info(
                f"Interpolating the flux in the spectral range {self.spec_rect[idx_list[0]]:.2f} - {self.spec_rect[idx_list[-1]]:.2f} Ang..."
            )
            # The range of the spectrum to interpolate
            spec_min = idx_list[0]
            spec_max = idx_list[-1] + 1
            # Initialize the interpolators (with padding)
            fit_min = max(spec_min - 1, 0)
            fit_max = min(spec_max + 1, points.shape[1])
            interp2d.fit(points=points[:, fit_min:fit_max], values=flux[:, fit_min:fit_max])
            interp2d_ivar.fit(points=points[:, fit_min:fit_max], values=ivar[:, fit_min:fit_max])
            # Query points
            query_points = jnp.stack(
                [spat_pix_rect[:, spec_min:spec_max].ravel(), spec_pix_rect[:, spec_min:spec_max].ravel()], axis=-1
            )
            flux_rect[:, spec_min:spec_max] = interp2d.predict(query_points=query_points).reshape(
                flux_rect[:, spec_min:spec_max].shape
            )
            flux_ivar_rect[:, spec_min:spec_max] = interp2d_ivar.predict(query_points=query_points).reshape(
                flux_ivar_rect[:, spec_min:spec_max].shape
            )

        return flux_rect, flux_ivar_rect

    @show_and_save
    def _get_offset(self, points: ArrayLike, flux: ArrayLike, host_prior: Callable, mask_wid: float = 2.0) -> float:
        """
        Center the trace of the science object.
        """

        def binned_mean_with_clipping(
            points: ArrayLike, values: ArrayLike, bins: int, bin_range: tuple[float, float], **kwargs
        ):
            from astropy.stats import sigma_clip, mad_std

            bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1)
            bin_indices = np.digitize(points, bin_edges) - 1

            # Initialize an array to store the sigma-clipped statistics
            obs = np.full(bins, np.nan)

            # Iterate over each bin
            n_bin = np.array([np.sum(bin_indices == i) for i in range(bins)])
            med_n_bin = np.median(n_bin)
            std_n_bin = mad_std(n_bin)
            for i in range(bins):
                # Skip the bin if there are too few points
                # Sometimes std_n_bin can be zero, which means essentially all bins should be included
                if n_bin[i] < med_n_bin - 3 * (std_n_bin + 1):
                    continue
                # Get the indices of points in the current bin
                bin_mask = bin_indices == i
                valid_mask = np.isfinite(values)

                # Extract the values in the current bin
                y_in_bin = values[bin_mask & valid_mask]

                # Apply sigma clipping to the values in the bin
                y_clipped = sigma_clip(y_in_bin, stdfunc="mad_std", **kwargs)

                # Compute the statistic (e.g., mean) of the clipped values (requires at least 80% valid data)
                if (~y_clipped.mask).sum() > 0.80 * len(y_clipped):
                    obs[i] = np.nanmean(y_clipped[~y_clipped.mask])

            obs[obs <= 0] = np.nan

            return obs

        slit_len_max = min(-self.spat_rect[0], self.spat_rect[-1]) * 2
        spat = self.spat_rect[np.abs(self.spat_rect) <= slit_len_max / 2]

        obs = binned_mean_with_clipping(
            points[..., 0],
            flux,
            bins=len(spat),
            bin_range=(spat[0] - self.pixel_scale / 2, spat[-1] + self.pixel_scale / 2),
            sigma=5,
        )

        sci_obj_mask = (jnp.abs(spat) >= mask_wid) & jnp.isfinite(obs)

        # Profile from the prior - flux evaluated at the weighted-mean wavelength
        wv_mean = jnp.nanmean(points[..., 1] * flux) / jnp.nanmean(flux) * jnp.ones_like(spat)

        def corr_coef(offset):
            # Evaluate the profile from the prior at the offset position
            prior = host_prior(jnp.stack([spat - offset, wv_mean], axis=-1))[0]
            return jnp.corrcoef(
                (prior[sci_obj_mask] - jnp.nanmin(prior[sci_obj_mask]))
                / (jnp.nanmax(prior[sci_obj_mask]) - jnp.nanmin(prior[sci_obj_mask])),
                (obs[sci_obj_mask] - jnp.nanmin(obs[sci_obj_mask]))
                / (jnp.nanmax(obs[sci_obj_mask]) - jnp.nanmin(obs[sci_obj_mask])),
            )[0, 1]

        offset_list = np.arange(-2, 2 + self.pixel_scale / 10, self.pixel_scale / 10)
        ccf = jax.vmap(corr_coef)(offset_list)

        # F_obs(x_spat) = F_prior(x_spat - offset_opt)
        # => F_obs(offset_opt) = F_prior(0) = Location of the SN
        # => Subtract offset_opt from the spatial coordinates of the 2D spectrum
        offset_opt = offset_list[np.argmax(ccf)]

        _, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        ax[0].plot(offset_list, ccf)
        ax[0].set_xlabel(r"$\mathrm{SCI - STD\ [arcsec]}$")
        ax[0].set_ylabel(r"$\mathrm{Correlation Coefficient}$")
        ax[0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0].xaxis.set_minor_locator(plt.MultipleLocator(0.02))
        ax[0].axvline(offset_opt, color="k", linestyle="--")

        ax[1].scatter(
            spat - offset_opt,
            (obs - jnp.nanmin(obs)) / (jnp.nanmax(obs) - jnp.nanmin(obs)),
            label="obs",
        )

        profile_prior = host_prior(
            jnp.stack(
                [
                    spat - offset_opt,
                    jnp.nanmean(points[:, :, 1] * flux) / jnp.nanmean(flux) * jnp.ones_like(spat),
                ],
                axis=-1,
            )
        )[0]
        ax[1].scatter(
            spat - offset_opt,
            (profile_prior - profile_prior.min()) / (profile_prior.max() - profile_prior.min()),
            label="prior",
        )
        ax[1].set_xlabel(r"$\mathrm{Spat\ [arcsec]}$")
        ax[1].set_ylabel(r"$\mathrm{Normalized\ Counts}$")
        ax[1].xaxis.set_major_locator(plt.MultipleLocator(5))
        ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax[1].legend()

        return offset_opt
