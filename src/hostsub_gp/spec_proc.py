# hostsub_gp/spec_proc.py

__all__ = ["SpecData"]

import numpy as np
import jax
import jax.numpy as jnp
import os

# jax.config.update("jax_enable_x64", True)

from jax._src.typing import ArrayLike, Array
from typing import Callable, Optional, Literal

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from .interp import Interp1D_Grid, Interp2D_Grid
from .interp import Interp2D_Scipy, Interp2D_RBF, Interp2D_Linear
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
        flux_rect: Optional[Array] = None,
        flux_ivar_rect: Optional[Array] = None,
        dist: Optional[Array] = None,
        waveimg: Optional[Array] = None,
        flux: Optional[Array] = None,
        flux_global_sky: Optional[Array] = None,
        flux_ivar: Optional[Array] = None,
        sky_offset: Optional[float] = None,
        survey: Literal["PS1", "LS"] = "PS1",
        to_caches: Optional[bool] = False,
        cache_path: Optional[str] = None,
    ):
        assert np.all(spat_rect[1:] > spat_rect[:-1]), (
            "Spatial coordinates must be in ascending order."
        )
        assert np.all(spec_rect[1:] > spec_rect[:-1]), (
            "Spectral coordinates must be in ascending order."
        )
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
            assert dist is not None, "Distance array is not available."
            assert flux is not None and flux_ivar is not None, (
                "Flux and ivar arrays are not available."
            )

            if self.sky_offset is None:
                slit_len = (
                    min(
                        self.spat_rect.max().item() - self.spat_rect.min().item(),
                        30 // self.pixel_scale * self.pixel_scale,
                    )
                    + self.pixel_scale
                )
                img_products = HostProfile.load_archival_images(
                    center_ra=self.center_ra,
                    center_dec=self.center_dec,
                    slit_len=slit_len,
                    slit_wid=self.slit_wid,
                    position_angle=self.position_angle,
                    survey=survey,
                )
                host_prior = HostProfile.from_archival(
                    img_products=img_products,
                    slit_len=slit_len,
                    slit_wid=self.slit_wid,
                    pixel_scale=self.pixel_scale,
                ).model_host_profile_prior()

                self.sky_offset = self._get_offset(
                    points=jnp.stack([dist, waveimg], axis=-1),
                    flux=flux,
                    show=True,
                    mask_wid=2.0,
                    host_prior=host_prior,
                )
            self._points = jnp.stack([dist - self.sky_offset, waveimg], axis=-1)

            # Rectify the 2D spectrum
            if flux_global_sky is not None:
                self.flux_rect, self.flux_ivar_rect = self._rectify(
                    points=self._points,
                    f_values=(flux - flux_global_sky, flux_ivar),
                    interp_method="scipy",
                )
            else:
                self.flux_rect, self.flux_ivar_rect = self._rectify(
                    points=self._points,
                    f_values=(flux, flux_ivar),
                    interp_method="scipy",
                )

            # Save the 2D spectra to cache files
            self.cache_path = cache_path
            if to_caches:
                self.to_fits()

    @classmethod
    def from_pypeit(
        cls,
        sci_file: str,
        raw_dir: Optional[str] = None,
        obj_id: Optional[str] = None,
        std_file: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        spat_resln: Optional[float] = None,
        spat_rect: Optional[ArrayLike] = None,
        spec_rect: Optional[ArrayLike] = None,
        rect_file: Optional[str] = None,
        preproc_file: Optional[str] = None,
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
        raw_file = pypeit_header["FILENAME"]

        # Items to load
        # - RA, DEC (in degrees)
        # - detector ID
        # - pixel scale (in arcsec/pixel)
        # - position angle (in degrees)
        # - slit width (in arcsec)

        def _get_raw_header(
            raw_dir: str | None, raw_file: str, **kwargs
        ) -> fits.Header:
            """
            Get the header of the raw file.
            """
            if raw_dir is None:
                raise ValueError(
                    f"The raw file directory is needed for {pypeit_header['PYP_SPEC']} data."
                )
            return fits.getheader("/".join([raw_dir, raw_file]), **kwargs)

        bad_rows = []

        if pypeit_header["PYP_SPEC"] in [
            "keck_lris_blue",
            "keck_lris_red",
            "keck_lris_red_mark4",
        ]:
            # Keck/LRIS

            if ra is None or dec is None:
                ra, dec = pypeit_header["RA"], pypeit_header["DEC"]

            if "red_mark4" in pypeit_header["PYP_SPEC"]:
                det = "DET01"
            else:
                det = "DET02"

            binning = int(
                pypeit_header["BINNING"].split(",")[1]
            )  # in the spatial direction
            pixel_scale = 0.135 * binning

            position_angle = pypeit_header["ROTPOSN"] + 90

            slit_wid = float(pypeit_header["SLITNAME"].split("_")[-1])

        elif pypeit_header["PYP_SPEC"] == "mmt_binospec":
            # MMT/Binospec

            raw_header = _get_raw_header(raw_dir, raw_file, ext=1)

            if ra is None or dec is None:
                # RA and DEC in the header are in the format of 'HH:MM:SS.SS' and 'DD:MM:SS.SS'
                ra_str, dec_str = (
                    str(raw_header["CATRA"]).strip("'"),
                    str(raw_header["CATDEC"]).strip("'"),
                )
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra, dec = coord.ra.deg, coord.dec.deg

            det = "DET02"

            pixel_scale = 0.24

            # PA: parallactic angle
            # ROT: instrument rotator angle (relative to the parallactic angle)
            position_angle = raw_header["PA"] - raw_header["ROT"]

            slit_wid = float(raw_header["MASK"].split("Longslit")[-1])

            # MMT faulty rows to be removed
            bad_rows = [113, 210, 719, 1999, 2099, 3336, 3337, 4056, 4057] + [1717, 3011]

        elif pypeit_header["PYP_SPEC"] == "not_alfosc":
            # NOT/ALFOSC

            raw_header = _get_raw_header(raw_dir, raw_file, ext=0)

            if ra is None or dec is None:
                ra, dec = pypeit_header["RA"], pypeit_header["DEC"]

            det = "DET01"

            binning = int(
                pypeit_header["BINNING"].split(",")[1]
            )  # in the spatial direction
            pixel_scale = 0.2138 * binning

            position_angle = raw_header["FIELD"] + 180

            # Slit width format: "Slit_1.3"
            slit_wid = float(pypeit_header["DECKER"].split("_")[-1])

        elif pypeit_header["PYP_SPEC"] == "vlt_fors2":
            # VLT/FORS2

            raw_header = _get_raw_header(raw_dir, raw_file, ext=0)

            if ra is None or dec is None:
                ra, dec = pypeit_header["RA"], pypeit_header["DEC"]

            det = "DET01"

            pixel_scale = 0.25

            position_angle = -raw_header["HIERARCH ESO ADA POSANG"] + 180

            # Slit width format: "Slit1_0arcsec"
            slit_wid = float(
                ".".join(
                    pypeit_header["DECKER"]
                    .split("Slit")[-1]
                    .split("arcsec")[0]
                    .split("_")
                )
            )

        else:
            raise NotImplementedError(
                "Only LRIS, Binospec, ALFOSC, and FORS2 are supported"
            )

        # If the object ID in the science frame is provided (i.e., object successfully found), use the object trace
        if obj_id is not None:
            trace_file = spec1d_file
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            name_idx = trace_objs.name_indices(obj_id)
            if not any(name_idx):
                raise ValueError(f"Object {obj_id} not found in the trace file.")
            trace_obj = trace_objs[np.where(name_idx)[0][0]]

        # If the object ID is not provided, use the standard star trace
        elif std_file is not None:
            trace_file = std_file
            trace_objs = specobjs.SpecObjs.from_fitsfile(trace_file, det=det)
            # Find the SpecObj with the highest signal-to-noise ratio (S2N) in the SpecObjs
            objs2n = []
            for obj in trace_objs:
                assert obj is not None
                objs2n.append(obj["S2N"])
            trace_obj = trace_objs[np.argmax(objs2n)]
        else:
            raise ValueError("No spec1d file provided for identifying the trace.")

        if spat_resln is None:
            if std_file is None:
                raise ValueError(
                    "The spatial resolution needs to be either provided in the config file or estimated from the standard."
                )
            std_objs = specobjs.SpecObjs.from_fitsfile(std_file, det=det)
            objs2n = []
            for obj in std_objs:
                assert obj is not None
                objs2n.append(obj["S2N"])
            trace_obj = std_objs[np.argmax(objs2n)]
            spat_resln = trace_obj["FWHM"] * pixel_scale

        assert trace_obj is not None, "Trace object is not available."
        trace_spat_pix = jnp.asarray(
            trace_obj["TRACE_SPAT"], dtype=float
        )  # spatial pixel of the trace

        sci2d = spec2dobj.Spec2DObj.from_file(spec2d_file, detname=det)

        # Refill bad rows
        for row in bad_rows:
            sci2d.sciimg[row, :] = (
                sci2d.sciimg[row - 1, :] + sci2d.sciimg[row + 1, :]
            ) / 2
            sci2d.ivarraw[row, :] = (
                sci2d.ivarraw[row - 1, :] + sci2d.ivarraw[row + 1, :]
            ) / 2

        # Spectral resolution: the FWHM of the arc lines
        try:
            spec_resln = sci2d["wavesol"]["measured_fwhm"].value[0]
        except KeyError:
            spec_resln = sci2d["wavesol"]["mesured_fwhm"].value[
                0
            ]  # A typo in old PypeIt versions

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
        trace_spec = Interp2D_Grid(
            points=(jnp.arange(waveimg.shape[0]), jnp.arange(waveimg.shape[1])),
            values=jnp.asarray(waveimg),
        )(np.stack([trace_spat_pix, np.arange(waveimg.shape[1])], axis=-1))
        assert trace_spec is not None, "Trace spectral pixels are not available."
        trace_spec_pix = jnp.where(
            tilts != 0,
            Interp1D_Grid(
                points=jnp.asarray(trace_spec), values=jnp.arange(len(trace_spec))
            )(waveimg),
            np.nan,
        )

        # indices of the spatial and spectral pixels
        spat_pix = jnp.tile(jnp.arange(waveimg.shape[0]), (waveimg.shape[1], 1)).T
        spec_pix = jnp.tile(jnp.arange(waveimg.shape[1]), (waveimg.shape[0], 1))

        dist_spat_pix = spat_pix - trace_spat_pix
        dist_spec_pix = spec_pix - trace_spec_pix
        dist = (
            np.sqrt(dist_spat_pix**2 + dist_spec_pix**2)
            * np.where(dist_spat_pix > 0, 1, -1)
            * pixel_scale
        )

        # Preliminary sky line removal - reduce the noise introduced in the rectification
        wave_sky = np.nanmedian(np.where(waveimg > 0, waveimg, np.nan), axis=0)
        flux_sky = np.nanmedian(np.where(waveimg > 0, flux, np.nan), axis=0)
        global_sky = Interp1D_Grid(points=wave_sky, values=flux_sky, method="cubic")(
            waveimg
        )

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
        hdu_global_sky.header["COMMENT"] = (
            "Global sky background (average across the slit)"
        )

        if preproc_file is None:
            preproc_file = spec2d_file.replace(".fits", "_preproc.fits")
        preproc_dir = os.path.dirname(preproc_file)
        if preproc_dir:
            os.makedirs(preproc_dir, exist_ok=True)
        hdul = fits.HDUList([primary_hdu, hdu_dist, hdu_waveimg, hdu_global_sky])
        hdul.writeto(preproc_file, overwrite=True)

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
            spat_rect = (
                jnp.arange(
                    dist[0].max() // pixel_scale, dist[-1].min() // pixel_scale + 1
                )[1:-1]
                * pixel_scale
            )
            msgs.info(
                f"Distance from the trace: {spat_rect[0]:.2f} - {spat_rect[-1]:.2f} arcsec"
            )

        if spec_rect is None:
            # Spectral coordinates: at the location of the trace
            # Remove the first and last spectral pixels to avoid the edge effects
            assert trace_spec is not None, "Trace spectral pixels are not available."
            spec_rect = trace_spec[1:-1]
            msgs.info(
                f"Wavelength range: {spec_rect[0]:.2f} - {spec_rect[-1]:.2f} Angstrom"
            )

        assert ra is not None, "RA is not available."
        assert dec is not None, "DEC is not available."
        assert position_angle is not None, "Position angle is not available."

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
            cache_path=rect_file or spec2d_file.replace(".fits", "_rect.fits"),
            to_caches=True,
            **kwargs,
        )

    @classmethod
    def from_fits(cls, fits_path: Optional[str] = None, **kwargs):
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
            f = fits.open(fits_path)
            header = fits.getheader(fits_path, ext=0)
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
            f.close()
        except FileNotFoundError:
            raise FileNotFoundError(f"Fits file {fits_path} not found.")
        return cls(**data, **kwargs)

    @classmethod
    @show_and_save
    def coadd2d(
        cls, spec_data_list: list["SpecData"], **kwargs
    ) -> tuple["SpecData", np.ndarray | None]:
        """
        Coadd multiple SpecData objects.

        Parameters
        ----------
        spec_data_list : list[SpecData]
            A list of SpecData objects to be coadded.

        Returns
        -------
        SpecData
            The coadded SpecData object.
        cr_mask : np.ndarray
            The mask of cosmic rays.
        """

        from astropy.stats import mad_std

        if len(spec_data_list) == 0:
            raise ValueError("No SpecData object provided.")

        if len(spec_data_list) == 1:
            return spec_data_list[0], None

        # Check if the pixel scales are the same
        if not all(
            spec_data.pixel_scale == spec_data_list[0].pixel_scale
            for spec_data in spec_data_list
        ):
            raise ValueError("All SpecData objects must have the same pixel scale.")

        # Check if the spatial and spectral coordinates are the same
        if not all(
            (spec_data.spat_rect == spec_data_list[0].spat_rect).all()
            and (spec_data.spec_rect == spec_data_list[0].spec_rect).all()
            for spec_data in spec_data_list
        ):
            raise ValueError(
                "All SpecData objects must have the same spatial and spectral coordinates."
            )

        # Check if the flux and ivar arrays have the same shape
        if not all(
            (spec_data.flux_rect.shape == spec_data_list[0].flux_rect.shape)
            and (
                spec_data.flux_ivar_rect.shape == spec_data_list[0].flux_ivar_rect.shape
            )
            for spec_data in spec_data_list
        ):
            raise ValueError(
                "All SpecData objects must have the same flux and ivar arrays."
            )

        msgs.info(f"Coadding 2D spectra from {len(spec_data_list)} objects...")

        # Coadd the flux and ivar arrays
        flux_rect_stack = jnp.stack(
            [spec_data.flux_rect for spec_data in spec_data_list], axis=0
        )
        flux_ivar_rect_stack = jnp.stack(
            [spec_data.flux_ivar_rect for spec_data in spec_data_list], axis=0
        )
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
                diff[i] = (
                    flux_rect_stack[i] - jnp.nanmedian(flux_rect_stack, axis=0)
                ) / flux_err_rect_stack[i]
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
                diff[k], cmap=cmap, origin="lower", aspect="auto", vmin=-5, vmax=5
            )
            ax[k].set_title(f"Object {k + 1} - Mean")
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
        weighted_errors = np.where(
            valid_mask & cr_mask, (flux_err_rect_stack * w) ** 2, 0
        )
        flux_err_rect = np.sqrt(
            np.sum(weighted_errors, axis=0) / np.sum(weights, axis=0) ** 2
        )
        flux_ivar_rect = np.where(np.isfinite(flux_err_rect), flux_err_rect**-2, 0)

        return (
            cls(
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
                flux_ivar_rect=jnp.asarray(flux_ivar_rect),
                **kwargs,
            ),
            cr_mask,
        )

    def to_fits(self):
        """
        Save the 2D spectra to cache files.

        Parameters
        ----------
        cache_path : str, optional (default: ".cache.json")
            The path to the cache file.
        """

        public_data = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

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
        cache_dir = os.path.dirname(public_data["cache_path"])
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        hdul.writeto(public_data["cache_path"], overwrite=True)

    @staticmethod
    def write_pypeit_spec1d(
        spec_model: SpecModel,
        template_spec1d: str,
        output_file: str,
        obj_id: Optional[str] = None,
        duplicate_to_opt: bool = True,
        extraction: Literal["optimal", "hostsub"] = "optimal",
        pypeit_local_sky: Optional[dict] = None,
    ) -> None:
        """
        Write the HostSub 1D extraction to a PypeIt-style spec1d file.

        The original PypeIt spec1d file is used as a metadata template; only the
        extracted spectral arrays are replaced by HostSub's coadded rectified
        extraction.
        """
        from pypeit import specobjs

        msgs.info(f"Using {template_spec1d} as HostSub spec1d metadata template")
        sobj, header = SpecData._build_hostsub_specobj(
            spec_model=spec_model,
            template_spec1d=template_spec1d,
            obj_id=obj_id,
        )

        if extraction == "optimal":
            source = "HOSTSUB_OPT"
            local_sky_info = SpecData._extract_pypeit_optimal(
                spec_model, sobj, pypeit_local_sky=pypeit_local_sky
            )
        elif extraction == "hostsub":
            source = SpecData._fill_specobj_hostsub_extraction(
                spec_model, sobj, duplicate_to_opt=duplicate_to_opt
            )
            local_sky_info = {"enabled": False, "applied": False}
        else:
            raise ValueError("extraction must be either 'optimal' or 'hostsub'.")

        sobj.set_name()

        counts = np.asarray(sobj.OPT_COUNTS if sobj.OPT_COUNTS is not None else sobj.BOX_COUNTS)
        counts_sig = np.asarray(
            sobj.OPT_COUNTS_SIG
            if sobj.OPT_COUNTS_SIG is not None
            else sobj.BOX_COUNTS_SIG
        )
        good = np.isfinite(counts) & np.isfinite(counts_sig) & (counts_sig > 0)
        sobj.S2N = (
            float(np.nanmedian(np.abs(counts[good]) / counts_sig[good]))
            if np.any(good)
            else 0.0
        )
        sobj.FWHM = spec_model.spat_resln / spec_model.pixel_scale
        sobj.SPAT_FWHM = spec_model.spat_resln / spec_model.pixel_scale

        out_objs = specobjs.SpecObjs([sobj])
        skip_keys = {
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "EXTEND",
            "DMODCLS",
            "DMODVER",
            "NSPEC",
            "COMMENT",
            "HISTORY",
            "",
        }
        subheader = {
            key: header[key]
            for key in header.keys()
            if key not in skip_keys and not key.startswith("EXT") and len(key) <= 8
        }
        subheader["HSTSUB"] = True
        subheader["HSTSRC"] = source
        subheader["HSTNOTE"] = "HostSub_GP coadded rectified 1D extraction"
        subheader["HSTLSKY"] = bool(local_sky_info.get("applied", False))
        if local_sky_info.get("enabled", False):
            subheader["HSTLSBSP"] = float(local_sky_info.get("bsp", 0.6))
            subheader["HSTLSSIG"] = float(local_sky_info.get("sigrej", 3.5))
            subheader["HSTLSORD"] = int(local_sky_info.get("npoly", 1))

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        out_objs.write_to_fits(subheader, output_file, overwrite=True)
        pypeline = getattr(sobj, "PYPELINE", None) or header.get("PYPELINE")
        if pypeline is None:
            pypeline = "MultiSlit"
        txt_file = output_file.replace(".fits", ".txt")
        out_objs.write_info(txt_file, pypeline)
        msgs.info(f"Wrote HostSub PypeIt spec1d file to {output_file}")
        msgs.info(f"Wrote HostSub PypeIt spec1d info file to {txt_file}")

    @staticmethod
    def _build_hostsub_specobj(
        spec_model: SpecModel,
        template_spec1d: str,
        obj_id: Optional[str] = None,
    ):
        """
        Build a fresh PypeIt SpecObj for HostSub output.

        The template is used only for stable PypeIt geometry metadata, such as
        PYPELINE, DET, SLITID, and DETECTOR. Trace-dependent fields like
        SPAT_PIXPOS and TRACE_SPAT are set later from HostSub's corrected
        coordinate system.
        """
        from pypeit import specobj
        from pypeit.images import detector_container

        header = fits.getheader(template_spec1d, ext=0)
        pypeline = header.get("PYPELINE", "MultiSlit")
        det = None
        slitid = None
        detector = None

        with fits.open(template_spec1d) as hdul:
            specobj_hdus = []
            for hdu in hdul[1:]:
                if hdu.header.get("DMODCLS") == "SpecObj":
                    specobj_hdus.append(hdu)
                elif "DETECTOR" in hdu.name:
                    det_from_name = hdu.name.split("-")[0]
                    if det is None:
                        det = det_from_name
                    try:
                        detector = detector_container.DetectorContainer.from_hdu(
                            hdu, chk_version=False
                        )
                    except Exception as err:
                        msgs.warning(
                            f"Could not parse detector metadata from {hdu.name}: {err}"
                        )

            template_hdu = None
            if obj_id is not None:
                for hdu in specobj_hdus:
                    if hdu.name == obj_id:
                        template_hdu = hdu
                        break
                if template_hdu is None:
                    msgs.warning(
                        f"Object {obj_id} not found in {template_spec1d}; using metadata from the first SpecObj HDU."
                    )
            if template_hdu is None and len(specobj_hdus) > 0:
                template_hdu = specobj_hdus[0]

            if template_hdu is not None:
                # These identify the PypeIt slit/detector geometry and should not
                # be confused with the template object's trace position.
                pypeline = template_hdu.header.get("PYPELINE", pypeline)
                det = template_hdu.header.get("DET", det)
                slitid = template_hdu.header.get("SLITID", slitid)

        if det is None:
            det = "DET01"
            msgs.warning("Could not determine detector from template; using DET01.")
        if slitid is None:
            slitid = 0
            msgs.warning("Could not determine slit ID from template; using SLIT0000.")

        sobj = specobj.SpecObj(pypeline, det, OBJTYPE="science", SLITID=int(slitid))
        if detector is not None:
            sobj.DETECTOR = detector
        if spec_model.center_ra is not None:
            sobj.RA = spec_model.center_ra
        if spec_model.center_dec is not None:
            sobj.DEC = spec_model.center_dec
        return sobj, header

    @staticmethod
    def _fill_specobj_hostsub_extraction(
        spec_model: SpecModel,
        sobj,
        duplicate_to_opt: bool = True,
    ) -> str:
        """
        Fill a PypeIt SpecObj with the existing HostSub 1D extraction.
        """
        if hasattr(spec_model, "f_sci_pred_1d"):
            f_sci = spec_model.f_sci_pred_1d
            source = "HOSTSUB_GP"
        elif hasattr(spec_model, "f_sci_bspline_1d"):
            f_sci = spec_model.f_sci_bspline_1d
            source = "HOSTSUB_BSPLINE"
        elif hasattr(spec_model, "f_sci_linear_1d"):
            f_sci = spec_model.f_sci_linear_1d
            source = "HOSTSUB_LINEAR"
        else:
            raise ValueError("No HostSub 1D extraction is available on spec_model.")

        wave = np.asarray(f_sci.X.ravel(), dtype=np.float64)
        counts = np.asarray(f_sci.y, dtype=np.float64)
        counts_sig = np.asarray(f_sci.yerr, dtype=np.float64)
        good = (
            np.isfinite(wave)
            & np.isfinite(counts)
            & np.isfinite(counts_sig)
            & (counts_sig > 0)
        )
        counts_ivar = np.zeros_like(counts_sig)
        counts_ivar[good] = counts_sig[good] ** -2

        box_npix = np.ones_like(counts, dtype=np.float64) * np.sum(
            np.asarray(spec_model.spat_filter["mask"])
        )
        box_radius = 0.5 * (
            spec_model.spat_edges["mask"][-1] - spec_model.spat_edges["mask"][0]
        ) / spec_model.pixel_scale

        sobj.BOX_WAVE = wave
        sobj.BOX_COUNTS = counts
        sobj.BOX_COUNTS_SIG = counts_sig
        sobj.BOX_COUNTS_IVAR = counts_ivar
        sobj.BOX_MASK = good
        sobj.BOX_NPIX = box_npix
        sobj.BOX_RADIUS = float(box_radius)
        sobj.BOX_COUNTS_SKY = np.zeros_like(counts)

        if duplicate_to_opt:
            sobj.OPT_WAVE = wave
            sobj.OPT_COUNTS = counts
            sobj.OPT_COUNTS_SIG = counts_sig
            sobj.OPT_COUNTS_IVAR = counts_ivar
            sobj.OPT_MASK = good

        return source

    @staticmethod
    def _build_pypeit_extraction_inputs(spec_model: SpecModel) -> dict:
        """
        Build PypeIt-shaped arrays from the HostSub coadded rectified model.
        """
        if not hasattr(spec_model, "_f_pred"):
            raise ValueError("Please run the HostSub GP model before optimal extraction.")
        assert spec_model.f_sky_sub.Y is not None and spec_model.f_sky_sub.Yerr is not None

        host_model = np.asarray(spec_model._f_pred.reshape(spec_model.shape))
        imgminsky = np.asarray(spec_model.f_sky_sub.Y) - host_model
        skyimg = np.asarray(spec_model.f_sky_1d.Y)[None, :]
        variance = np.asarray(spec_model.f_sky_sub.Yerr) ** 2
        ivar = np.zeros_like(variance)
        good_var = np.isfinite(variance) & (variance > 0)
        ivar[good_var] = variance[good_var] ** -1

        nspat, nspec = imgminsky.shape
        waveimg = np.tile(np.asarray(spec_model.spec)[None, :], (nspat, 1))
        spat_img = np.tile(np.arange(nspat, dtype=float)[:, None], (1, nspec))
        host_region = (
            (np.asarray(spec_model.spat, dtype=float) >= spec_model.spat_edges["host"][0])
            & (np.asarray(spec_model.spat, dtype=float) <= spec_model.spat_edges["host"][1])
        )[:, None]
        thismask = np.tile(host_region, (1, nspec))
        mask = np.isfinite(imgminsky) & np.isfinite(ivar) & (ivar > 0) & thismask

        # PypeIt extraction routines expect (nspec, nspat).
        return dict(
            imgminsky=imgminsky.T,
            ivar=ivar.T,
            mask=mask.T,
            waveimg=waveimg.T,
            skyimg=skyimg.T,
            thismask=thismask.T,
            spat_img=spat_img.T,
            nspec=nspec,
            nspat=nspat,
        )

    @staticmethod
    def _extract_pypeit_optimal(
        spec_model: SpecModel, sobj, pypeit_local_sky: Optional[dict] = None
    ) -> dict:
        """
        Fill a PypeIt SpecObj using PypeIt's boxcar and optimal extraction.
        """
        from pypeit.core import extract

        inputs = SpecData._build_pypeit_extraction_inputs(spec_model)
        nspec = inputs["nspec"]
        nspat = inputs["nspat"]

        trace_center = float(getattr(spec_model, "mask_offset", 0.0))
        trace_spat = np.interp(
            trace_center,
            np.asarray(spec_model.spat, dtype=float),
            np.arange(nspat, dtype=float),
        )
        trace = np.full(nspec, trace_spat, dtype=float)
        box_radius = 0.5 * (
            spec_model.spat_edges["mask"][-1] - spec_model.spat_edges["mask"][0]
        ) / spec_model.pixel_scale

        sobj.trace_spec = np.arange(nspec, dtype=int)
        sobj.TRACE_SPAT = trace
        sobj.SPAT_PIXPOS = float(trace_spat)
        sobj.SPAT_FRACPOS = float(trace_spat / max(nspat - 1, 1))
        sobj.BOX_RADIUS = float(box_radius)
        sobj.FWHM = spec_model.spat_resln / spec_model.pixel_scale
        sobj.SPAT_FWHM = spec_model.spat_resln / spec_model.pixel_scale

        extract.extract_boxcar(
            inputs["imgminsky"],
            inputs["ivar"],
            inputs["mask"],
            inputs["waveimg"],
            inputs["skyimg"],
            sobj,
        )

        oprof, trace_new, fwhmfit, _ = extract.fit_profile(
            image=inputs["imgminsky"],
            ivar=inputs["ivar"],
            waveimg=inputs["waveimg"],
            thismask=inputs["thismask"],
            spat_img=inputs["spat_img"],
            trace_in=trace,
            wave=sobj.BOX_WAVE,
            flux=sobj.BOX_COUNTS,
            fluxivar=sobj.BOX_COUNTS_IVAR,
            inmask=inputs["mask"],
            thisfwhm=max(float(spec_model.spat_resln / spec_model.pixel_scale), 1.0),
            obj_string=sobj.NAME,
            show_profile=False,
        )

        fwhmfit_finite = fwhmfit[np.isfinite(fwhmfit)]
        trace_shift = trace_new - trace
        trace_shift_finite = trace_shift[np.isfinite(trace_shift)]
        if fwhmfit_finite.size > 0:
            msgs.info(
                "PypeIt optimal extraction FWHM: "
                f"initial={spec_model.spat_resln / spec_model.pixel_scale:.2f} pix, "
                f"median fit={np.nanmedian(fwhmfit_finite):.2f} pix, "
                f"range={np.nanmin(fwhmfit_finite):.2f}-{np.nanmax(fwhmfit_finite):.2f} pix"
            )
        if trace_shift_finite.size > 0:
            msgs.info(
                "PypeIt optimal extraction trace shift: "
                f"median={np.nanmedian(trace_shift_finite):.2f} pix, "
                f"range={np.nanmin(trace_shift_finite):.2f}-{np.nanmax(trace_shift_finite):.2f} pix"
            )

        sobj.TRACE_SPAT = trace_new
        sobj.SPAT_PIXPOS = float(np.nanmedian(trace_new))
        sobj.SPAT_FRACPOS = float(sobj.SPAT_PIXPOS / max(nspat - 1, 1))
        sobj.FWHMFIT = fwhmfit

        local_sky_info = SpecData._default_pypeit_local_sky(pypeit_local_sky)
        if local_sky_info["enabled"]:
            local_sky_resid, local_sky_mask = SpecData._fit_pypeit_local_sky(
                inputs=inputs,
                oprof=oprof,
                trace=trace_new,
                box_radius=box_radius,
                **local_sky_info,
            )
            if np.any(local_sky_mask):
                inputs["imgminsky"] = inputs["imgminsky"] - local_sky_resid
                inputs["skyimg"] = inputs["skyimg"] + local_sky_resid
                local_sky_info["applied"] = True
                msgs.info(
                    "Applied PypeIt local sky residual model: "
                    f"median={np.nanmedian(local_sky_resid[local_sky_mask]):.3e}, "
                    f"std={np.nanstd(local_sky_resid[local_sky_mask]):.3e}"
                )
                extract.extract_boxcar(
                    inputs["imgminsky"],
                    inputs["ivar"],
                    inputs["mask"],
                    inputs["waveimg"],
                    inputs["skyimg"],
                    sobj,
                )
            else:
                local_sky_info["applied"] = False
                msgs.warning("PypeIt local sky residual model was not applied.")
        else:
            local_sky_info["applied"] = False

        extract.extract_optimal(
            imgminsky=inputs["imgminsky"],
            ivar=inputs["ivar"],
            mask=inputs["mask"],
            waveimg=inputs["waveimg"],
            skyimg=inputs["skyimg"],
            thismask=inputs["thismask"],
            oprof=oprof,
            spec=sobj,
        )

        return local_sky_info

    @staticmethod
    def _default_pypeit_local_sky(pypeit_local_sky: Optional[dict] = None) -> dict:
        """
        Normalize PypeIt local-sky residual fitting options.
        """
        cfg = {
            "enabled": True,
            "bsp": 0.6,
            "sigrej": 3.5,
            "npoly": 1,
            "bkpts_optimal": True,
        }
        if pypeit_local_sky is not None:
            cfg.update(pypeit_local_sky)
        cfg["enabled"] = str(cfg["enabled"]).lower() == "true" if isinstance(cfg["enabled"], str) else bool(cfg["enabled"])
        cfg["bkpts_optimal"] = str(cfg["bkpts_optimal"]).lower() == "true" if isinstance(cfg["bkpts_optimal"], str) else bool(cfg["bkpts_optimal"])
        cfg["bsp"] = float(cfg["bsp"])
        cfg["sigrej"] = float(cfg["sigrej"])
        cfg["npoly"] = int(cfg["npoly"])
        return cfg

    @staticmethod
    def _fit_pypeit_local_sky(
        inputs: dict,
        oprof: np.ndarray,
        trace: np.ndarray,
        box_radius: float,
        enabled: bool = True,
        bsp: float = 0.6,
        sigrej: float = 3.5,
        npoly: int = 1,
        bkpts_optimal: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit residual skyline/background structure on the HostSub rectified grid.
        """
        from pypeit.core import skysub

        nspec = inputs["nspec"]
        nspat = inputs["nspat"]
        piximg = np.tile(np.arange(nspec, dtype=float)[:, None], (1, nspat))
        fitmask = np.asarray(inputs["mask"] & inputs["thismask"], dtype=bool)
        local_sky = np.zeros_like(inputs["imgminsky"], dtype=float)
        local_sky_mask = np.zeros_like(fitmask, dtype=bool)
        if not enabled or not np.any(fitmask):
            return local_sky, local_sky_mask

        try:
            evalmask = fitmask.copy()
            source_aperture = np.abs(inputs["spat_img"] - trace[:, None]) <= box_radius
            fitmask = fitmask & ~source_aperture
            if not np.any(fitmask):
                return local_sky, local_sky_mask
            bsp_pix = max(float(bsp), 2.0)
            fullbkpt = skysub.optimal_bkpts(
                bkpts_optimal,
                bsp_pix,
                piximg,
                fitmask,
                skyimage=inputs["imgminsky"],
            )
            isub = np.where(evalmask.ravel())[0]
            fit_ivar = np.asarray(inputs["ivar"], dtype=float).copy()
            fit_ivar[~fitmask] = 0.0
            oprof_sky = np.asarray(oprof, dtype=float).copy()
            oprof_sky[source_aperture] = 0.0
            oprof_flat = oprof_sky.reshape(nspec * nspat, 1)
            sky_flat, _, gpm = skysub.skyoptimal(
                piximg.ravel()[isub],
                np.asarray(inputs["imgminsky"], dtype=float).ravel()[isub],
                fit_ivar.ravel()[isub],
                oprof_flat[isub, :],
                spatial_img=np.asarray(inputs["spat_img"], dtype=float).ravel()[isub],
                fullbkpt=fullbkpt,
                sigrej=sigrej,
                npoly=npoly,
            )
            if bkpts_optimal and not np.any(np.isfinite(sky_flat) & (sky_flat != 0)):
                fullbkpt = skysub.optimal_bkpts(
                    False,
                    bsp_pix,
                    piximg,
                    fitmask,
                    skyimage=inputs["imgminsky"],
                )
                sky_flat, _, gpm = skysub.skyoptimal(
                    piximg.ravel()[isub],
                    np.asarray(inputs["imgminsky"], dtype=float).ravel()[isub],
                    fit_ivar.ravel()[isub],
                    oprof_flat[isub, :],
                    spatial_img=np.asarray(inputs["spat_img"], dtype=float).ravel()[isub],
                    fullbkpt=fullbkpt,
                    sigrej=sigrej,
                    npoly=npoly,
                )
        except Exception as err:
            msgs.warning(f"PypeIt local sky residual fitting failed: {err}")
            return local_sky, local_sky_mask

        sky_flat = np.asarray(np.ma.filled(sky_flat, 0.0), dtype=float)
        gpm = np.asarray(np.ma.filled(gpm, False), dtype=bool)
        if not np.any(np.isfinite(sky_flat) & (sky_flat != 0)):
            sky_pixels = np.where(fitmask, inputs["imgminsky"], np.nan)
            resid_1d = np.nanmedian(sky_pixels, axis=1)
            good_resid = np.isfinite(resid_1d)
            if not np.any(good_resid & (resid_1d != 0)):
                return local_sky, local_sky_mask
            if not np.all(good_resid):
                resid_1d = np.interp(
                    np.arange(nspec, dtype=float),
                    np.where(good_resid)[0].astype(float),
                    resid_1d[good_resid],
                )
            local_sky = np.tile(resid_1d[:, None], (1, nspat))
            local_sky_mask = evalmask & np.isfinite(local_sky)
            return local_sky, local_sky_mask
        sub_idx = np.unravel_index(isub, local_sky.shape)
        local_sky[sub_idx] = sky_flat
        local_sky_mask[sub_idx] = gpm if np.any(gpm) else np.isfinite(sky_flat)
        return local_sky, local_sky_mask

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
            msgs.info(
                f"Spectral resolution specified in the config file: setting it to {self.spec_resln:.2f} Ang."
            )
        if spat_resln_cfg is not None:
            self.spat_resln = spat_resln_cfg
            msgs.info(
                f"Spatial resolution specified in the config file: setting it to {self.spat_resln:.2f} arcsec."
            )

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

    def update_pypeit_skymodel(
        self,
        spec_model: SpecModel,
        spec2d_file: str,
        preproc_file: Optional[str] = None,
        rect_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        """
        Update the sky model and the mask in the PypeIt spec2d file.
        """
        import os
        from pypeit import spec2dobj

        # https://pypeit.readthedocs.io/en/1.17.0/out_masks.html
        BIT_CR = 2**1  # Cosmic rays
        BIT_OFFSLIT = 2**4  # Off-slit pixels

        preproc_file = preproc_file or spec2d_file.replace(".fits", "_preproc.fits")
        rect_file = rect_file or spec2d_file.replace(".fits", "_rect.fits")
        output_file = output_file or spec2d_file.replace(".fits", "_hostsub.fits")

        if not (os.path.exists(spec2d_file) | os.path.exists(rect_file)):
            raise FileNotFoundError("Spec2D file or rectified file not found.")

        # The new sky model will be composed of
        ###### (i) the preliminary global sky (average across the slit) removed before the rectification
        ###### (ii) the global sky (average of the sky region) subtracted after the rectification
        ###### (iii) the local sky modeled with HostSub

        # Preliminary sky line removal - reduce the noise introduced in the rectification
        hdul_preproc = fits.open(preproc_file)
        waveimg = jnp.array(hdul_preproc["WAVEIMG"].data, dtype=jnp.float32)
        dist = jnp.array(hdul_preproc["DIST"].data - self.sky_offset, dtype=jnp.float32)
        global_sky_pre = jnp.array(hdul_preproc["GLOBALSKY"].data, dtype=jnp.float32)
        det = hdul_preproc[0].header["DET"]
        hdul_preproc.close()

        sci2d = spec2dobj.Spec2DObj.from_file(spec2d_file, detname=det)

        # Mask the regions outside the wavelength/dist range
        offslit = np.asarray(
            np.where(
                (waveimg < spec_model.spec.item(0))
                | (waveimg > spec_model.spec.item(-1))
                | (dist < spec_model.spat_edges["host"][0])
                | (dist > spec_model.spat_edges["host"][1]),
                BIT_OFFSLIT,
                0,
            ),
            dtype=np.int16,
        )
        sci2d.bpmmask.mask = sci2d.bpmmask.mask | offslit.T

        # Update the bmpmask to include cosmic rays
        hdul_rect = fits.open(rect_file)
        if "CR_MASK" in hdul_rect:
            mask_cr = jnp.argwhere(
                ~jnp.array(hdul_rect["CR_MASK"].data, dtype=jnp.bool)
            )
            wave_rect = jnp.array(hdul_rect["SPEC"].data, dtype=jnp.float32)
            dist_rect = jnp.array(hdul_rect["SPAT"].data, dtype=jnp.float32)

            # The mask of the pixels with cosmic rays
            coords_rect = jnp.meshgrid(dist_rect, wave_rect, indexing="ij")

            # First create the interpolator object for the cosmic ray mask
            interpolator = Interp2D_Scipy(method="nearest")

            # Create the coordinates for the rectified grid
            # coords_rect is already [wave_rect, dist_rect] from meshgrid
            points = jnp.stack(coords_rect, axis=-1)

            # Create the mask values (2 for cosmic rays, 0 for non-cosmic rays)
            mask_values = jnp.zeros(points.shape[:-1], dtype=jnp.int32)
            mask_values = mask_values.at[mask_cr[:, 0], mask_cr[:, 1]].set(BIT_CR)

            # Fit the interpolator with the rectified grid points and mask values
            interpolator.fit(points, mask_values)

            # Create query points from the non-uniform grid
            query_points = jnp.stack([dist, waveimg], axis=-1)
            # Ensure the query points are finite
            query_points = jnp.where(np.isfinite(query_points), query_points, 0)

            # Interpolate the mask onto the non-uniform grid
            mapped_cr_mask = np.asarray(
                interpolator.predict(query_points.reshape(-1, 2)).reshape(
                    waveimg.shape
                ),
                dtype=np.int16,
            )
            # Update the mask in the PypeIt spec2d file
            sci2d.bpmmask.mask = sci2d.bpmmask.mask | mapped_cr_mask.T
        hdul_rect.close()

        # The global sky subtracted after the rectification
        assert spec_model.f_sky_1d.y is not None, (
            "The global sky model is not available."
        )
        global_sky_post = Interp1D_Grid(
            points=spec_model.f_sky_1d.X.ravel(),
            values=spec_model.f_sky_1d.y,
            method="cubic",
        )(waveimg)

        # The local sky
        x = np.stack([dist.ravel(), waveimg.ravel()], axis=-1).reshape(-1, 2)
        spat_mask = (dist.ravel() >= spec_model.spat_edges["host"][0]) & (
            dist.ravel() <= spec_model.spat_edges["host"][-1]
        )
        spec_mask = (waveimg.ravel() >= spec_model.spec[0]) & (
            waveimg.ravel() <= spec_model.spec[-1]
        )
        local_mask = spat_mask & spec_mask

        x_mask = jnp.asarray(x[local_mask], dtype=jnp.float32)
        _, _, sky_pred = spec_model._get_pred(
            spec_model._gp_1d, spec_model._gp_2d, x_mask, return_var=False
        )

        sky_local = np.zeros_like(dist.ravel())
        sky_local[local_mask] = np.asarray(sky_pred)
        sky_local = sky_local.reshape(dist.shape)

        sky_model = np.array(global_sky_pre + global_sky_post + sky_local)
        self._plot_writeback_diagnostic(
            sciimg=np.array(sci2d.sciimg.T),
            sky_model=sky_model,
            sky_local=np.array(sky_local),
            dist=np.array(dist),
            waveimg=np.array(waveimg),
            spec_model=spec_model,
            save=os.path.join(
                "QA",
                os.path.basename(spec2d_file)
                .replace(".fits", "")
                .replace("spec2d_", "")
                + "_writeback_native.pdf",
            ),
        )

        # assert sci2d.sciimg.shape == sky_model.T.shape
        sci2d.skymodel = sky_model.T

        all_spec2d = spec2dobj.AllSpec2DObj()
        all_spec2d[det] = sci2d

        hdul = fits.open(spec2d_file)
        pri_hdr = hdul[0].header
        hdul.close()

        all_spec2d.write_to_fits(output_file, pri_hdr=pri_hdr)

    @staticmethod
    def _plot_writeback_diagnostic(
        sciimg: ArrayLike,
        sky_model: ArrayLike,
        sky_local: ArrayLike,
        dist: ArrayLike,
        waveimg: ArrayLike,
        spec_model: SpecModel,
        save: str,
    ) -> None:
        """
        Plot native-grid diagnostics for the PypeIt skymodel writeback.
        """
        os.makedirs(os.path.dirname(save), exist_ok=True)

        sciimg = np.asarray(sciimg)
        sky_model = np.asarray(sky_model)
        sky_local = np.asarray(sky_local)
        dist = np.asarray(dist)
        waveimg = np.asarray(waveimg)
        residual = sciimg - sky_model

        spec_mask = (
            np.isfinite(waveimg)
            & (waveimg >= spec_model.spec[0])
            & (waveimg <= spec_model.spec[-1])
        )
        host_mask = (
            np.isfinite(dist)
            & (dist >= spec_model.spat_edges["host"][0])
            & (dist <= spec_model.spat_edges["host"][-1])
            & spec_mask
        )

        def finite_scale(values: ArrayLike) -> float:
            values = np.asarray(values)
            values = np.abs(values[np.isfinite(values)])
            if values.size == 0:
                return 1.0
            scale = np.nanpercentile(values, 95)
            return scale if np.isfinite(scale) and scale > 0 else 1.0

        resid_scale = finite_scale(residual[host_mask])
        if resid_scale == 1.0:
            resid_scale = finite_scale(residual)
        local_scale = finite_scale(sky_local[host_mask])
        if local_scale == 1.0:
            local_scale = finite_scale(sky_local)

        bins = np.linspace(
            spec_model.spat_edges["host"][0], spec_model.spat_edges["host"][-1], 80
        )
        bin_center = 0.5 * (bins[1:] + bins[:-1])
        resid_profile = np.full(bin_center.shape, np.nan)
        model_profile = np.full(bin_center.shape, np.nan)
        for i in range(len(bin_center)):
            bin_mask = host_mask & (dist >= bins[i]) & (dist < bins[i + 1])
            if np.any(bin_mask):
                resid_profile[i] = np.nanmedian(residual[bin_mask])
                model_profile[i] = np.nanmedian(sky_local[bin_mask])

        _, ax = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        image_kwargs = dict(origin="lower", aspect="auto", cmap="RdBu_r")
        ax[0].imshow(residual, vmin=-resid_scale, vmax=resid_scale, **image_kwargs)
        ax[0].set_title(r"Native residual: SCIIMG $-$ proposed SKYMODEL")
        ax[0].set_ylabel("Spatial pixel")

        ax[1].imshow(sky_local, vmin=-local_scale, vmax=local_scale, **image_kwargs)
        ax[1].set_title("Native local HostSub model")
        ax[1].set_ylabel("Spatial pixel")

        ax[2].plot(bin_center, resid_profile, label="median residual")
        ax[2].plot(bin_center, model_profile, label="median local model", alpha=0.8)
        ax[2].axvline(0, color="0.5", ls=":", label="trace")
        ax[2].axvspan(
            spec_model.spat_edges["mask"][0],
            spec_model.spat_edges["mask"][-1],
            color="0.8",
            alpha=0.4,
            label="source mask",
        )
        ax[2].axhline(0, color="k", ls="--", lw=1)
        ax[2].set_xlabel("Distance from trace [arcsec]")
        ax[2].set_ylabel("Median counts")
        ax[2].legend(loc="best")

        for ax_ in ax[:2]:
            ax_.set_xlabel("Spectral pixel")

        plt.savefig(save)
        plt.close()
        msgs.info(f"Saved writeback diagnostic plot to {save}")

    def _rectify(
        self,
        points: ArrayLike,
        f_values: tuple[Array, Array],
        batch_size: int = 1024,
        interp_method: str = "rbf",
    ) -> tuple[Array, Array]:
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

        if ~jnp.all(jnp.isfinite(points)):
            raise ValueError("Some points are NaN.")

        points = jnp.asarray(points, dtype=jnp.float32)

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
        spec_batch_idx = jnp.array_split(
            jnp.arange(len(self.spec_rect)), len(self.spec_rect) // batch_size + 1
        )

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
            spec_min = idx_list.item(0)
            spec_max = idx_list.item(-1) + 1
            # Initialize the interpolators (with padding)
            fit_min = max(spec_min - 1, 0)
            fit_max = min(spec_max + 1, points.shape[1])
            interp2d.fit(
                points=points[:, fit_min:fit_max], values=flux[:, fit_min:fit_max]
            )
            interp2d_ivar.fit(
                points=points[:, fit_min:fit_max], values=ivar[:, fit_min:fit_max]
            )
            # Query points
            query_points = jnp.stack(
                [
                    spat_pix_rect[:, spec_min:spec_max].ravel(),
                    spec_pix_rect[:, spec_min:spec_max].ravel(),
                ],
                axis=-1,
            )
            flux_rect[:, spec_min:spec_max] = interp2d.predict(
                query_points=query_points
            ).reshape(flux_rect[:, spec_min:spec_max].shape)
            flux_ivar_rect[:, spec_min:spec_max] = interp2d_ivar.predict(
                query_points=query_points
            ).reshape(flux_ivar_rect[:, spec_min:spec_max].shape)

        return jnp.asarray(flux_rect), jnp.asarray(flux_ivar_rect)

    @show_and_save
    def _get_offset(
        self,
        points: Array,
        flux: Array,
        host_prior: Callable,
        mask_wid: float = 2.0,
    ) -> float:
        """
        Center the trace of the science object.
        """

        def binned_mean_with_clipping(
            points: Array,
            values: Array,
            bins: int,
            bin_range: tuple[float, float],
            **kwargs,
        ):
            from astropy.stats import sigma_clip, mad_std

            points = jnp.asarray(points, dtype=jnp.float32)
            values = jnp.asarray(values, dtype=jnp.float32)

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
                assert isinstance(y_clipped, np.ma.MaskedArray), (
                    "Clipped values are not a masked array."
                )
                # Compute the statistic (e.g., mean) of the clipped values (requires at least 80% valid data)
                if (~y_clipped.mask).sum() > 0.80 * len(y_clipped):
                    obs[i] = np.nanmean(y_clipped[~y_clipped.mask])

            obs[obs <= 0] = np.nan

            return obs

        slit_len_max = min(
            min(-self.spat_rect.item(0), self.spat_rect.item(-1)) * 2, 30
        )
        spat = self.spat_rect[np.abs(self.spat_rect) <= slit_len_max / 2]

        obs = binned_mean_with_clipping(
            points[..., 0],
            flux,
            bins=len(spat),
            bin_range=(
                spat.item(0) - self.pixel_scale / 2,
                spat.item(-1) + self.pixel_scale / 2,
            ),
            sigma=5,
        )

        sci_obj_mask = (jnp.abs(spat) >= mask_wid) & jnp.isfinite(obs)

        # Profile from the prior - flux evaluated at the weighted-mean wavelength
        wv_mean = (
            jnp.nanmean(points[..., 1] * flux) / jnp.nanmean(flux) * jnp.ones_like(spat)
        )

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

        _, ax = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
        ax[0].plot(offset_list, ccf)
        ax[0].set_xlabel(r"$\mathrm{SCI - STD\ [arcsec]}$")
        ax[0].set_ylabel(r"$\mathrm{Correlation Coefficient}$")
        ax[0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0].xaxis.set_minor_locator(plt.MultipleLocator(0.02))
        ax[0].axvline(offset_opt, color="k", linestyle="--")

        profile_prior = host_prior(
            jnp.stack(
                [
                    spat - offset_opt,
                    jnp.nanmean(points[:, :, 1] * flux)
                    / jnp.nanmean(flux)
                    * jnp.ones_like(spat),
                ],
                axis=-1,
            )
        )[0]

        def normalize_profile(values: ArrayLike) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            finite = np.isfinite(values)
            norm = np.full(values.shape, np.nan, dtype=float)
            if not np.any(finite):
                return norm
            vmin = np.nanmin(values[finite])
            vmax = np.nanmax(values[finite])
            if not np.isfinite(vmax - vmin) or vmax == vmin:
                return norm
            norm[finite] = (values[finite] - vmin) / (vmax - vmin)
            return norm

        corrected_spat = np.asarray(spat - offset_opt, dtype=float)
        obs_norm = normalize_profile(obs)
        prior_norm = normalize_profile(profile_prior)
        residual = obs_norm - prior_norm
        finite_resid = (
            np.isfinite(corrected_spat)
            & np.isfinite(residual)
            & (np.abs(corrected_spat) <= 1.0)
        )
        suggested_mask_offset = (
            corrected_spat[finite_resid][np.nanargmax(residual[finite_resid])]
            if np.any(finite_resid)
            else np.nan
        )

        ax[1].scatter(corrected_spat, obs_norm, label="obs")
        ax[1].scatter(
            corrected_spat,
            prior_norm,
            label="prior",
        )
        ax[1].axvline(0, color="k", linestyle="--", label="corrected origin")
        ax[1].axvline(
            -offset_opt,
            color="0.5",
            linestyle=":",
            label="PypeIt trace",
        )
        if np.isfinite(suggested_mask_offset):
            ax[1].axvline(
                suggested_mask_offset,
                color="C3",
                linestyle="-.",
                label="suggested mask_offset",
            )
        ax[1].set_xlabel(r"$\mathrm{Corrected\ spat\ [arcsec]}$")
        ax[1].set_ylabel(r"$\mathrm{Normalized\ Counts}$")
        ax[1].xaxis.set_major_locator(plt.MultipleLocator(5))
        ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax[1].legend()

        ax[2].scatter(corrected_spat, residual, color="C3", label="obs - prior")
        ax[2].axhline(0, color="k", linestyle="--", linewidth=1)
        ax[2].axvline(0, color="k", linestyle="--", label="corrected origin")
        ax[2].axvline(
            -offset_opt,
            color="0.5",
            linestyle=":",
            label="PypeIt trace",
        )
        if np.isfinite(suggested_mask_offset):
            ax[2].axvline(
                suggested_mask_offset,
                color="C3",
                linestyle="-.",
                label=f"suggested mask_offset = {suggested_mask_offset:.2f} arcsec",
            )
        ax[2].set_xlabel(r"$\mathrm{Corrected\ spat\ [arcsec]}$")
        ax[2].set_ylabel(r"$\mathrm{Residual}$")
        ax[2].xaxis.set_major_locator(plt.MultipleLocator(5))
        ax[2].xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax[2].legend()

        if np.isfinite(suggested_mask_offset):
            msgs.info(
                f"Suggested mask_offset from trace-alignment residual: {suggested_mask_offset:.2f} arcsec"
            )

        return offset_opt
