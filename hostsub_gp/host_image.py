# hostsub_gp/host_image.py

__all__ = ["PS1Image", "SDSSImage"]

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std

import subprocess
import os

from ._utils import msgs
from ._utils._astronometry import query_astrometry_net_wcs

from numpy.typing import NDArray


class ImageProduct:
    """
    Class to save main data products after resampling an archival image onto the slit
    """

    def __init__(
        self,
        center_ra: float,
        center_dec: float,
        slit_len: float,
        slit_wid: float,
        position_angle: float,
        img: NDArray,
        header: fits.Header,
        flt: str,
        wv_eff: float,
    ):
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        from reproject import reproject_adaptive

        self.flt = flt
        self.wv_eff = wv_eff

        # Load FITS image and WCS info
        wcs = WCS(header)

        # Step 1. Get the position angle and pixel scale of the image cutout
        if wcs.wcs.has_cd():
            pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600  # arcsec/pixel
        else:
            pixel_scale = wcs.wcs.cdelt[0] * 3600  # arcsec/pixel

        # Step 2. Resample the image to the slit with a given position angle

        slit_len_pix = slit_len / pixel_scale  # Slit length in pixels
        slit_wid_pix = slit_wid / pixel_scale  # Slit width in pixels
        shape = (
            int(np.ceil(slit_len_pix / 2)) * 2,
            int(np.ceil(slit_wid_pix / 2))
            * 2
            * 10,  # 10 times the width - room for blurring
        )

        # Define the target wcs
        wcs_target = WCS(naxis=2)
        wcs_target.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs_target.wcs.crval = [center_ra, center_dec]
        wcs_target.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]

        # Include rotation in CD matric
        # Add 90 degrees to the position angle to align slit with the x-axis - w.r.t. the west
        theta = np.deg2rad(position_angle) + np.pi
        wcs_target.wcs.cd = (
            np.array(
                [
                    [
                        -pixel_scale * np.cos(theta),
                        -pixel_scale * np.sin(theta),
                    ],
                    [pixel_scale * np.sin(theta), -pixel_scale * np.cos(theta)],
                ]
            )
            / 3600
        )

        # Set additional required WCS parameters
        wcs_target.wcs.radesys = "ICRS"
        wcs_target.wcs.equinox = 2000.0

        # Reproject the image to the slit-aligned WCS
        data_reproj, _ = reproject_adaptive(
            (img, wcs),
            wcs_target,
            shape_out=shape,
        )

        self.img = data_reproj
        # import matplotlib.pyplot as plt
        # err = mad_std(
        #     self.img[
        #         np.isfinite(self.img) & (self.img < np.nanpercentile(self.img, 50))
        #     ]
        # ) / np.sqrt(1 - 2 / np.pi)
        bins, bin_edges = np.histogram(
            self.img[np.isfinite(self.img)],
            bins=200,
        )
        bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        mode = bin_center[np.argmax(bins)]
        self.err = mad_std(
            self.img[
                np.isfinite(self.img)
                & (self.img < mode)
            ]
        ) #/ np.sqrt(1 - 2 / np.pi)
        self.pixel_scale = pixel_scale
        self.shape = shape
        self.slit_len_pix = slit_len_pix
        self.slit_wid_pix = slit_wid_pix
        self.wcs = wcs_target

        # Obtain the pixel coordinates along the slit
        self.spat_slit = (
            np.arange(self.shape[0]) + 1 - self.wcs.wcs.crpix[1]
        ) * self.pixel_scale

        # Obtain the pixel coordinates perpendicular to the slit
        self.spat_slit_wid = (
            np.arange(self.shape[1]) + 1 - self.wcs.wcs.crpix[0]
        ) * self.pixel_scale


class ArchivalImage:
    """
    Base class for loading images from archival services
    """

    def __init__(self, ra: float, dec: float, filters: str, path: str):
        self.ra = ra
        self.dec = dec
        self.filters = filters
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.wv_eff_dict: dict[str, float] = dict()

    def check_exists(self) -> bool:
        """
        Check if images already exist in the path
        """

        for flt in self.filters:
            file = f"{self.path}/{flt}.fits"
            if not os.path.exists(file):
                return False
        return True

    def load(self) -> tuple[list, list]:
        """
        Load images from the path
        """
        data_list = []
        header_list = []
        loaded_filters = []

        for flt in self.filters:
            file = f"{self.path}/{flt}.fits"
            try:
                # Check if the file exists and is not empty
                data = fits.getdata(file)
                if np.all(np.isnan(data)) or np.all(data == 0):
                    msgs.warning(f"No data for filter {flt}.")
                    self.filters = "".join([x for x in self.filters if x != flt])
                    continue
            except FileNotFoundError:
                msgs.warning(f"File {file} not found.")
                continue
            file_wcs = f"{self.path}/{flt}_wcs.fits"
            file_orig = f"{self.path}/{flt}.fits"
            
            if os.path.exists(file_wcs):
                file_to_load = file_wcs
                msgs.info(f"Loading WCS calibrated file: {file_wcs}")
            elif os.path.exists(file_orig):
                file_to_load = file_orig
                msgs.warning(f"Loading original file (WCS not calibrated by Astrometry.net): {file_orig}")
            else:
                msgs.warning(f"No file found for filter {flt}.")
                continue

            # Load the file
            try:
                data = fits.getdata(file_to_load)
                header = fits.getheader(file_to_load)
                
                # Store successful load
                data_list.append(data)
                header_list.append(header)
                loaded_filters.append(flt)
                
            except Exception as e:
                msgs.warning(f"Failed to load {file_to_load}: {e}")
                continue

        # Update filters with successfully loaded ones
        self.filters = loaded_filters
        return data_list, header_list

    def get_cutout(self, overwrite: bool = False):
        """
        Get cutout images from the archival service
        """

        # Check if images already exist
        if not overwrite and self.check_exists():
            msgs.info("Images already exist.")
        else:
            self.download(overwrite=overwrite)

        # WCS calibration with Astrometry.net
        query_astrometry_net_wcs(self.path, overwrite=overwrite)

    def download(self, overwrite: bool = True):
        """
        Download images from the archival service
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PS1Image(ArchivalImage):
    """
    Class to load images from the PS1 Image Cutout Service
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        filters: str = "grizy",
        path: str = "./ps1_cutout",
        size: int = 2400,
    ):
        super().__init__(ra, dec, filters, path)
        self.size = size
        self.wv_eff_dict = dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60)

    def download(self, overwrite: bool = False):
        """
        Download images from the PS1 Image Cutout Service
        """
        if self.dec < -30:
            msgs.warning("PS1 images are not available for declinations < -30 degrees.")
            return
        # Check if images already exist
        if not overwrite and self.check_exists():
            msgs.info("PS1 images already exist.")
            return

        fitsurl = self._geturl()
        if len(fitsurl) == 0:
            msgs.warning("No images found in the PS1 database.")
            return
        os.makedirs(self.path, exist_ok=True)
        for k, flt in enumerate(self.filters):
            fitspath = f"{self.path}/{flt}.fits"
            if os.path.exists(fitspath) and not overwrite:
                continue
            subprocess.run(["wget", fitsurl[k], "-O", fitspath])

        query_astrometry_net_wcs(self.path, overwrite=overwrite)

    def _getimages(self):
        """Query ps1filenames.py service to get a list of images"""

        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = f"{service}?ra={self.ra}&dec={self.dec}&filters={self.filters}"
        table = Table.read(url, format="ascii")
        return table

    def _geturl(self):
        """Get URL for images in the table"""

        table = self._getimages()
        url = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={self.ra}&dec={self.dec}&size={self.size}&format=fits"
        )
        # sort filters from red to blue
        flist = ["grizy".find(x) for x in table["filter"]]
        table = table[np.argsort(flist)]
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
            url.append(urlbase + filename)
        return url


class SDSSImage(ArchivalImage):
    """
    Class to load images from astroquery.sdss.SDSS
    """

    def __init__(
        self, ra: float, dec: float, filters: str = "ugriz", path: str = "./sdss_cutout"
    ):
        super().__init__(ra, dec, filters, path)
        self.wv_eff_dict = dict(u=3556.52, g=4702.50, r=6175.58, i=7489.98, z=8946.71)

    def download(self, overwrite: bool = False) -> None:
        """
        Download images from astroquery.sdss.SDSS
        """
        # Check if images already exist
        if not overwrite and self.check_exists():
            msgs.info("SDSS images already exist.")
            return

        from astroquery.sdss import SDSS

        # Define target coordinates (RA, Dec)
        coord = SkyCoord(ra=self.ra, dec=self.dec, unit="deg")

        # Query the SDSS image cutout
        result = SDSS.query_region(coord, radius=30 * u.arcsec)
        if result is None or len(result) == 0:
            msgs.warning("No images found in the SDSS database.")
            return

        # Get the images
        os.makedirs(self.path, exist_ok=True)
        for flt in self.filters:
            img = SDSS.get_images(matches=result, band=flt)
            if img is None:
                continue
            elif len(img) == 0:
                continue
            fitspath = f"{self.path}/{flt}.fits"
            if os.path.exists(fitspath) and not overwrite:
                continue
            img[0][0].writeto(fitspath)


class LSImage(ArchivalImage):
    """
    Class to load images from the DESI Legacy Imaging Surveys
    """

    def __init__(
        self, ra: float, dec: float, filters: str = "griz", path: str = "./ls_cutout"
    ):
        super().__init__(ra, dec, filters, path)
        # https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera
        self.wv_eff_dict = dict(g=4730, r=6420, i=7840, z=9260)

        self.pixel_scale = 0.25
        self.size = 10  # arcmin

    def download(self, overwrite: bool = False):
        """
        Download images from the DESI Legacy Imaging Surveys
        """
        # Check if images already exist
        if not overwrite and self.check_exists():
            msgs.info("LS images already exist.")
            return

        url = "https://www.legacysurvey.org/viewer/fits-cutout?ra={ra:.4f}&dec={dec:.4f}&pixscale={px_scale}&layer=ls-dr10&size={size:g}".format(
            ra=self.ra,
            dec=self.dec,
            px_scale=self.pixel_scale,
            size=self.size * 60 / self.pixel_scale,
        )

        status = self._download_url(url=url, outfile=f"{self.path}/dummy.fits")

        # Postprocessing
        if status:
            hdu = fits.open(f"{self.path}/dummy.fits")
            hdu_header = hdu[0].header

            for k, f in enumerate([*str(hdu_header["bands"])]):
                hdu_header["FILTER"] = f
                hdu_header["GAIN"] = 100
                data = hdu[0].data[k, :, :]

                fits.writeto(
                    f"{self.path}/{f}.fits",
                    data,
                    hdu_header,
                    output_verify="silentfix",
                    overwrite=True,
                )

                # If the filter is missing, update the filters list
                if np.all(np.isnan(data)) or np.all(data == 0):
                    msgs.warning(f"No data for filter {f}.")
                    self.filters = "".join([x for x in self.filters if x != f])
                    continue

            # Remove the dummy file
            os.remove(f"{self.path}/dummy.fits")

        else:
            msgs.warning("Failed to download LS images.")
            msgs.warning(
                "Field might not be covered by DESI LS or https://www.legacysurvey.org/viewer might be offline."
            )
            return

    def _download_url(self, url: str, outfile: str, max_attempts: int = 5) -> bool:
        """
        Download a file from a URL using wget with retry logic.
        Adapted from: https://github.com/ygordon/cutout_fetching/blob/master/cutout_fetching_lite.py
        """

        # Often encounter the following error:
        # urllib.error.HTTPError: HTTP Error 504: Gateway Time-out
        # Repeat the download attempt for up to `max_attempts` tries
        # Return True if the download was successful

        import time
        from urllib.error import HTTPError, URLError

        for attempt in range(max_attempts):
            try:
                subprocess.run(["wget", url, "-O", outfile], check=True)
                return True
            except HTTPError as e:
                print(
                    f"Failed attempt {attempt} to download {outfile} with an HTTPError: {e}"
                )
            except URLError as e:
                print(
                    f"Failed attempt {attempt} to download {outfile} with a URLError: {e}"
                )
            time.sleep(1)

        print(f"Failed to download image {outfile}")
        return False
