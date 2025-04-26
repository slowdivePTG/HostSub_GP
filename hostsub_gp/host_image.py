# hostsub_gp/host_image.py

__all__ = ["PS1Image", "SDSSImage"]

import numpy

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

import subprocess
import os

from ._utils import msgs


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
        Load images from the PS1 Image Cutout Service
        """

        data_list = []
        header_list = []
        filters = []

        for flt in self.filters:
            file = f"{self.path}//{flt}.fits"
            try:
                with fits.open(file) as hdulist:
                    data_list.append(hdulist[0].data)
                    header_list.append(hdulist[0].header)
                    filters.append(flt)
            except FileNotFoundError:
                msgs.warning(f"File {file} not found.")
            except Exception as e:
                raise e

        # Update filters with the files that were successfully loaded
        self.filters = filters

        return data_list, header_list

    def download(self, overwrite: bool = False):
        """
        Download images from the archival service
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PS1Image(ArchivalImage):
    """
    Class to load images from the PS1 Image Cutout Service
    """

    def __init__(self, ra: float, dec: float, filters: str = "grizy", path: str = "./ps1_cutout", size: int = 600):
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
        table = table[numpy.argsort(flist)]
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
            url.append(urlbase + filename)
        return url


class SDSSImage(ArchivalImage):
    """
    Class to load images from astroquery.sdss.SDSS
    """

    def __init__(self, ra: float, dec: float, filters: str = "ugriz", path: str = "./sdss_cutout"):
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
        for k, flt in enumerate(self.filters):
            img = SDSS.get_images(matches=result, band=flt)
            if len(img) == 0:
                continue
            fitspath = f"{self.path}/{flt}.fits"
            if os.path.exists(fitspath) and not overwrite:
                continue
            img[0][0].writeto(fitspath)


class LSImage(ArchivalImage):
    """
    Class to load images from the DESI Legacy Imaging Surveys
    """

    def __init__(self, ra: float, dec: float, filters: str = "griz", path: str = "./ls_cutout"):
        super().__init__(ra, dec, filters, path)
        # https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera
        self.wv_eff_dict = dict(g=4730, r=6420, i=7840, z=9260)

        self.pixel_scale = 0.25
        self.size = 3  # arcmin

    def download(self, overwrite: bool = False):
        """
        Download images from the DESI Legacy Imaging Surveys
        """
        # Check if images already exist
        if not overwrite and self.check_exists():
            msgs.info("LS images already exist.")
            return

        url = "https://www.legacysurvey.org/viewer/fits-cutout?ra={ra:.4f}&dec={dec:.4f}&pixscale={px_scale}&layer=ls-dr10&size={size:g}".format(
            ra=self.ra, dec=self.dec, px_scale=self.pixel_scale, size=self.size * 60 / self.pixel_scale
        )

        status = self._download_url(url=url, outfile=f"{self.path}/dummy.fits")

        # Postprocessing
        if status:
            hdu = fits.open(f"{self.path}/dummy.fits")
            hdu_header = hdu[0].header

            for k, f in enumerate([*hdu_header["bands"]]):
                hdu_header["FILTER"] = f
                hdu_header["GAIN"] = 100
                fits.writeto(
                    f"{self.path}/{f}.fits",
                    hdu[0].data[k, :, :],
                    hdu_header,
                    output_verify="silentfix",
                    overwrite=True,
                )

            # Remove the dummy file
            os.remove(f"{self.path}/dummy.fits")

        else:
            msgs.warning("Failed to download LS images.")
            msgs.warning("Field might not be covered by DESI LS or https://www.legacysurvey.org/viewer might be offline.")
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
                print(f"Failed attempt {attempt} to download {outfile} with an HTTPError")
            except URLError as e:
                print(f"Failed attempt {attempt} to download {outfile} with a URLError")
            time.sleep(1)

        print(f"Failed to download image {outfile}")
        return False
