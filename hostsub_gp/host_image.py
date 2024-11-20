# hostsub_gp/host_image.py

__all__ = ["PS1Image"]

import numpy
from astropy.table import Table
import subprocess
import os
from astropy.io import fits


class PS1Image:
    """
    Class to load images from the PS1 Image Cutout Service
    """

    def __init__(self, ra: float, dec: float, size: int = 240, filters: str = "grizy", path: str = "./ps1_cutout/"):
        self.ra = ra
        self.dec = dec
        self.size = size
        self.filters = filters
        self.path = path
        self.wv_eff_dict = dict(g=4810.16, r=6155.47, i=7503.03, z=8668.36, y=9613.60)

    def download(self, overwrite: bool = False):
        """
        Download images from the PS1 Image Cutout Service
        """
        os.makedirs(self.path, exist_ok=True)
        for k, flt in enumerate(self.filters):
            fitspath = f"{self.path}{flt}.fits"
            if os.path.exists(fitspath) and not overwrite:
                continue
            fitsurl = self._geturl()
            subprocess.run(["wget", fitsurl[k], "-O", fitspath])

    def load(self) -> tuple[list, list]:
        """
        Load images from the PS1 Image Cutout Service
        """

        files = [f"{self.path}/{flt}.fits" for flt in self.filters]

        data_list = []
        header_list = []

        for file in files:
            with fits.open(file) as hdulist:
                data_list.append(hdulist[0].data)
                header_list.append(hdulist[0].header)

        return data_list, header_list

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
