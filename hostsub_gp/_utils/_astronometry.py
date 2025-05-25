# hostsub_gp/_utils/_astronometry.py

import os

from astroquery.astrometry_net import AstrometryNet
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from pathlib import Path

from ._msgs import msgs


class AstrometryNetSolver:
    def __init__(self):
        """Initialize the solver with your API key"""
        self.ast = AstrometryNet()
        try:
            API_KEY = os.getenv("ASTROMETRY_NET_TOKEN")
            self.ast.api_key = API_KEY
        except Exception as e:
            raise ValueError(
                "API key not found. Please set the 'ASTROMETRY_NET_TOKEN' environment variable."
            ) from e

    def solve_field(self, fits_path: str, **settings) -> str:
        """
        Submit FITS file for plate solving and retrieve WCS solution

        Parameters
        ----------
        fits_path : str
            Path to the FITS file
        **kwargs : dict
            Additional solving parameters like:
            - scale_est : float (arcsec per pixel)
            - center_ra : float (degrees)
            - center_dec : float (degrees)
            - radius : float (degrees)
            - tweak_order : int
            - downsample_factor : int

        Returns
        -------
        str
            Path to the new FITS file with WCS solution
        """
        # Default settings
        settings["allow_commercial_use"] = "n"
        settings["allow_modifications"] = "n"
        settings["publicly_visible"] = "n"
        settings["scale_type"] = "ev"
        settings["scale_err"] = 5
        settings["scale_units"] = "arcsecperpix"

        # Check if WCS is already present
        astrometry_dict = self.extract_wcs_info(fits_path)
        if astrometry_dict is None:
            msgs.warning("No WCS information found in the FITS header.")
            msgs.warning("Attempting to solve the field with the provided parameters.")
            astrometry_dict = settings
        else:
            msgs.info("WCS information found in the FITS header.")
            astrometry_dict = {**astrometry_dict, **settings}
        try:
            msgs.info(f"Submitting {fits_path} to astrometry.net...")

            # Submit the image for solving
            wcs_header = self.ast.solve_from_image(
                fits_path, **astrometry_dict, solve_timeout=600
            )

            if wcs_header:
                print()
                msgs.info("Successfully solved field!")

                # Create new FITS file with WCS solution
                output_path = (
                    Path(fits_path).parent / f"{Path(fits_path).stem}_wcs.fits"
                )

                hdul = fits.open(fits_path)
                # Update the original FITS file with WCS solution
                hdul[0].header.update(wcs_header)
                # Save the updated file
                hdul.writeto(output_path, overwrite=True)
                hdul.close()

                msgs.info(f"WCS solution saved to: {output_path}")
                return str(output_path)
            else:
                msgs.error("Plate solving failed!")
                return ""

        except Exception as e:
            msgs.error(f"Error during plate solving: {str(e)}")
            return ""

    def extract_wcs_info(self, fits_path: str) -> dict | None:
        """
        Extract WCS information from FITS header

        Parameters
        ----------
        fits_path : str
            Path to the FITS file

        Returns
        -------
        dict
            Dictionary containing WCS parameters or None if extraction fails
        """
        try:
            header = fits.getheader(fits_path, 0)

            # Check for required basic WCS keywords
            basic_required = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"]
            if not all(keyword in header for keyword in basic_required):
                msgs.warning(
                    "Basic WCS keywords (CRVAL1, CRVAL2, CRPIX1, CRPIX2) not found."
                )
                return None

            # Create WCS object
            wcs = WCS(header)

            # Calculate image center coordinates
            naxis1 = header.get("NAXIS1", 0)
            naxis2 = header.get("NAXIS2", 0)
            if naxis1 == 0 or naxis2 == 0:
                return None

            # Get center coordinates
            center_x = naxis1 / 2
            center_y = naxis2 / 2
            center_ra, center_dec = wcs.all_pix2world(center_x, center_y, 0)

            # Get the position angle of the image cutout
            # Get the CD or PC matrix from WCS
            if wcs.wcs.has_cd():  # Check if CD matrix is present
                pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600  # arcsec/pixel
            else:  # Otherwise, use PC matrix with CDELT
                pixel_scale = wcs.wcs.cdelt[0] * 3600  # arcsec/pixel

            return {
                "center_ra": float(center_ra),  # in degrees
                "center_dec": float(center_dec),  # in degrees
                "scale_est": float(pixel_scale),  # in arcseconds per pixel
                "radius": float(
                    (naxis1**2 + naxis2**2) ** 0.5 * pixel_scale / 3600.0 / 2
                ),  # in degrees
            }

        except Exception as e:
            msgs.warning(f"WCS extraction failed: {str(e)}")
            return None


def query_astrometry_net_wcs(directory, overwrite=False, **kwargs):
    """
    Process all FITS files in a directory

    Parameters
    ----------
    directory : str
        Path to directory containing FITS files
    **kwargs : dict
        Additional solving parameters
    """
    solver = AstrometryNetSolver()
    directory = Path(directory)

    raw_list = [f for f in directory.glob("[!.*]*.fits") if "wcs" not in str(f)]
    wcs_list = [f for f in directory.glob("[!.*]*.fits") if "wcs" in str(f)]

    for fits_path in raw_list:
        # Check if the file has already been processed
        if not overwrite and any(fits_path.stem in str(wcs) for wcs in wcs_list):
            msgs.info(f"File {fits_path} already processed. Skipping.")
            continue
        msgs.info(f"Processing: {fits_path}")
        solved_path = solver.solve_field(str(fits_path), **kwargs)
        if solved_path:
            msgs.info(f"Successfully processed {fits_path}")
        else:
            msgs.error(f"Failed to process {fits_path}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process FITS files with Astrometry.net"
    )

    parser.add_argument("directory", type=str, help="Directory containing FITS files")
    parser.add_argument(
        "--scale_est", type=float, default=None, help="Estimated scale in arcsec/pixel"
    )
    parser.add_argument(
        "--center_ra", type=float, default=None, help="Center RA in degrees"
    )
    parser.add_argument(
        "--center_dec", type=float, default=None, help="Center Dec in degrees"
    )
    parser.add_argument("--radius", type=float, default=None, help="Radius in degrees")
    parser.add_argument("--tweak_order", type=int, default=None, help="Tweak order")
    parser.add_argument(
        "--downsample_factor", type=int, default=None, help="Downsample factor"
    )
    args = parser.parse_args()

    # Generate kwargs excluding None values
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != "directory"}

    # Call the function with the directory and kwargs
    query_astrometry_net_wcs(args.directory, **kwargs)
