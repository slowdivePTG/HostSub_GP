#!/usr/bin/env python
"""
This script bins a MUSE data cube spatially, merging every 2x2 pixels into 1.
The WCS is preserved (same pixel scale), and the header is updated accordingly.
The resulting file is saved at the same location with a suffix 'binned'.

The script handles different HDU extensions in a special way:
- HDU 0 (Primary): Copied as is
- HDU 1 (Flux data): Pixels are SUMMED when binned (physically combining fluxes)
- HDU 2 (Variance/Error data): Values are handled for proper error propagation
  * If variance values: directly summed
  * If standard errors: squared, summed, then square root taken
- Other HDUs: Copied as is

For proper error propagation:
- When summing N measurements, the variances (error^2) add
- The script assumes HDU 2 contains variance values, not standard errors
"""

import numpy as np
import os
from astropy.io import fits
import argparse


def bin_data_cube(data, bin_size=2, operation='mean', truncate_method='auto'):
    """
    Bin a 3D data cube along the spatial dimensions.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data cube with shape (n_wavelength, n_y, n_x)
    bin_size : int, optional
        The binning size, by default 2
    operation : str, optional
        The operation to perform on the binned pixels, either 'mean' or 'sum',
        by default 'mean'
    truncate_method : str, optional
        How to handle dimensions not divisible by bin_size:
        - 'auto': Discard pixels at the end (default)
        - 'start': Discard pixels from the beginning
        - 'end': Discard pixels from the end
        - 'center': Discard pixels from both ends equally
        
    Returns
    -------
    numpy.ndarray
        The binned data cube
    """
    n_wavelength, n_y, n_x = data.shape
    
    # Calculate how many rows/columns to keep (must be divisible by bin_size)
    keep_n_y = (n_y // bin_size) * bin_size
    keep_n_x = (n_x // bin_size) * bin_size
    
    # Calculate the new dimensions after binning
    new_n_y = keep_n_y // bin_size
    new_n_x = keep_n_x // bin_size
    
    # Calculate how many pixels to discard
    discard_y = n_y - keep_n_y
    discard_x = n_x - keep_n_x
    
    # Calculate start indices for slicing based on truncation method
    if truncate_method == 'auto' or truncate_method == 'end':
        # Discard from the end (default)
        start_y, start_x = 0, 0
    elif truncate_method == 'start':
        # Discard from the beginning
        start_y, start_x = discard_y, discard_x
    elif truncate_method == 'center':
        # Discard equally from both ends
        start_y = discard_y // 2
        start_x = discard_x // 2
    else:
        raise ValueError(f"Truncate method '{truncate_method}' not supported")
    
    # Truncate the data to dimensions divisible by bin_size
    if keep_n_y < n_y or keep_n_x < n_x:
        print(f"Original dimensions: {n_y}x{n_x}")
        print(f"Truncating to dimensions divisible by {bin_size}: {keep_n_y}x{keep_n_x}")
        print(f"Truncation method: {truncate_method}")
        print(f"Discarding {discard_y} rows and {discard_x} columns")
        
        end_y = start_y + keep_n_y
        end_x = start_x + keep_n_x
        truncated_data = data[:, start_y:end_y, start_x:end_x]
    else:
        truncated_data = data
        
    # Reshape to separate the pixels that will be binned together
    reshaped = truncated_data.reshape(n_wavelength, new_n_y, bin_size, new_n_x, bin_size)
    
    # Apply the requested operation
    if operation.lower() == 'mean':
        binned = np.nanmean(reshaped, axis=(2, 4))
    elif operation.lower() == 'sum':
        binned = np.nansum(reshaped, axis=(2, 4))
    else:
        raise ValueError(f"Operation '{operation}' not supported. Use 'mean' or 'sum'.")
    
    return binned


def update_header(header, bin_size=2, truncated_shape=None, truncate_method='auto'):
    """
    Update the header for the binned data.
    
    Parameters
    ----------
    header : astropy.io.fits.Header
        The original header
    bin_size : int, optional
        The binning size, by default 2
    truncated_shape : tuple, optional
        The shape of the truncated data (n_wavelength, n_y, n_x), if any truncation was done
    truncate_method : str, optional
        Method used for truncation ('auto', 'start', 'end', 'center')
        
    Returns
    -------
    astropy.io.fits.Header
        The updated header
    """
    # Create a copy of the header
    new_header = header.copy()
    
    # Get original dimensions from header
    orig_naxis1 = new_header.get('NAXIS1', 0)
    orig_naxis2 = new_header.get('NAXIS2', 0)
    
    # Calculate dimensions to keep (must be divisible by bin_size)
    keep_naxis1 = (orig_naxis1 // bin_size) * bin_size
    keep_naxis2 = (orig_naxis2 // bin_size) * bin_size
    
    # Calculate how many pixels to discard
    discard_naxis1 = orig_naxis1 - keep_naxis1
    discard_naxis2 = orig_naxis2 - keep_naxis2
    
    # Calculate offset for CRPIX based on truncation method
    offset_x = 0
    offset_y = 0
    
    if truncate_method == 'start':
        offset_x = discard_naxis1
        offset_y = discard_naxis2
    elif truncate_method == 'center':
        offset_x = discard_naxis1 // 2
        offset_y = discard_naxis2 // 2
    # For 'end' or 'auto', offset remains 0
    
    # Update NAXISn values to reflect the new dimensions
    if 'NAXIS1' in new_header:
        new_header['NAXIS1'] = keep_naxis1 // bin_size
    if 'NAXIS2' in new_header:
        new_header['NAXIS2'] = keep_naxis2 // bin_size
    
    # Update CRPIX values (reference pixel) to account for binning and truncation
    # Note: CRPIX is 1-indexed, so we need to adjust
    if 'CRPIX1' in new_header:
        # Adjust for truncation first, then for binning
        new_header['CRPIX1'] = (new_header['CRPIX1'] - offset_x - 0.5) / bin_size + 0.5
    if 'CRPIX2' in new_header:
        # Adjust for truncation first, then for binning
        new_header['CRPIX2'] = (new_header['CRPIX2'] - offset_y - 0.5) / bin_size + 0.5
    
    # Add history record
    new_header['HISTORY'] = f'Binned {bin_size}x{bin_size} pixels spatially'
    if discard_naxis1 > 0 or discard_naxis2 > 0:
        new_header['HISTORY'] = f'Truncated {discard_naxis2}x{discard_naxis1} pixels using "{truncate_method}" method'
    
    return new_header


def bin_fits_file(input_file, output_file=None, bin_size=2, is_variance=True, truncate_method='auto'):
    """
    Bin a FITS file spatially and save the result.
    
    Parameters
    ----------
    input_file : str
        The input FITS file
    output_file : str, optional
        The output FITS file, by default None (will use input_file with suffix 'binned')
    bin_size : int, optional
        The binning size, by default 2
    is_variance : bool, optional
        Whether HDU 2 contains variance values (True) or standard errors (False),
        by default True for MUSE data which typically contains variance
    truncate_method : str, optional
        How to handle dimensions not divisible by bin_size:
        - 'auto': Discard pixels at the end (default)
        - 'start': Discard pixels from the beginning
        - 'end': Discard pixels from the end
        - 'center': Discard pixels from both ends equally
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = base + '_binned' + ext
    
    print(f"Reading data from {input_file}")
    with fits.open(input_file) as hdul:
        # Create a new HDUList for the output
        new_hdul = fits.HDUList()
        
        # Process each HDU
        for i, hdu in enumerate(hdul):
            if i == 0:
                # Primary header - copy it
                new_hdul.append(fits.PrimaryHDU(header=hdu.header))
            elif i == 1:  # Flux data
                if len(hdu.data.shape) == 3:  # Check if it's a 3D data cube
                    print(f"Binning flux data in HDU {i} using SUM operation")
                    # For flux, we sum the values
                    binned_data = bin_data_cube(hdu.data, bin_size, operation='sum', truncate_method=truncate_method)
                    new_header = update_header(hdu.header, bin_size, truncated_shape=binned_data.shape, truncate_method=truncate_method)
                    new_hdu = fits.ImageHDU(data=binned_data, header=new_header)
                    new_hdul.append(new_hdu)
                else:
                    # Just copy the HDU if it's not a 3D data cube
                    new_hdul.append(hdu.copy())
            elif i == 2:  # Variance/error data
                if len(hdu.data.shape) == 3:  # Check if it's a 3D data cube
                    print(f"Binning error data in HDU {i} with proper error propagation")
                    
                    # Use the is_variance parameter to determine how to handle HDU 2
                    # For MUSE data, HDU 2 typically contains variance values (not standard errors)
                    
                    # You could also use header keywords to determine this if available
                    # For example: is_variance = 'VARIANCE' in hdu.header.get('EXTNAME', '').upper()
                    
                    if is_variance:
                        print("  Treating HDU 2 as variance data (summing directly)")
                        # For variance values, we simply sum them
                        binned_data = bin_data_cube(hdu.data, bin_size, operation='sum', truncate_method=truncate_method)
                    else:
                        print("  Treating HDU 2 as standard error data (converting to variance before summing)")
                        # For standard errors, we need to square, sum, then take sqrt
                        # Square the errors to get variance
                        variance = hdu.data**2
                        # Sum the variances
                        binned_variance = bin_data_cube(variance, bin_size, operation='sum', truncate_method=truncate_method)
                        # Convert back to standard errors
                        binned_data = np.sqrt(binned_variance)
                    
                    new_header = update_header(hdu.header, bin_size, truncated_shape=binned_data.shape, truncate_method=truncate_method)
                    new_hdu = fits.ImageHDU(data=binned_data, header=new_header)
                    new_hdul.append(new_hdu)
                else:
                    # Just copy the HDU if it's not a 3D data cube
                    new_hdul.append(hdu.copy())
            else:
                # Copy any other HDUs
                new_hdul.append(hdu.copy())
        
        print(f"Writing binned data to {output_file}")
        new_hdul.writeto(output_file, overwrite=True)
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bin a MUSE data cube spatially, merging every 2x2 pixels into 1'
    )
    parser.add_argument('input_file', help='Input FITS file')
    parser.add_argument(
        '-o', '--output_file',
        help='Output FITS file (default: input_file with suffix "_binned")',
        default=None
    )
    parser.add_argument(
        '-b', '--bin_size',
        help='Binning size (default: 2)',
        type=int,
        default=2
    )
    parser.add_argument(
        '--error_type',
        help='Type of values in HDU 2 (default: "variance")',
        choices=['variance', 'stderr'],
        default='variance'
    )
    parser.add_argument(
        '--truncate',
        help='How to handle dimensions not divisible by bin_size',
        choices=['auto', 'start', 'end', 'center'],
        default='auto',
        dest='truncate_method'
    )
    
    args = parser.parse_args()
    is_variance = args.error_type == 'variance'
    bin_fits_file(args.input_file, args.output_file, args.bin_size, is_variance, args.truncate_method)

