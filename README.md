# HostSub_GP
Modeling the 2d spectrum of host galaxies with Gaussian Process for better background subtraction in supernova spectroscopy.

## Input
### Required
- A reduced 2d spectrum (2d array: spatial v.s. spectral)
    - Flat/bias/dark fields calibrated
    - Wavelength calibrated
    - Cosmic ray removed
- Spatial location of the SN
- Seeing
- Spectral resolution
- Instrumental sensitivity function

### Optional
- Slit position/orientation
- Images of the host galaxy (in multiple filters)
- A spectrum of the host galaxy

## Output
- Fluxed calibrated spectra (2d/1d) of the host galaxy/SN 
- Estimated flux uncertainties