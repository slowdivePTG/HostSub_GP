# To-do list

## Step 1: Build a 2d GP model

### Host modeling aided with images
- SN position known
- Ground-based (PSF not known)
- Galaxy profiles extracted from (multi-band) images
    - Prior of host light - images
- References
    - `pypeit`: `skysub` - for global sky background modeling

## Step 2: Test on Real Data
- Tilts and distortion correction
- Flux resampled on a pseudo-image
- Robust inference with cosmic rays
- References
    - `pypeit`: `coadd2d` - for tilts and distortion correction & cosmic ray removal
    - `ASPIRED`: for tilts and distortion correction

## Step 3: Host modeling with a known PSF
- SN position known
- Space telescope (PSF known)
- Galaxy profiles extracted from (multi-band) images
    - Prior of host light - TBD
- JWST (+?)