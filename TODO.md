# To-do list

## Step 1: Generate mock 2d spectra
### Sources
- Galaxy (early/late-type)
- SN (photometric/nebular-phase)
### Contamination
- Skylines
- Cosmic rays (assuming removed)
### Instrumental limits (LRIS)
- Realistic PSF/seeing
- Spectral resolution
- Sensitivity function

## Step 2: Build a 2d GP model
### Fiducial model
- SN position known
- Ground-based (PSF not known)
- No prior knowledge on the galaxy profile - purely determined by the GP
    - Prior of host light - mean/median
- Modeling the host with the SN masked, then extract the SN

### Host modeling aided with images
- SN position known
- Ground-based (PSF not known)
- Galaxy profiles extracted from (multi-band) images
    - Prior of host light - images
- Modeling the host and SN simultaneously

### Host modeling aided with a known PSF (e.g., for JWST)
- SN position known
- Space telescope (PSF known)
- Galaxy profiles extracted from (multi-band) images
    - Prior of host light - TBD

## Step 3: Test on Real Data
- LRIS (+ PanSTARRS/Legacy Surveys/ZTF reference image(?))
- JWST (+?)