# HostSub_GP
Modeling the 2d spectrum of host galaxies with Gaussian process (GP) for better background subtraction in supernova spectroscopy.

## Input
- A reduced 2d spectrum (2d array: spatial v.s. spectral)
    - Flat/bias/dark fields calibrated
    - Wavelength calibrated
    - Cosmic ray removed
- Spatial location of the SN
- Seeing
- Spectral resolution
- Slit position/orientation
- Images of the host galaxy (in multiple filters)

Users are recommended to use `PypeIt` for basic image processing, and the `spec2d` data products are naturally adopted as the input of `hostsub_gp`. Please raise an issue or reach out to the developer if you would like to post-process the data products of your favorate pipeline with `hostsub_gp`.

## Output
- Sky/host background subtracted 2D spectra

## Installation (under developement)

### Basic Installation
```shell
git clone https://github.com/slowdivePTG/HostSub_GP.git
cd HostSub_GP
conda env create -f environment.yml
conda activate hostsub
pip install -e .
```

### Optional Dependencies
The package includes optional dependencies that can be installed as needed:

- **PypeIt integration**: For working with PypeIt data products
  ```shell
  pip install -e ".[pypeit]"
  ```

**Note**: `pypeit` is an optional dependency. If you plan to use PypeIt's `spec2d` data products as input, install with the `pypeit` extra as shown above.


## Citation

If you make use of `hostsub_gp` in your research, please cite the following publication:

- Liu et al. (2025): in prep.
