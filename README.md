# HostSub_GP
`HostSub_GP` is a Python package designed for precise removal of host galaxy contamination in transient longslit spectroscopy. By utilizing archival images of the host galaxies, `HostSub_GP` constructs a robust 2D model of the galaxy spectrum using Gaussian Processes (GPs). This advanced modeling enables the accurate recovery of flux from faint extragalactic transients, even when superimposed on a bright host galaxy background.

Users are recommended to use [`PypeIt`](https://pypeit.readthedocs.io/en/stable/index.html) for basic image processing, and the `spec2d` data products are naturally adopted as the input of `HostSub_GP`. Please raise an issue if you would like to post-process the data products of your favorate pipeline with `HostSub_GP`.

## Installation

### Pull the Repository

```shell
git clone https://github.com/slowdivePTG/HostSub_GP.git
cd HostSub_GP
```

### Set up a Clean Environment
For `venv` users:
```shell
python3 -m venv .venv
source .venv/bin/activate
```

For `conda` users:
```shell
conda create -n `HostSub_GP` python=3.12
conda activate `HostSub_GP`
```

### Install via `pip`

```shell
pip install .
```

### Optional Dependencies
The package includes optional dependencies that can be installed as needed:

- **PypeIt integration**: For working with PypeIt data products
  ```shell
  pip install ".[pypeit]"
  ```

**Note**: `pypeit` is an optional dependency. If you plan to use PypeIt's `spec2d` data products as input, install with the `pypeit` extra as shown above.


## Citation

If you make use of ``HostSub_GP`` in your research, please cite the following publication:

- Liu & Miller, 2025, [arXiv:2508.15278](https://arxiv.org/abs/2508.15278).
