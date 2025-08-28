.. HostSub_GP documentation master file, created by
   sphinx-quickstart on Wed Aug 27 15:35:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HostSub_GP documentation
========================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   changelog
   cookbook

.. toctree::
   :maxdepth: 1
   :caption: Core Processing
   :hidden:

   processing/qa
   
.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   reference/input_file
   reference/parameters
   reference/spectrographs

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   api/modules

Overview
--------

HostSub_GP is a Python package for modeling the 2D spectrum of host galaxies with Gaussian Processes (GPs), enabling improved background subtraction in transient (e.g., supernova) spectroscopy.

Features
--------

- Utilizing archival multi-band images to construct a wavelength-dependent prior model of the host galaxy spectrum
- Robust modeling of host galaxy spectra using Gaussian Processes (GPs)
- Using JAX for efficient computation and automatic differentiation (GPU acceleration to be supported soon)
- Integration with PypeIt data products

.. note::

   For installation instructions, see :doc:`installation`.

Application Requirements
-------------------------

- Only longslit spectra are supported.
- The host should be spatially resolved in the 2D spectrum (i.e., much larger than the seeing).
- The host should have no significant variability (e.g., not an AGN).
- Multi-band archival images of the host galaxy are required to construct a prior model covering the full wavelength range of the spectrum.

Citation
-------------

If you find HostSub_GP useful in your research, please cite the following publication:

- Liu & Miller, 2025, `arXiv:2508.15278 <https://arxiv.org/abs/2508.15278>`_.

.. automodule:: hostsub_gp
   :members:
   :undoc-members:
   :show-inheritance: