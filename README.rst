HostSub_GP
==========

`HostSub_GP` is a Python package designed for precise removal of host galaxy contamination in transient longslit spectroscopy. By utilizing archival images of the host galaxies, `HostSub_GP` constructs a robust 2D model of the galaxy spectrum using Gaussian Processes (GPs). This advanced modeling enables the accurate recovery of flux from faint extragalactic transients, even when superimposed on a bright host galaxy background.

Users are recommended to use `PypeIt <https://pypeit.readthedocs.io/en/stable/index.html>`_ for basic image processing, and the `spec2d` data products are naturally adopted as the input of `HostSub_GP`. Please raise an issue if you would like to post-process the data products of your favorite pipeline with `HostSub_GP`.

Citation
--------

If you make use of ``HostSub_GP`` in your research, please cite the following publication:

- Liu & Miller, 2025, `arXiv:2508.15278 <https://arxiv.org/abs/2508.15278>`_.