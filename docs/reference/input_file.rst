Input File
-----------

Overview
~~~~~~~~~~

An example input file for HostSub_GP is provided below. This file specifies the necessary parameters in the pipeline, including regions on the 2D spectrum to be used for modeling, properties of spectrographs, as well as details in building the host galaxy prior, adaptive batching for host emission lines, and seeing matching between imaging priors and spectra. The file also includes a data block that lists the input spectra to be processed, along with their associated metadata (e.g., ``frametype`` and ``objid``).

.. code-block:: ini

    # User-defined parameters
    [hostsub]
        slit_len = 60.0
        spec_range = 3000, 10000
        mask_wid = 2.5
        sky_region = -10, 10.
        host_wid = 10.0
        mask_offset = 0.0
        sky_offset = 0.0
        batch_2d = 2, 256
        raw_dir = /path/to/raw/data/
        spat_resln = 1.0

        [[host_prior]]
            survey = PS1
            filters = grizy

        [[host_emission]]
            find_host_emission = True
            # z = 0.0

        [[seeing_match]]
            dseeing_lower = 0.0
            dseeing_upper = 1.0

    # Data block for input spectra
    hostsub read
        path /path/to/spec2d/files/
        filename              | frametype | objid
        spec1d_STANDARD.fits  |  standard | SPAT2032-SLIT2055-DET02
        spec1d_SCIENCE_1.fits |   science | SPAT2032-SLIT2055-DET02
        spec1d_SCIENCE_2.fits |   science | SPAT2032-SLIT2055-DET02
    hostsub end

File Format
~~~~~~~~~~~