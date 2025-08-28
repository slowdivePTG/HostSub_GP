Spectrographs
----------------
The HostSub_GP package supports the following spectrographs:

- Keck/LRIS
- MMT/Binospec
- NOT/ALFOSC

Add New Spectrograph
~~~~~~~~~~~~~~~~~~~~~~
Adding support for a new spectrograph that can be processed by PypeIt is straightforward. All what HostSub_GP needs is the headers of a few key parameters in the FITS files, including:
- Slit width
- Slit position angle
- Spectral resolution
- Pixel scale

Otherwise, please raise an issue to request support for your favorite spectrograph.