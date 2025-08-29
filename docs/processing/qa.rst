QA
----------------

As a critical step in the HostSub_GP workflow, users should perform quality assurance (QA) checks to ensure the accuracy and reliability of the host galaxy subtraction. HostSub_GP generates a series of QA plots that visualize the results of the data preprocessing, GP modeling, and host subtraction. These plots are saved in the ``QA/`` folder.

Here we provide an overview of the key QA plots with the example of `SN2019eix <https://www.wis-tns.org/object/2019eix>`_, a Type Ia supernova emerged on the bar of a bright galaxy, observed with Keck/LRIS.

Host Galaxy Prior
~~~~~~~~~~~~~~~~~~~~~
.. image:: ../figures/host_prior.svg
   :alt: Host galaxy prior from archival multi-band images
   :align: center
   :width: 700px


Pre-processed 2D Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: ../figures/raw.svg
   :alt: Pre-processed 2D spectra, user-selected sky regions, and the separation of 1D and 2D components
   :align: center
   :width: 700px


Prior Check
~~~~~~~~~~~~~~~~~
.. image:: ../figures/host_profile_prior.svg
   :alt: Comparison between the observed and prior 2D spectral profiles
   :align: center
   :width: 700px


Posterior Check
~~~~~~~~~~~~~~~~~~~~~
.. image:: ../figures/host_profile_pred.svg
   :alt: 2D spectral profile predicted by the GP model
   :align: center
   :width: 700px


Predicted 2D Spectra
~~~~~~~~~~~~~~~~~~~~~~
.. image:: ../figures/pred.svg
   :alt: Predicted 2D spectra
   :align: center
   :width: 700px


Extracted 1D Spectra
~~~~~~~~~~~~~~~~~~~~~~
.. image:: ../figures/sci.svg
   :alt: 1D spectra extracted from the host-subtracted 2D spectra
   :align: center
   :width: 700px