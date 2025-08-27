Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

**Pull the Repository**

.. code-block:: shell

   git clone https://github.com/slowdivePTG/HostSub_GP.git
   cd HostSub_GP

**Set up a Clean Environment**

For ``venv`` users:

.. code-block:: shell

   python3 -m venv .venv
   source .venv/bin/activate

For ``conda`` users:

.. code-block:: shell

   conda create -n HostSub_GP python=3.12
   conda activate HostSub_GP

**Install via pip**

.. code-block:: shell

   pip install .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The package includes optional dependencies that can be installed as needed:

- **PypeIt integration**: For working with PypeIt data products

  .. code-block:: shell

     pip install ".[pypeit]"

.. note::

   If you plan to use ``PypeIt`` for basic image processing and use the output ``spec2d`` data products as input, install with the ``pypeit`` extra as shown above.