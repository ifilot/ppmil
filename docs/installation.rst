.. _installation:
.. index:: Installation

Installation
============

.. tip::
    * For Windows users with relatively little experience with Python, we warmly
      recommend to use the `Anaconda distribution <https://www.anaconda.com/products/distribution>`_.
      Anaconda is an all-in-one package containing the Python compiler,
      an integrated desktop environment (Spyder) and plenty of useful Python
      packages such as numpy and matplotlib.
    * :program:`PPMIL` plays nicely with the 
      `PyLebedev <https://github.com/ifilot/pylebedev>`_ and 
      `PyTessel <https://pytessel.imc-tue.nl/>`_ packages
      which we install together with the ppmil package.

:program:`PPMIL` is distributed via both Anaconda package as well as PyPI. For
Windows, it is recommended to install :program:`PPMIL` via Anaconda, while
for Linux, we recommend to use PyPI.

Windows / Anaconda
------------------

To install :program:`PPMIL` under Windows, open an Anaconda Prompt window
and run::

    conda install -c ifilot ppmil pylebedev pytessel

.. note::
    Sometimes Anaconda is unable to resolve the package dependencies. This can
    be caused by a broken environment. An easy solution is to create a new
    environment. See the "Troubleshooting" section at the end of this page
    for more information.

Linux / PyPI
------------

To install :program:`PPMIL` systemwide, run::

    sudo pip install ppmil pylebedev pytessel

or to install :program:`PPMIL` only for the current user, run::

    pip install ppmil pylebedev
