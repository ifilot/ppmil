PPMIL: Pure Python Molecular Integral Library
=============================================

.. image:: https://img.shields.io/github/v/tag/ifilot/ppmil?label=version
   :alt: GitHub tag (latest SemVer)
.. image:: https://github.com/ifilot/ppmil/actions/workflows/build_pypi.yml/badge.svg
   :target: https://github.com/ifilot/ppmil/actions/workflows/build_pypi.yml
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

:program:`PPMIL` is a pure-Python library for computing one- and two-electron
integrals over Cartesian Gaussian basis functions, as commonly required in
electronic structure calculations. It was developed as a counterpart to
`PyQInt <https://ifilot.github.io/pyqint>`_. Unlike :program:`PyQInt`,
:program:`PPMIL` does not rely on a Cython back end. While this design choice
results in lower computational performance, it aims to improve usability and
accessibility by leveraging only the Python programming language.

.. warning::

   :program:`PPMIL` is stil very much in development. Do not use it for any
   production purposes.

:program:`PPMIL` currently supports the following type of integrals:

* Overlap integrals
* Dipole integrals
* Kinetic integrals
* Nuclear integrals
* Two-electron integrals

:program:`PPMIL` has been developed at the Eindhoven University of Technology,
Netherlands. :program:`PPMIL` and its development are hosted on `github
<https://github.com/ifilot/ppmil>`_.  Bugs and feature
requests are ideally submitted via the `github issue tracker
<https://github.com/ifilot/ppmil/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   community_guidelines

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
