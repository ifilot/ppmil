PPMIL: Pure Python Molecular Orbital Library
============================================

.. image:: https://img.shields.io/static/v1?label=status&message=under%20development&color=ff0000
   :alt: Development status
.. image:: https://img.shields.io/github/v/tag/ifilot/ppmil?label=version
   :alt: GitHub tag (latest SemVer)
.. image:: https://github.com/ifilot/ppmil/actions/workflows/build_pypi.yml/badge.svg
   :target: https://github.com/ifilot/ppmil/actions/workflows/build_pypi.yml
.. image:: https://github.com/ifilot/ppmil/actions/workflows/build_conda.yml/badge.svg
   :target: https://github.com/ifilot/ppmil/actions/workflows/build_conda.yml
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

:program:`PPMIL` is a pure Python package for solving one- and two-electron
integrals using Cartesian Gaussian basis functions as encountered in
electronic structure calculations. :program:`PPMIL` has been created as the
counterpart of `PyQInt <https://pyqint.imc-tue.nl>`_ that explicitly does not
contain a Cython back-end and as such has a more lenient set of dependencies.

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
