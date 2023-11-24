.. pybl documentation master file, created by KilinWei

Welcome to pybBL's documentation!
=================================

pyBL is an open-source Python package for stochastic rainfall modelling based upon the randomised Bartlett-Lewis (BL) rectangular pulse model.
The BL model is a type of stochastic model that represents rainfall using a Poisson cluster point process.
This package implements the most recent version of the BL model, based upon the state-of-the-art BL model developed in Onof and Wang (2020).

**Features:**

- Optimized data structure and algorithms for performance.
- Loosely coupled components for easy extension for specific needs.
- User-friendly interface for anyone to use.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: For Users:

   Installation <user_guide/installation>
   API Reference <pybl_reference/index>
   Configuration <user_guide/configuring>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: For developers:

   Contributing <developer_guide/contribute_guideline.rst>
   Testing <developer_guide/test_pybl.rst>
   Building Docs <developer_guide/build_the_doc.rst>
   Packaging <developer_guide/packaging.rst>
