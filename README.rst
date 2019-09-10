================
Tensorsignatures
================


.. image:: https://img.shields.io/pypi/v/tensorsignatures.svg
        :target: https://pypi.python.org/pypi/tensorsignatures

.. image:: https://img.shields.io/travis/sagar87/tensorsignatures.svg
        :target: https://travis-ci.org/sagar87/tensorsignatures

.. image:: https://readthedocs.org/projects/tensorsignatures/badge/?version=latest
        :target: https://tensorsignatures.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


TensorSignatures is a tensor factorization framework for mutational signature
analysis, which in contrast to other methods, deciphers mutational processes
not only in terms of mutational spectra, but also assess their properties with
respect to various genomic variables.

Quick install
-------------

To install :code:`tensorsignatures` simply type

.. code-block:: console
    $ pip install

into your shell. To get started with tensorsignatures please refer to the
documentation.

* Free software: MIT license
* Documentation: https://tensorsignatures.readthedocs.io.


Features
--------

* Run :code:`tensorsignatures` on your dataset using the :code:`TensorSignature`
class provided by thepackage or via the command line tool.
* Compute percentile based bootstrap confidence intervals for inferred paramters.
* Basic plotting tools to visualize tensor signatures and inferred parameters

Credits
-------

* Harald Vöhringer and Moritz Gerstung
