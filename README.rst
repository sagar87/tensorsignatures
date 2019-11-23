################
TensorSignatures
################


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

*************
Quick install
*************

To install :code:`tensorsignatures` simply type

.. code-block:: console

    $ pip install tensorsignatures

into your shell. To get started with tensorsignatures please refer to the
documentation.

* Free software: MIT license
* Documentation: https://tensorsignatures.readthedocs.io.


***************
Getting started
***************


Step 1: Data preparation
========================

To apply TensorSignatures on your data single nucleotide variants (SNVs) need to
be split according to their genomic context and represented in a highdimensional
count tensor. Similarly, multinucleotide variants (MNVs), deletions and indels
(indels) have to be classified and represented in count matrix (currently we
do not provide a automated way of generating a structural variant table yet).
Despite the fact that TensorSignatures is written in Python, this part of the
pipeline runs in :code:`R` and and depends on the :code:`bioconductor` packages
:code:`VariantAnnotation` and :code:`rhdf5`. Make sure you have :code:`R3.4.x`
installed, and the packages :code:`VariantAnnotation` and :code:`rhdf5`. You can
install them, if necessary, by executing

.. code-block:: console

    $ Rscript -e "source('https://bioconductor.org/biocLite.R'); biocLite('VariantAnnotation')"

and

.. code-block:: console

    $ Rscript -e "source('https://bioconductor.org/biocLite.R'); biocLite('rhdf5')"

from your command line.

To get started, download the following files and place them in the same directory:

`Constants.RData <http://193.62.55.163/file/R/constants.RData>`_ (contains
:code:`GRanges` objects that annotate transcription/replication orientation,
nucleosomal and epigenetic states)

`mutations.R <http://193.62.55.163/file/R/mutations.R>`_ (all required functions
to partiton SNVs, MNVs and indels)

`processVcf.R <http://193.62.55.163/file/R/processVcf.R>`_ (loads :code:`vcf`
files and creates the SNV count tensor, MNV and indel count matrix; eventually
needs custom modification to make the script run on your vcfs.)

`genome.zip <http://193.62.55.163/file/R/genome.zip>`_ (optionally).

To obtain the SNV count tensor and the matrices containing all other mutation
types try to execute

.. code-block:: console

    $ Rscript processVcf.R yourVcfFile1.vcf.gz yourVcfFile2.vcf.gz ... yourVcfFileN.vcf.gz outputHdf5File.h5

which ideally outputs an hdf5 file that can be used as an input for the TensorSignatures
software. In case of errors please check wether you have correctly specified paths
in line 6-8. Also, take a look at the :code:`readVcfSave` function and adjust it
in case of errors.


********
Features
********

* Run :code:`tensorsignatures` on your dataset using the :code:`TensorSignature` class provided by the package or via the command line tool.
* Compute percentile based bootstrap confidence intervals for inferred parameters.
* Basic plotting tools to visualize tensor signatures and inferred parameters

*******
Credits
*******

* Harald Vöhringer and Moritz Gerstung
