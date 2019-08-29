=========
Tutorials
=========


Understanding tensor signatures
===============================

Creating sample data with :code:`TensorSignatureData`
-----------------------------------------------------

The :code:`tensorsignatures.data` module provides :obj:`TensorSignatureData`
class allowing us to create simulated data, which we will explore in the
following to understand the meaning of various parameters of the model. To
create such data, we load the module and initalize :obj:`TensorSignatureData`
object

>>> from tensorsignatures.data import TensorSignatureData
>>> data_set = TensorSignatureData(seed=97873, rank=2, samples=100, dim=[2], mutations=1000)

which will create a dataset comprising 100 cancer genomes exposed to two
signatures each with 1000 mutations. To obtain the corresponding count tensor,
we invoke the :code:`snv` method of :code:'data_set'.

>>> snv = data_set.snv()
>>> snv.shape
(3, 3, 2, 96, 100)








To use tensorsignatures in a project::

    import tensorsignatures
