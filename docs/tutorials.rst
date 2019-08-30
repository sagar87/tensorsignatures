=========
Tutorials
=========


Creating sample data with :code:`TensorSignatureData`
=====================================================

TensorSignatures deciphers mutational signatures in context transcription,
replication and arbitrary genomic environments, which is achieved by partitioning
single nucleotide variants (SNVs) to meaningful states dependent on their
genomic location. In transcribed genomic regions, for example, mutations may
arise on coding or template strand DNA. In TensorSignatures, we represent each
genomic feature in a separate dimension with a discrete number of states.


The :code:`tensorsignatures.data` module provides :obj:`TensorSignatureData`
class allowing us to create simulated data, which we will explore in the
following to understand the meaning of various parameters of the model. To
create such data, we load the module and initalize :obj:`TensorSignatureData`
object

>>> from tensorsignatures.data import TensorSignatureData
>>> data_set = TensorSignatureData(seed=97873, rank=2, samples=100, dim=[2], mutations=1000)

which will create a dataset comprising 100 cancer genomes exposed to two
signatures each with 1000 mutations. To obtain the corresponding count tensor,
we invoke the :code:`snv` method of :code:`data_set`.

>>> snv = data_set.snv()
>>> snv.shape
(3, 3, 2, 96, 100)

The shape attribute of the :code:`snv` object is tuple of integeres indicating
size of the array in each dimension. To decipher tensor signatures from a
SNV count tensor it must have the following structure:

* :code:`snv.shape[0] == 3` Transcription (coding strand, template strand,
    unknown)
* :code:`snv.shape[1] == 3` Replication (leading strand, lagging strand,
    unknown)
* :code:`snv.shape[2:-3]` Arbitrary genomic dimension
* :code:`snv.shape[-2] == 96` Trinucleotide dimension
* :code:`snv.shape[-1] == n` Number of samples

From this we can see that :code:`data_set` contains only a single additional
genomic dimension of size 2 as :code:`snv.shape[2] == 2`. Note, that we can
reconstruct well acquinted 96 trinucleotide profiles for each sample by summing
over the first dimensions

>>> snv_collapsed = snv.sum(axis=(0, 1, 2,)) # snv_collapsed.shape == (96, 100)
>>> fig, axes = plt.subplots(3, 3, sharey=True, sharex=True)
>>> for i, ax in enumerate(np.ravel(axes)):
        ax.bar(np.arange(96), snv_collapsed[:, i], color=ts.DARK_PALETTE, edgecolor="None")
        ax.set_title('Sample {}'.format(i))
        plt.tight_layout()

.. figure::  images/samples.png
   :align:   center













To use tensorsignatures in a project::

    import tensorsignatures
