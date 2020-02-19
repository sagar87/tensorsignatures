=========
Tutorials
=========


Getting started
===============

TensorSignatures deciphers mutational signatures in context transcription,
replication and arbitrary genomic environments, which is achieved by partitioning
single nucleotide variants (SNVs) to different states dependent on their
genomic location, and representing this data in a multidimensional array (tensor).
Additionally, the algorithm takes a secondary mutation matrix as input to link
other variant types to these signatures by sharing their exposure. In the following
tutorial, we want to convey an intutition for working with such highdimensional
data and explain the usage of the :code:`tensorsignatures` API and command line
interface by

1. simulating data set
2. running Tensorsignatures
3. exploring inferred parameters.


Creating sample data with :code:`TensorSignatureData`
-----------------------------------------------------

The :code:`tensorsignatures.data` module provides the :obj:`TensorSignatureData`
class allowing us to create simulated data, which we will explore in the
following to understand the meaning of various parameters of the model. To
create such data, we load the module and initalize :obj:`TensorSignatureData`
object

>>> import tensorsignatures as ts
>>> data_set = ts.TensorSignatureData(seed=573, rank=3, samples=100, dimensions=[3, 5], mutations=1000)

which will create a dataset comprising 100 cancer genomes (:code:`samples`)
exposed to three signatures (:code:`rank`) each with 1000 mutations. By passing
the list :code:`[3, 5]` to the :code:`dimension` argument, we create two additional
genomic dimensions with size 3 and 5 respectively. To obtain the SNV count tensor,
we invoke the :code:`snv` method of :code:`data_set`, which returns a
multidimensional array.

>>> snv = data_set.snv()
>>> snv.shape
(3, 3, 3, 5, 96, 100)

The shape attribute of the :code:`snv` object is tuple of :code:`int` s indicating
the size of the array in each dimension. TensorSignatures expects input data to follow
a specific structure which is explained in the following table.

+----------------------------+-----------+-----------+---------------------------+
| Dimension                  | Size      | Index     | Data                      |
+----------------------------+-----------+-----------+---------------------------+
| Transcription              | :code:`3` | :code:`0` | Coding strand mutations   |
|                            |           +-----------+---------------------------+
| (:code:`snv.shape[0]`)     |           | :code:`1` | Template strand mutations |
|                            |           +-----------+---------------------------+
|                            |           | :code:`2` | Unassigned mutations      |
+----------------------------+-----------+-----------+---------------------------+
| Replication                | :code:`3` | :code:`0` | Leading strand mutations  |
|                            |           +-----------+---------------------------+
| (:code:`snv.shape[1]`)     |           | :code:`1` | Lagging strand mutations  |
|                            |           +-----------+---------------------------+
|                            |           | :code:`2` | Unassigned mutations      |
+----------------------------+-----------+-----------+---------------------------+
| First aribtrary genomic    |:code:`t+1`| :code:`0` | Unassigned mutations      |
| dimension                  |           +-----------+---------------------------+
|                            |           | :code:`1` | Genomic state 1 mutations |
| (eg. epigenetic states)    |           +-----------+---------------------------+
|                            |           | ...       |                           |
|                            |           +-----------+---------------------------+
| (:code:`snv.shape[2]`)     |           | :code:`t` | Genomic state t mutations |
+----------------------------+-----------+-----------+---------------------------+
| Last arbitrary genomic     |:code:`r+1`| :code:`0` | Unassigned mutations      |
| dimension                  |           +-----------+---------------------------+
|                            |           | :code:`1` | Genomic state 1 mutations |
| (eg. nucleosomal states)   |           +-----------+---------------------------+
|                            |           | ...       |                           |
|                            |           +-----------+---------------------------+
| (:code:`snv.shape[-3]`)    |           | :code:`r` | Genomic state r mutations |
+----------------------------+-----------+-----------+---------------------------+
| Base substitution types    | p=        | :code:`0` | A[C>A]A                   |
|                            | :code:`96`+-----------+---------------------------+
| (:code:`snv.shape[-2]`)    |           | :code:`1` | A[C>A]C                   |
|                            |           +-----------+---------------------------+
|                            |           | ...       |                           |
|                            |           +-----------+---------------------------+
|                            |           | :code:`p` | T[T>C]T                   |
+----------------------------+-----------+-----------+---------------------------+
| Samples                    | :code:`n` | :code:`0` | Sample 1                  |
|                            |           +-----------+---------------------------+
| (:code:`snv.shape[-1]`)    |           | ...       |                           |
|                            |           +-----------+---------------------------+
|                            |           | :code:`n` | Sample n                  |
+----------------------------+-----------+-----------+---------------------------+

From this we can see that our simulated :code:`data_set` contains two additional
genomic dimensions with size 3 and 5 respectively. We can index the SNV tensor like 
any other numpy array. For example, to obtain template and leading strand mutations 
from "unassigned" genomic regions, we would simply index the tensor with 
:code:`snv[0, 1, 0, 0, :, :]` which returns a two dimensional array with mutation 
types along the first axis and samples along the other.

Note, that we can reconstruct the :math:`p\times n` mutation count matrix, which
usually serves as an input for conventional mutational signature analysis, by summing
over all dimensions except the last two (representing base substitution types
and samples respectively). The following code illustrates this operation.

>>> snv_collapsed = snv.sum(axis=(0,1,2,3,))

To inspect the mutational spectra of the first 9 samples we excecute.

>>> import matplotlib.pyplot as plt
>>> fig, axes = plt.subplots(3, 3, sharey=True, sharex=True)
>>> for i, ax in enumerate(np.ravel(axes)):
>>>    ax.bar(np.arange(96), snv_collapsed[:, i], color=ts.DARK_PALETTE)
>>>    ax.set_title('Sample {}'.format(i))
>>>    if i%3==0: ax.set_ylabel('Counts')
>>>    if i>=6: ax.set_xlabel('Mutation type')

.. figure::  images/samples.png
   :align:   center

However, by first selecting a specific states and then summing over all
remaining dimensions, we can reveal changes across different genomic dimensions
or states. For example, to get all coding and template strand mutations of the
dataset we could compute

>>> snv_coding = snv[0,].sum(axis=(0,1,2,4))
>>> snv_template = snv[1,].sum(axis=(0,1,2,4))

and then inspect both spectra by

>>> fig, axes = plt.subplots(1, 2, sharey=True)
>>> axes[0].bar(np.arange(96), snv_coding, color=ts.DARK_PALETTE)
>>> axes[0].set_title('Coding strand mutations')
>>> axes[1].bar(np.arange(96), snv_template, color=ts.DARK_PALETTE)
>>> axes[1].set_title('Template strand mutations')

.. figure::  images/transcription.png
   :align:   center

which reveals that especially C>A (blue), C>T (red) and T>A (grey) mutations
are more prevalent on coding strand DNA.

By indexing the the SNV tensor appropriately we can also recover mutational
spectra from different state combinations, eg. :code:`snv[0,:,2].sum(axis=(0,1))`
would return a :math:`p\times n` matrix representing the coding strand
mutations in state 2 of the first additional genomic dimension.

To summarize, in this section we created a simulated SNV tensor using the
:obj:`tensorsignatures.util.TensorSignatureData` class. TensorSignatures features
the characterization of mutational processes across an arbitrary number of
genomic dimensions and states, but requires the user structure their input array in
specified manner. The SNV count tensor must contain transcriptional and
replicational variants in the first two dimension, and specify base substitutions
as well as samples in the last two dimensions. To recover mutational spectra
in specific contexts, the SNV count tensor has to be indexed and summed over
all remaining dimensions (except the ones containing base substitutions and
samples).

Understanding transcriptional and replicational biases
------------------------------------------------------

In the previous section, we created a example dataset using the
:obj:`TensorSignaturesData` object and investigated the data by plotting
mutational spectra in various genomic contexts. While doing this, we discovered
that some variant types seem to occur with higher frequency on conding strand
DNA as compared to their equivalents on template DNA. Such phenomena have been
observed in several mutational processes and are, for example, attributed to DNA
repair mechanisms such as transcription coupled repair (TCR), which actively
depletes mutations gene encoding regions.

TensorSignatures models variability in mutagenesis due to transcription and
replication by

1. extracting separate mutational spectra for coding and template strand DNA, and lead and lagging strand DNA
2. fitting a scalar for each signature that quantifies the overall shift of mutations in pyrimidine context (bias matrix :code:`b`)
3. fitting a scalar for each signature that is interpreted as the relative signature activity of signature in transcribed vs untranscribed regions, and early and late replicating regions (activity matrix :code:`a`).

To understand this, we can plot the signatures that created the 
simulated counts in :code:`snv` by

>>> plt.figure(figsize=(16, 3))
>>> ts.plot_signatures(data_set.S.reshape(3,3,-1,96,3))

.. figure::  images/signatures.png
   :align:   center

which reveals the SNV spectra of three signatures (rows) for transcription and
replication in the left and right column. In this representation colors indicate 
the mutation type (blue C>A, black C>G, red C>T, grey T>A, green
T>C and salmon T>G), while shading indicates the mutation type probabilities for 
coding strand and leading strand DNA (dark), and for template and lagging strand DNA 
(light), respectively. Notice, how in the first signature (second row) dark blue bars 
are larger in comparison to accompanying light blue bars, indicative for the fact that
this mutational process is more likely to produce C>A mutations on coding and leading 
strand mutations.

Rather than investigating strand biases on mutation type level, it may be desireable 
to obtain an point estimate that quantifies the overall propensity of each mutational 
process to generate strand specific mutations. To accomplish this, TensorSignatures 
fits two (for transcription and replication) scalar variables for each extracted signature
that scale trinucleotide spectra by the amount of the detected strand asymmetry. 
We can employ the :code:`ts.heatmap` function to visualize them for our simulated data set.

>>> plt.figure(figsize=(6,3))
>>> ts.heatmap(np.exp(data_set.b0), 
           vmin=.5, vmax=2, # allows to specify the limits of the colorbar
           row_labels=['transcription', 'replication'],
           col_labels=['Signature {}'.format(i) for i in range(3)],
           cbarlabel='Strand bias (No bias = 1)' # color bar label
          )

.. figure::  images/strand_biases.png
   :align:   center

Rows display the context and columns repspective signatures. Note the logarithmic scaling 
of the colorbar which indicates that a baseline value of 1 resembles a mutational process 
with no strand preference.


Signature activities on transcribed/untranscribed and early/late replicating genomic regions
--------------------------------------------------------------------------------------------

In addition to strand biases, TensorSignatures evaluates the activity of signatures in
transcribed vs. untranscribed and early vs. late replicating regions. To visualize the
these coefficients for our artificial dataset, we again make use of the :code:`ts.heatmap`
function

>>> plt.figure(figsize=(6,3))
>>> ts.heatmap(np.exp(data_set.a0), 
           vmin=.5, vmax=2, # allows to specify the limits of the colorbar
           row_labels=['transcription', 'replication'],
           col_labels=['Signature {}'.format(i) for i in range(3)],
           cbarlabel='Signature activities (Baseline = 1)' # color bar label
          )

.. figure::  images/amplitudes.png
   :align:   center

The interpretation of this plot is as follows: A baseline value of 1 indicates that there is
no difference in signature activity between transcribed and untranscribed, or early and late
replicating regions. Values < 1 indicate a larger activity of the mutational process in untranscribed
or late replicating regions, while coefficients > 1 imply the opposite; an enhanced activity
in transcribed or early replicating regions.

Signature activities in specific genomic regions
------------------------------------------------

The multidimensional representation of SNV count data allows TensorSignatures to quantify the propensity of
mutational processes within confined genomic regions. These genomic contexts, thereafter also genomic states, 
can represent genomic features such as specific chromatin marks or nucleosome occupancy. To illustrate this,
we depicted a genomic region in the figure below together with arbitrary genomic states
and respective mutations from pooled cancer genomes.  

.. figure::  images/genomic_states_dist.png
   :align:   center
   
   Illustration of genomic states which represent confined genomic regions with certain features.

The rainfall plot representation may not always reveal changes in the mutational spectrum on first sight. 
However, the SNV count tensor contains the mutational spectra of each state combination. We can inspect 
them by indexing the respective state and summing over all remaining dimension except the one for
trinucleotides. To visualize the mutational spectra along the five states of the fourth dimension of
our simulated dataset we would type

>>> fig, ax = plt.subplots(1, 5, figsize=(16, 2.5), sharey=True)
>>> ax[0].bar(np.arange(96), snv[:,:,:,0].sum(axis=(0,1,2,4)), color=ts.DARK_PALETTE)
>>> ax[0].set_title('Baseline')
>>> ax[1].bar(np.arange(96), snv[:,:,:,1].sum(axis=(0,1,2,4)), color=ts.DARK_PALETTE)
>>> ax[1].set_title('Genomic state 1')
>>> ax[2].bar(np.arange(96), snv[:,:,:,2].sum(axis=(0,1,2,4)), color=ts.DARK_PALETTE)
>>> ax[2].set_title('Genomic state 2')
>>> ax[3].bar(np.arange(96), snv[:,:,:,3].sum(axis=(0,1,2,4)), color=ts.DARK_PALETTE)
>>> ax[3].set_title('Genomic state 3')
>>> ax[4].bar(np.arange(96), snv[:,:,:,4].sum(axis=(0,1,2,4)), color=ts.DARK_PALETTE)
>>> ax[4].set_title('Genomic state 4')

.. figure::  images/state_spectra.png
   :align:   center

   Pooled mutational spectra across the fourth dimension of the SNV count tensor.


To address questions whether a mutational signature is found within genomic regions 
with a specific histone moddification (thereafter a *genomic state*), one could classify SNVs accordingly, and 
introduce them as state in a dimension of the SNV  count tensor. Based on the SNV 
count patterns present in these regions, the program fits a parameter for each mutational 
signature and genomic state with respect to the baseline (NA, contains all mutations that 
could not be assigned to a specificic genomic feature) state to account for differential signature activity.

>>> plt.figure(figsize=(3,2))
>>> ts.heatmap(data_set.K['k1'].reshape(-1, data_set.rank),
           row_labels=['Baseline (NA)', 'Genomic state 1', 'Genomic state 2', 'Genomic state 3', 'Genomic state 4'],
           col_labels=['Sig. {}'.format(i) for i in range(3)],
           cbarlabel='Relative Signature\nactivity (Baseline = 1)' # color bar label
          ) 

.. figure::  images/genomic_state.png
   :align:   center

In this plot, we see the genomic activity of mutational signatures across the second dimension of our 
simulated SNV tensor :code:`(3,3,4,5,96,n)`. Since we deal with simulated data, we used generic
row labels (*Genomic state 1-4*) to indicate arbitrary genomic states. Usually, the majority of SNVs 
do not fall into specific genomic states and therefore end up in the baseline state, which is always 1, 
and to which all other coefficients are inferred relatively to. To understand this, consider the signature 1 
column: in genomic state 1 signature activity is decreased by one half while in genomic state 2 propensity 
of signature 1 is elevated 6x in comparison to the genomic baseline.





Running TensorSignatures on example data
========================================











To use tensorsignatures in a project::

    import tensorsignatures
