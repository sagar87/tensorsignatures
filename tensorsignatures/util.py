

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from scipy.stats import nbinom
from scipy.stats import poisson
from scipy.stats import kstest
from scipy.stats import uniform
from collections import defaultdict

from multiprocessing import Pool
import functools
import pickle
import numpy as np
import pandas as pd
import h5py as h5
import os
import sys
import re

from tensorsignatures.config import *
from tensorsignatures.plot import plot_signatures
from tqdm import tqdm
from tqdm import trange


def lazy_property(function):
    # Source: https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def assign_signatures(reference, signature):
    r"""Assigns signatures to reference a set of reference signatures.

    Args:
        reference (:obj:`array`, :code:`shape` :math:`(p,s)`): Reference
            signature matrix (mutation types :math:`\times` signatures).
        signature (:obj:`array`, :code:`shape` :math:`(p,s)`): Second signature
            matrix.
    Returs:
        A tuple of three arrays containing the signature indices of the
            reference signatures array, corresponding signatures indices
            of the second signature matrix and underlying cosine distances.

    Examples:

    Consider signature matrices :code:`S1` and :code:`S2`.

    >>> S1.shape, S2.shape
    (96, 5), (96, 5)
    >>> ridx, sidx, dist = assign_signatures(S1, S2)
    >>> S2[:, sidx] # sorts signatures from S2 to match S1
    """
    with np.errstate(divide='raise', invalid='raise'):
        dist = cdist(reference.T, signature.T, 'cosine')

    row_ind, col_ind = linear_sum_assignment(dist)
    distances = []

    for i, j in zip(row_ind, col_ind):
        distances.append(dist[i, j])

    return row_ind, col_ind, distances


class Initialization(object):
    """Stores results of a TensorSignature run.

    Args:
        S0 (:obj:`array`): Fitted signature tensor.
        a0 (:obj:`array`): Fitted signature amplitudes.
        b0 (:obj:`array`): Fitted signature biases.
        k0 (:obj:`array`): Fitted signature activities.
        m0 (:obj:`array`): Fitted signature mixing factors.
        T0 (:obj:`array`): Fitted other mutation types signature matrix.
        E0 (:obj:`array`): Fitted exposures.
        rank (:obj:`int`): Rank of decomposition.
        dispersion (:obj:`int`): Dispersion
        objective (:obj:`str`): Used likelihood function.
        starter_learning_rate (:obj:`float`): Starter learning rate.
        decay_learning_rate (:obj:`float`): Decay of learning rate.
        optimizer (:obj:`str`): Used optimizer.
        epochs (:obj:`int`): Number of training epochs.
        log_step (:obj:`int`): Logging steps.
        display_step (:obj:`int`): Diplay steps.
        observations (:obj:`int`): Number of observations (non NA entries).
        seed (:obj:`int`): Used seed.
    Returns:
        A :obj:`tensorsignatures.util.Initialization` object.
    """
    def __init__(self, S0, a0, b0, ki, m0, T0, E0, rank, size, objective,
                 starter_learning_rate, decay_learning_rate, optimizer,
                 epochs, log_step, display_step, observations, id, init,
                 seed, log_epochs, log_learning_rate, log_L, log_L1, log_L2,
                 sample_indices):
        # store hyperparameters
        self.seed = seed
        self.objective = objective
        self.epochs = epochs
        self.log_step = log_step
        self.display_step = display_step
        self.starter_learning_rate = starter_learning_rate
        self.decay_learning_rate = decay_learning_rate
        self.optimizer = optimizer
        self.observations = observations
        self.rank = rank
        self.size = size
        self.id = id
        self.init = init
        self.iter = 1

        # make data accessible
        self._S0 = self._add_iterdim(S0)
        self._a0 = self._add_iterdim(a0)
        self._b0 = self._add_iterdim(b0)
        self._ki = {k: self._add_iterdim(v) for k, v in ki.items()}
        self._kdim = []

        for key, value in self._ki.items():
            ki = np.exp(value)
            _ki = value
            setattr(self, 'k' + str(key), ki)
            setattr(self, '_k' + str(key), _ki)
            self._kdim.append(ki.shape[0])

        self._m0 = self._add_iterdim(m0)
        self._T0 = self._add_iterdim(T0)
        self._E0 = self._add_iterdim(E0)

        self.log_epochs = self._add_iterdim(log_epochs)
        self.log_learning_rate = self._add_iterdim(log_learning_rate)
        self.log_L = self._add_iterdim(log_L)
        self.log_L1 = self._add_iterdim(log_L1)
        self.log_L2 = self._add_iterdim(log_L2)
        self.sample_indices = self._add_iterdim(sample_indices)

    def _remove_iterdim(self, array):
        # removes the iter dimension
        return array[..., 0]

    def _add_iterdim(self, array):
        # adds the iter dimension to all arrays
        return array.reshape(*array.shape, self.iter)

    @lazy_property
    def _S1(self):
        # computes the SNV signature tensor
        S0 = np.concatenate(
            [self._S0, np.zeros((2, 2, 1, self.rank, self.iter))], axis=2)
        S0 = np.exp(S0) \
            / np.sum(np.exp(S0), axis=2, keepdims=True)

        S1 = np.stack([
            S0[0, 0],
            S0[1, 0],
            0.5 * S0[:, 0].sum(axis=0),
            S0[1, 1],
            S0[0, 1],
            0.5 * S0[:, 1].sum(axis=0),
            0.5 * (S0[0, 0] + S0[1, 1]),
            0.5 * (S0[1, 0] + S0[0, 1]),
            0.25 * S0.sum(axis=(0, 1))
        ]).reshape(3, 3, -1, self.rank, self.iter)

        return S1

    @lazy_property
    def _B(self):
        # computes the bias tensor
        B = np.exp(np.stack([
            self._b0[0] + self._b0[1],
            self._b0[0] - self._b0[1],
            self._b0[0],
            self._b0[1] - self._b0[0],
            -self._b0[1] - self._b0[0],
            -self._b0[0],
            self._b0[1],
            -self._b0[1], np.zeros(self._b0[0].shape)
        ])).reshape(3, 3, 1, self.rank, self.iter)

        return B

    @lazy_property
    def _A(self):
        # computes the amplitude tensor
        a1 = np.concatenate(
            [self.a, self.a, np.ones([2, self.rank, self.iter])],
            axis=0).reshape(3, 2, self.rank, self.iter)

        A = a1[:, 0, :, :][:, None, :, :] \
            * a1[:, 1, :, :][None, :, :, :]

        return A

    @lazy_property
    def _T1(self):
        # computes softmax on the other signature matrix
        T1 = np.concatenate(
            [self._T0, np.zeros((1, self.rank, self.iter))], axis=0)
        T1 = np.exp(T1) / np.sum(np.exp(T1), axis=0, keepdims=True)

        return T1

    @lazy_property
    def _Sc(self):
        # concatenates signature profiles to cluster them
        Sc = np.concatenate([
            self.S.reshape(3, 3, -1, self.S.shape[-3], self.rank)[2, 2, 0],
            self.T[..., 0]])

        return Sc

    @lazy_property
    def S(self):
        """Returns the SNV signature tensor."""
        S = self._S1 \
            * self._B \
            * self._A.reshape(3, 3, 1, self.rank, self.iter) \
            * self.m.reshape(1, 1, 1, self.rank, self.iter)

        S = S.reshape(3, 3, *([1] * len(self._kdim)), -1, self.rank, self.iter)

        for i, j in enumerate(self._kdim):
            dim = [1] * len(self._kdim)
            dim[i] = j + 1
            dim = [1, 1] + dim + [1, self.rank, self.iter]
            ki = getattr(self, 'k' + str(i))
            S = S * np.concatenate(
                [np.ones((1, *ki.shape[1:])), ki]).reshape(*dim)

        return S

    @lazy_property
    def T(self):
        """Returns the mutatonal spectrum for other mutation types."""
        return self._T1 * (1 - self.m)

    @lazy_property
    def E(self):
        """Returns the exposure matrix."""
        return np.exp(self._E0)

    @lazy_property
    def a(self):
        """Returns signature activities in transcribed/non-transcribed (
        :code:`a[0]`), and early/late replicating regions :code:`a[1]`.
        """
        return np.exp(self._a0)

    @lazy_property
    def b(self):
        """Returns transcriptional (:code:`b[0]`) and replicational strand
        (:code:`b[1]`) biases.
        """
        return np.exp(self._b0)

    @lazy_property
    def m(self):
        """Returns the proportion of SNVs for each signature.
        """
        return 1 / (1 + np.exp(-self._m0))

    def to_dic(self):
        """Returns all parameters and settings for this initialization in form
        of a dictionary."""
        data = {}
        for var in DUMP:
            if var in VARS or var in LOGS:
                if var == ki:
                    for k, v in self._ki.items():
                        data['_k' + str(k)] = self._remove_iterdim(v)
                else:
                    data[var] = self._remove_iterdim(getattr(self, var))
            else:
                data[var] = getattr(self, var)

        return data

    def dump(self, path):
        """Saves the dictionary returned by the to_dic method to disk
        (pickleable)."""
        data = self.to_dic()
        save_dict(data, path)

    def plot_signatures(self, bootstrap=None, init=None):
        """Plots SNV signatures in transcriptional and replicational context.
        """
        if init is None:
            init = 0
        plot_signatures(
            self.S[..., init].reshape(3, 3, -1, self.S.shape[-3], self.rank),
            bootstrap)


class Cluster(Initialization):
    """Clusters tensorsignatures intializations.

    Args:
        dset (:obj:`HDF5`): HDF5 file handle to a tensor signatures experiment.
    Returns:
        A :obj:`tensorsignatures.util.Cluster` object.

    Examples:

    The :obj:`tensorsignatures.util.Cluster` object is meant to be used in
    combination :obj:`tensorsignatures.util.Experiment`. The latter takes
    the path of a hdf file containing several tensor signature intialization.

    >>> E = Experiment('~/my_experiment.h5')
    >>> clu = E['/experiment/5'] # returns cluster object
    """
    def __init__(self, dset, **kwargs):
        self.dset = dset

        # set all parameters
        self.objective = self.dset.attrs[OBJECTIVE]
        self.epochs = self.dset.attrs[EPOCHS]
        self.log_step = self.dset.attrs[LOG_STEP]
        self.display_step = self.dset.attrs[DISPLAY_STEP]
        self.starter_learning_rate = self.dset.attrs[STARTER_LEARNING_RATE]
        self.decay_learning_rate = self.dset.attrs[DECAY_LEARNING_RATE]
        self.optimizer = self.dset.attrs[OPTIMIZER]
        self.observations = self.dset.attrs[OBSERVATIONS]
        self.size = self.dset.attrs[SIZE]
        self.id = self.dset.attrs[ID]

        # cluster init
        self._valid = None
        self.seed = np.argmax(
            np.ma.array(
                self.dset[LOG_L][()][-1, :],
                mask=self.dset[LOG_L][()][-1, :] >= 0))

        self._cluster()
        # self._S0, self._T0, self._E0, self.icol = Cluster.cluster_signatures(
        #     dset[S0], dset[T0], dset[E0], self.seed)

        self.iter = self._S0.shape[-1]
        self.rank = self._S0.shape[-2]
        self.samples = self._E0.shape[-2]

        # initialize variables
        self._a0 = self._sort_array(self.dset[a0][()])
        self._b0 = self._sort_array(self.dset[b0][()])
        self._kdim = []

        for key in [var for var in list(self.dset) if var.startswith('_k')]:
            ki = np.exp(self._sort_array(self.dset[key][()]))
            _ki = np.concatenate([np.ones((1, *ki.shape[1:])), ki])
            setattr(self, key[1:], ki)
            setattr(self, key, _ki)
            self._kdim.append(ki.shape[0])

        self._m0 = self._sort_array(self.dset[m0][()])

        # compute composite variables
        self.S
        self.T
        self.E

        self.log_epochs = self.dset[LOG_EPOCHS][..., self._valid]
        self.log_learning_rate = \
            self.dset[LOG_LEARNING_RATE][..., self._valid]
        self.log_L = self.dset[LOG_L][..., self._valid]
        self.log_L1 = self.dset[LOG_L1][..., self._valid]
        self.log_L2 = self.dset[LOG_L2][..., self._valid]
        self.sample_indices = \
            self.dset[SAMPLE_INDICES][..., self._valid]

    def __len__(self):
        return self.iter

    def __iter__(self):
        for i in range(self.iter):
            yield i

    def __getitem__(self, init):
        if self._valid is not None:
            init = self._valid[init]

        ki = {}
        for key in [var for var in list(self.dset) if var.startswith('_k')]:
            ki[int(key[2:])] = self.dset[key][..., init]

        initialization = Initialization(
            S0=self.dset[S0][..., init], a0=self.dset[a0][..., init],
            b0=self.dset[b0][..., init], ki=ki, m0=self.dset[m0][..., init],
            T0=self.dset[T0][..., init], E0=self.dset[E0][..., init],
            rank=self.dset.attrs[RANK], size=self.dset.attrs[SIZE],
            objective=self.dset.attrs[OBJECTIVE],
            epochs=self.dset.attrs[EPOCHS],
            starter_learning_rate=self.dset.attrs[STARTER_LEARNING_RATE],
            decay_learning_rate=self.dset.attrs[DECAY_LEARNING_RATE],
            optimizer=self.dset.attrs[OPTIMIZER],
            log_step=self.dset.attrs[LOG_STEP],
            display_step=self.dset.attrs[DISPLAY_STEP],
            observations=self.dset.attrs[OBSERVATIONS],
            id=self.dset.attrs[ID], init=init, seed=self.dset.attrs[SEED],
            log_epochs=self.dset[LOG_EPOCHS][..., init],
            log_learning_rate=self.dset[LOG_LEARNING_RATE][..., init],
            log_L=self.dset[LOG_L][..., init],
            log_L1=self.dset[LOG_L1][..., init],
            log_L2=self.dset[LOG_L2][..., init],
            sample_indices=self.dset[SAMPLE_INDICES][..., init]
        )

        return initialization

    def __contains__(self, item):
        if 0 <= item < self.iter:
            return True

        return False

    def as_list(self, key, sign=1):
        if key in self:
            if type(self[key]) == np.ndarray:
                return (sign * self[key]).tolist()

        return [np.nan] * self.iter

    def _sort_array(self, array):
        # sort array
        var_list = []

        for k, v in self.icol.items():
            var_list.append(array[..., v, k])

        return np.stack(var_list, axis=array.ndim - 1)

    def _cluster(self):
        # performs the clustering
        S_seed = self[self.seed]._Sc

        if (np.any(np.isnan(S_seed)) or
                np.any(np.isinf(S_seed)) or
                np.all(S_seed == 0)):

            print("Warning: seed {} corrupted.".format(seed))
            return (None, None, None, None)

        S_list, T_list, E_list = [], [], []
        self.icol = {}

        for i in range(self.dset[S0].shape[-1]):
            init = self[i]
            S_i = init._Sc

            if (np.any(np.isnan(S_i)) or
                    np.any(np.isinf(S_i)) or
                    np.all(S_i == 0) or
                    np.all(init._S0 == 0)):
                continue

            ridx, cidx, _ = assign_signatures(S_seed, S_i)

            S_list.append(init._S0[..., cidx, 0])
            T_list.append(init._T0[..., cidx, 0])
            E_list.append(init._E0[cidx, :, 0])
            self.icol[i] = cidx

        self._S0 = np.stack(S_list, axis=-1)
        self._T0 = np.stack(T_list, axis=-1)
        self._E0 = np.stack(E_list, axis=-1)
        self._valid = list(self.icol.keys())

    @staticmethod
    def cluster_signatures(S, T, E, seed=None):
        """Deprecated."""

        if seed is None:
            seed = np.random.randint(S.shape[-1])

        S_seed = S[..., seed]
        T_seed = T[..., seed]

        if (np.any(np.isnan(S_seed)) or
                np.any(np.isinf(S_seed)) or
                np.all(S_seed == 0)):

            print("Warning: seed {} corrupted.".format(seed))
            return (None, None, None, None)

        if (np.any(np.isnan(T_seed)) or
                np.any(np.isinf(T_seed)) or
                np.all(T_seed == 0)):

            print("Warning: seed {} corrupted.".format(seed))
            return (None, None, None, None)

        S_shape = S_seed.shape
        S_seed = np.concatenate([
            S_seed.reshape(-1, S_seed.shape[-1]), T_seed])

        S_list, T_list, E_list = [], [], []
        i_col = {}

        for i in range(S.shape[-1]):
            S_i = S[..., i]
            T_i = T[..., i]

            if (np.any(np.isnan(S_i)) or
                    np.any(np.isinf(S_i)) or
                    np.all(S_i == 0)):
                continue

            if (np.any(np.isnan(T_i)) or
                    np.any(np.isinf(T_i)) or
                    np.all(T_i == 0)):
                continue

            S_i = np.concatenate([
                S_i.reshape(-1, S_i.shape[-1]), T_i])
            E_i = E[..., i]

            ridx, cidx, _ = assign_signatures(S_seed, S_i)

            S_list.append(S[..., i][..., cidx])
            T_list.append(T[..., i][..., cidx])
            E_list.append(E[..., i][cidx, :])

            i_col[i] = cidx

        S_clu = np.stack(S_list, axis=S.ndim - 1)
        T_clu = np.stack(T_list, axis=T.ndim - 1)
        E_clu = np.stack(E_list, axis=2)

        return (S_clu, T_clu, E_clu, i_col)

    @lazy_property
    def parameters(self):
        """Returns the number of parameters in the model."""
        p = 4 * 95
        p += self._a0.shape[0]
        p += self._b0.shape[0]

        for dim in self._kdim:
            p += dim

        p += self._m0.shape[0]
        p += self._T0.shape[0]

        p = p * self.rank
        p += self.samples * self.rank

        return p

    @lazy_property
    def log_likelihood(self):
        """Returns the log likelihood of all initalizations in the cluster."""
        return self.log_L[-1, :]

    @lazy_property
    def init(self):
        """Returns the maximum likelihood initialization."""
        return np.argmax(self.log_likelihood)

    @lazy_property
    def summary_table(self):
        """Returns a pandas dataframe summarizing various statistics about the
        initialziations of the cluster."""
        df = pd.DataFrame({
            LOG_L1: self.log_L1[-1, :].tolist(),
            LOG_L2: self.log_L2[-1, :].tolist(),
            LOG_L: self.log_likelihood.tolist(),
            SIZE: [self.size] * self.iter,
            RANK: [self.rank] * self.iter,
            INIT: np.arange(0, self.iter),
            PARAMETERS: [self.parameters] * self.iter,
            OBSERVATIONS: [self.observations] * self.iter,
        })

        df[AIC] = 2 * df[PARAMETERS] - 2 * df[LOG_L]
        df[AIC_C] = df[AIC] + \
            (2 * df[PARAMETERS]**2 + 2 * df[PARAMETERS]) \
            / (df[OBSERVATIONS] - df[PARAMETERS] - 1)
        df[BIC] = np.log(df[OBSERVATIONS]) * df[PARAMETERS] - 2 * df[LOG_L]

        return df

    def coefficient_table(self, cdim='_b0', avg=False):
        """Returns a pandas data frame with extracted parameters for each
        signature and initialisation of the cluster.

        Parameters:
            cdim (:obj:`str`): Name of the parameter (eg. "b", "a" etc.)
            avg (:obj:`bool`): Computes the average of fitted coefficient over
                initialisations.
        Returns:
            A :obj:`pandas.DataFrame` object.

        Examples:

        >>> a_table = clu.coefficient_table('a')
        """
        coeff_table = pd.DataFrame({
            'sig': np.array(
                [[i] * self[cdim].shape[0] for i in range(self.rank)]
            ).reshape(-1).tolist() * self.iter,
            'dim': np.arange(
                self[cdim].shape[0]
            ).tolist() * self[cdim].shape[1] * self[cdim].shape[2],
            'init': np.array([
                [i] * self.rank * self[cdim].shape[0] for i in range(self.iter)
            ]).reshape(-1).tolist(),
            'val': self[cdim].T.reshape(-1).tolist()})

        if avg:
            coeff_table = coeff_table.groupby(
                ['sig', 'dim']).agg({'val': [np.mean, np.std]}).reset_index()
            coeff_table.columns = [
                ' '.join(col).strip() for col in coeff_table.columns]

        return coeff_table


class Experiment(object):

    def __init__(self, path):
        """Loads a hdf file containing (several) tensorsignature experiments.

        Args:
            path (:obj:`str`): Path to the hdf file containing the experiments.
        Returns:
            A :obj:`tensorsignatures.util.Experiment` object.

        Examples:

        A :obj:`tensorsignatures.util.Experiment` is a container of
        :obj:`tensorsignatures.util.Cluster` which can be accessed via its
        :code:`__getitem__` method.

        >>> E = Experiment('~/path/to/hdf.h5')
        >>> E.data # returns a set containing all experiment
        {'myexperiment/3', 'myexperiment/4', 'myexperiment/5'}
        >>> clu = E['myexperiment/4'] # returns the rank 4 cluster
        """
        self.dset = h5.File(path)
        self.data = set([])
        self.memo = dict()

        # walk through all experiment params
        self.dset.visititems(self._visitor_func)

        if len(self.data) == 0:
            self.dset.visititems(self._visitor_func_merged)

        for clu in tqdm(self, desc='Clustering initializations'):
            _ = self[clu]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for dset in self.data:
            yield dset

    def __getitem__(self, path):
        assert(path in self.data), 'Sorry, did not find requested dataset.'
        if path in self.memo:
            return self.memo[path]
        else:
            self.memo[path] = Cluster(self.dset[path])
            return self.memo[path]

    def _visitor_func(self, name, node):
        # traverses the hdf file
        if isinstance(node, h5.Dataset):
            path = '/'.join(node.name.split('/')[:-1])
            self.data.add(path)

    def _visitor_func_merged(self, name, node):
        # traverses the hdf file if the hdf file links several hdfs
        if isinstance(node, h5.Group):
            if len(node.attrs) != 0:
                self.data.add(name)

    def close(self):
        """Closes the hdf file handle."""
        self.dset.close()

    def items(self):
        """Similiar to :code:`dict.items()`, iterator which returns the key
        of the cluster and the cluster itself.
        """
        for dset in self.data:
            yield dset, self[dset]

    @property
    def summary_table(self, **kwargs):
        """Returns a :obj:`pandas.DataFrame` with summary statistics about all
        clusters.
        """
        if hasattr(self, '_summary'):
            return self._summary

        data_frames = []

        for k, v in self.items():
            df = v.summary_table
            df[ID] = k
            data_frames.append(df)

        self._summary = pd.concat(data_frames)

        return self._summary


class Bootstrap(object):
    """Filter TensorSignature bootstrap samples and compute percentile based
    CIs.

    Args:
        initialization (:obj:`tensorsignatures.util.Initialization`): The
            initialization for which bootstrap samples were created.
        bootstrap (:obj:`tensorsignatures.util.Cluster`): The bootstrap
            cluster.
        cutoff (:obj:`float`, :math:`0<c<1`): Cutoff :math:`c` for the total
            variation distance that is used to filter bootstrap samples. CI is
            are based on the bootstrap samples passing the cutoff.
        lower (:obj:`int`): Lower percentile for CIs.
        upper (:obj:`int`): Upper percentile for CIs.
    Return:
        A :obj:`tensorsignatures.util.Bootstrap` object.
    """

    def __init__(self,
                 initialization,
                 bootstrap,
                 cutoff=0.1,
                 lower=5,
                 upper=95,
                 init=None):
        self.initialization = initialization
        self.bootstrap = bootstrap
        self.cutoff = cutoff
        self.lower = lower
        self.upper = upper

        # compute distances from each bootstrap sample to seed cluster
        Sref = self.initialization._Sc
        boot_dic = defaultdict(list)

        for i in range(len(self.bootstrap)):
            sample_i = self.bootstrap[i]
            r, c, d = assign_signatures(Sref, sample_i._Sc)
            # print(i,r,c)
            boot_dic['ref'].extend(r.tolist())
            boot_dic['boot'].extend(c.tolist())
            boot_dic['init'].extend([i] * self.initialization.rank)
            boot_dic['tvd'].extend(
                1 / 2 * np.linalg.norm(
                    Sref[..., r] - sample_i._Sc[..., c], ord=1, axis=0))

        # convert it to dataframe
        self.boot_df = pd.DataFrame(boot_dic)
        # initialize dictionaries which store boundaries
        self.valid_init = defaultdict(dict)
        self.intervals = defaultdict(list)

    def _filter(self, var, sig):
        # filters signatures
        if var not in self.valid_init[sig]:
            self.valid_init[sig][var] = list()

            valid_init = self.boot_df[
                (self.boot_df.tvd < self.cutoff) &
                (self.boot_df.ref == sig)].init.tolist()
            valid_sig = self.boot_df[
                (self.boot_df.tvd < self.cutoff) &
                (self.boot_df.ref == sig)].boot.tolist()

            for i, (idx, init) in enumerate(zip(valid_sig, valid_init)):
                # progress(i, len(valid_init), 'Sig {} ({})'.format(sig, init))
                if (var == 'E' or var == 'E0'):
                    values = np.zeros(self.clu.samples)
                    values[:] = np.nan
                    values[self.bootstrap['sub'][..., init].astype(int)] = \
                        self.bootstrap[var][idx, ..., init]

                    self.valid_init[sig][var].append(values)
                else:
                    self.valid_init[sig][var].append(
                        getattr(self.bootstrap[init], var)[..., idx, 0])

            # stack
            self.valid_init[sig][var] = np.stack(
                self.valid_init[sig][var], axis=-1)

        return self.valid_init[sig][var]

    def boundaries(self, var, sig=None):
        """Computes CI boundaries.

        Args:
            var (:obj:`str`): Parameter of interest (eg. 'S', '_b0')
            sig (:obj:`int`): Signature of interest, if :code:`None` boundaries
                yields an array containing the boundary values for signatures.
        Returns:
            An :obj:`array` containing the estimates for the CIs.
        """
        if sig is not None:
            if (var == 'E' or var == 'E0'):
                return np.stack([
                    np.nanpercentile(
                        self._filter(var, sig), self.lower, axis=-1),
                    np.nanpercentile(
                        self._filter(var, sig), self.upper, axis=-1)], axis=-1)

            return np.stack([
                np.percentile(self._filter(var, sig), self.lower, axis=-1),
                np.percentile(self._filter(var, sig), self.upper, axis=-1)],
                axis=-1)
        else:
            if var not in self.intervals:
                for i in range(self.initialization.rank):
                    if (var == 'E' or var == 'E0'):
                        self.intervals[var].append(
                            np.stack([
                                np.nanpercentile(
                                    self._filter(var, i),
                                    self.lower,
                                    axis=-1),
                                np.nanpercentile(
                                    self._filter(var, i),
                                    self.upper,
                                    axis=-1)],
                                axis=-1))
                    else:
                        self.intervals[var].append(
                            np.stack([
                                np.percentile(
                                    self._filter(var, i),
                                    self.lower,
                                    axis=-1),
                                np.percentile(
                                    self._filter(var, i),
                                    self.upper,
                                    axis=-1)],
                                axis=-1))

                self.intervals[var] = np.stack(self.intervals[var], axis=-2)

        return self.intervals[var]

    def yerr(self, var, func=lambda x: x):
        """Returns the yerrors (difference between the 5th and 95th percentile)
        and the inferred parameter of the intialization (MLE).

        Args:
            var (:obj:`var`): Parameter of interest (eg. 'S', '_b0')
            func (:obj:`func`): Method accepts a function, for example
                :code:`np.exp`, which transform CI before they are returned.
                This is necessary for some of the inferred parameters.
        Returns:
            A :obj:`array` containing an array with the errors of inferred
            parameters.

        Examples:

        Most importantly the :code:`yerr` method returns an array that can be
        passed as :code:`yerr` argument of :code:`matplotlib.pyplot.bar`.
        """
        yerr = np.zeros([*getattr(self.initialization, var)[..., 0].shape, 2])

        for i in range(self.initialization.rank):
            # to make it more readable
            if (var == 'E' or var == 'E0'):
                lower = self.boundaries(var)[..., i, 0]
                upper = self.boundaries(var)[..., i, 1]
                mle = getattr(self.initialization, var)[i, ..., 0]
            else:
                lower = self.boundaries(var)[..., i, 0]
                upper = self.boundaries(var)[..., i, 1]
                mle = getattr(self.initialization, var)[..., i, 0]

            # select indices
            bounded = (lower <= mle) & (mle <= upper)
            positive = mle > 0
            negative = mle < 0

            bp = np.where(bounded & positive)
            bn = np.where(bounded & negative)

            if (var == 'E' or var == 'E0'):
                yerr[i, ..., 1][bp] = func(upper[bp]) - func(mle[bp])
                yerr[i, ..., 0][bp] = func(mle[bp]) - func(lower[bp])
                yerr[i, ..., 1][bn] = func(upper[bn]) - func(mle[bn])
                yerr[i, ..., 0][bn] = func(mle[bn]) - func(lower[bn])

            else:
                yerr[..., i, 1][bp] = func(upper[bp]) - func(mle[bp])
                yerr[..., i, 0][bp] = func(mle[bp]) - func(lower[bp])
                yerr[..., i, 1][bn] = func(upper[bn]) - func(mle[bn])
                yerr[..., i, 0][bn] = func(mle[bn]) - func(lower[bn])

        return yerr


def save_dict(data, out_path):
    """Dumps a pickleable object to disk.

    Args:
        data (:obj:`dict`): A dictionary (containing pickleable values only).
        out_path (:obj:`str`): Path to the destination file.

    Returns:
        None
    """
    with open(out_path, 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    """Loads pickle dump.

    Args:
        path (:obj:`str`): Path to the pickle file.
    Returns:
        A :obj:`tuple` containing the filename and the pickled object.
    """

    with open(path, 'rb') as fh:
        params = pickle.load(fh)

    return (path.split('/')[-1], params)


def load_dump(path):
    """Loads a pickled tensorsignature initalization and returns it.

    Args:
        path (:obj:`str`): Path to the pickle file.
    Returns:
        A :obj:`tensorsignature.util.Initialization`.
    """
    fname, data = load_dict(path)
    dims = [key for key in list(data.keys()) if key.startswith('_k')]
    ki = {int(key[2:]): data[key] for key in dims}

    init = Initialization(S0=data[S0], a0=data[a0], b0=data[b0],
        ki=ki, m0=data[m0], T0=data[T0], E0=data[E0], rank=data[RANK],
        size=data[SIZE], objective=data[OBJECTIVE],
        starter_learning_rate=data[STARTER_LEARNING_RATE],
        decay_learning_rate=data[DECAY_LEARNING_RATE],
        optimizer=data[OPTIMIZER], epochs=data[EPOCHS],
        log_step=data[LOG_STEP], display_step=data[DISPLAY_STEP],
        observations=data[OBSERVATIONS], id=data[ID], init=data[INIT],
        seed=data[SEED], log_epochs=data[LOG_EPOCHS],
        log_learning_rate=data[LOG_LEARNING_RATE], log_L=data[LOG_L],
        log_L1=data[LOG_L1], log_L2=data[LOG_L2],
        sample_indices=data[SAMPLE_INDICES])

    return init


def collapse_data(snv):
    r"""Deprecated convinience function to collapse pyrimidine/purine
    dimension (snv.shape[-2])

    Args:
        snv (array-like, shape :math:`(3,3,(t_1+1),\dots,(t_l),2,p,n)`):
            SNV count tensor with distinct pyrimidine purine dimension.
    Returns:
        snv (array, shape :math:`(3, 3, (t_1+1), \dots, (t_l), p, n)`):
            Collapsed SNV array.
    """
    col1 = snv[[slice(None)] * (snv.ndim - 3) + [0] + [slice(None)] * 2]
    col2 = []
    dims = [
        (1, 1), (1, 0), (1, 2),
        (0, 1), (0, 0), (0, 2),
        (2, 1), (2, 0), (2, 2)]

    for i, j in dims:
        idx = [i, j] \
            + [slice(None)] \
            * (snv.ndim - 5) \
            + [1] \
            + [slice(None)] \
            * 2
        col2.append(snv[idx])

    col2 = np.stack(col2).reshape(col1.shape)

    return col1 + col2


def prepare_data(path, output):
    """Brings the datatensor obtained from processVcf.R into the right shape
    and adds the normalization constant to the hdf5 file.

    Args:
        path (:obj:`str`): Path to the file outputed from processVcf.R.
        output (:obj:`str`): Path to the output file.
    Returns:
        None
    """

    with h5.File(NORM, 'r') as fh:
        M = fh['M'][()]

    with h5.File(path, 'r') as fh:
        # load extracted snvs
        snv = fh["SNVR"][()].T.reshape(3, 3, 16, 4, 2, 2, 96, -1)

        # compute the normalization constant
        N0 = (snv.sum(axis=(4, 5, 6, 7)) / snv.sum()).reshape(3, 3, 16, 4, 1)
        N1 = np.concatenate(
            [N0, N0[[1, 0, 2], :, :][:, [1, 0, 2], :, :]], axis=4)
        N2 = N1.reshape(3, 3, 16, 4, 1, 2, 1, 1)
        N = (N2 * M) / 2

        # collapse data
        N = collapse_data(np.concatenate([N] * 2, axis=-4))
        snv = collapse_data(snv)

        # to be changed soon
        sv = np.zeros([81, snv.shape[-1]])
        sv[:] = np.nan
        other = np.concatenate(
            [fh['MNV'][()].T, fh['INDELS'][()].T, sv], axis=0)

    with h5.File(output, 'w') as fh:
        fh.create_dataset('SNV', data=snv)
        fh.create_dataset('OTHER', data=other)
        fh.create_dataset('N', data=N)

    return 0


def normalize_counts(init, N=None, collapse=False):
    normed_mutations = []

    if N is None:
        with h5.File(NORM, 'r') as fh:
            N = collapse_data(np.concatenate([fh["N"][()]] * 2, axis=-4))

    if collapse:
        N = ts.TensorSignature.collapse_data(N).reshape(3, 3, -1, 96, 1)

    for s in range(init.rank):
        snv_counts = (init.S[..., s, 0].reshape(-1, 1) @
                      init.E[s, ..., 0].reshape(1, -1)).reshape(
                      [*init.S.shape[:-2], init.E.shape[-2]])
        snv_counts *= N
        snv_counts = snv_counts.sum(
            axis=tuple([i for i in range(len(snv_counts.shape[:-1]))]))
        other_counts = init.T[..., s, 0].reshape(-1, 1) @ \
            init.E[s, ..., 0].reshape(1, -1)
        other_counts = other_counts.sum(axis=0)
        normed_mutations.append(snv_counts + other_counts)

    Enormed = np.stack(normed_mutations)

    return Enormed
