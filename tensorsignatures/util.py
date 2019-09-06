

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
from tqdm import tqdm


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
    """
    Compute the distance to a reference signature and assigns signatures
    based on their cosine similarity.
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
        S0 (obj:`array`): Fitted signature tensor.
        a0 (obj:`array`): Fitted signature amplitudes.
        b0 (obj:`array`): Fitted signature biases.
        k0 (obj:`array`): Fitted signature activities.
        m0 (obj:`array`): Fitted signature mixing factors.
        T0 (obj:`array`): Fitted other mutation types signature matrix.
        E0 (obj:`array`): Fitted exposures.
        rank (obj:`int`): Rank of decomposition.
        dispersion (obj:`int`): Dispersion
        objective (:obj:`str`): Used likelihood function.
        starter_learning_rate (:obj:`float`): Starter learning rate.
        decay_learning_rate (:obj:`float`): Decay of learning rate.
        optimizer (:obj:`str`): Used optimizer.
        epochs (obj:`int`): Number of training epochs.
        log_step (obj:`int`): Logging steps.
        display_step (obj:`int`): Diplay steps.
        observations (obj:`int`): Number of observations (non NA entries).
        seed (obj:`int`): Used seed.
    Returns:
        A :obj:`TensorSingatureInit` object.
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

        for key, value in self._ki.items():
            setattr(self, 'k' + str(key), np.exp(value))

        self._m0 = self._add_iterdim(m0)
        self._T0 = self._add_iterdim(T0)
        self._E0 = self._add_iterdim(E0)

        self.log_epochs = self._add_iterdim(log_epochs)
        self.log_learning_rate = self._add_iterdim(log_learning_rate)
        self.log_L = self._add_iterdim(log_L)
        self.log_L1 = self._add_iterdim(log_L1)
        self.log_L2 = self._add_iterdim(log_L2)
        self.sample_indices = self._add_iterdim(sample_indices)

    def __getitem__(self, item):
        if hasattr(self, str(item)):
            return getattr(self, str(item))
        if hasattr(self, '_' + str(item)):
            return getattr(self, '_' + str(item))

        raise KeyError('key not found')

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
        S0 = np.exp(self._S0) \
            / np.sum(np.exp(self._S0), axis=2, keepdims=True)

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
        T1 = np.concatenate(
            [self._T0, np.zeros((1, self.rank, self.iter))], axis=0)
        T1 = np.exp(self._T1) \
            / np.sum(np.exp(self._T1), axis=0, keepdims=True)

        return T1

    @lazy_property
    def S(self):
        S = self._S1 \
            * self._B \
            * self._A.reshape(3, 3, 1, self.rank, self.iter) \
            * self._m.reshape(1, 1, 1, self.rank, self.iter)

        return S

    @lazy_property
    def T(self):
        return self._T1 * (1 - self._m)

    @lazy_property
    def a(self):
        return np.exp(self._a0)

    @lazy_property
    def b(self):
        return np.exp(self._b0)

    @lazy_property
    def m(self):
        return 1 / (1 + np.exp(-self._m0))

    def to_dic(self):
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
        data = self.to_dic()
        save_dict(data, path)


class Cluster(Initialization):
    """
    Base cluster class takes

    """
    def __init__(self, dset, **kwargs):
        self.dset = dset
        self.memo = {}
        self.seed = np.argmax(
            np.ma.array(
                self.dset[LOG_L][()][-1, :],
                mask=self.dset[LOG_L][()][-1, :] >= 0))

        # cluster init
        self._S0, self._T0, self._E0, self.icol = Cluster.cluster_signatures(
            dset[S0], dset[T0], dset[E0], self.seed)

        self.iter = self._S0.shape[-1]
        self.rank = self._S0.shape[-2]
        self.samples = self._E0.shape[-2]

        # initialize variables
        self._a0 = self[a0]
        self._b0 = self[b0]

        for key in [var for var in list(self.dset) if var.startswith('_k')]:
            setattr(self, key[1:], np.exp(self[key]))
            #self.memo[key] = getattr(self, key)

        self._m0 = self[m0]

        # compute composite variables
        self.S
        self.T

    def __len__(self):
        return self.iter

    def __iter__(self):
        for i in range(self.iter):
            yield i

    def __getitem__(self, item):
        if item in self.memo:
            return self.memo[item]
        elif item in list(self.dset):
            if item in VARS:
                return self.sort_array(item, self.dset[item][()])
            elif item.startswith('k'):
                return self.sort_array(item, self.dset[item][()])
            elif item in LOGS:
                return self.dset[item][()][..., list(self.icol.keys())]
            else:
                return self.dset[item][()]

        if item in list(self.dset.attrs):
            return self.dset.attrs[item]

        raise KeyError('Could not find item.')

    def __contains__(self, item):
        if (item in list(self.dset)) or (item in list(self.dset.attrs)):
            return True
        else:
            False

    def as_list(self, key, sign=1):
        if key in self:
            if type(self[key]) == np.ndarray:
                return (sign * self[key]).tolist()

        return [np.nan] * self.iter

    def sort_array(self, var, array):
        # sort array
        var_list = []

        for k, v in self.icol.items():
            var_list.append(array[..., v, k])

        self.memo[var] = np.stack(var_list, axis=array.ndim - 1)

        return self.memo[var]

    @staticmethod
    def cluster_signatures(S, T, E, seed=None):
        """
        Expects that iterations are found in the last axis, and signatures in
        the penultimate axis.
        """

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
        p = 4 * 95
        p += self[a0].shape[0] if a0 in self else 0
        p += self[b0].shape[0] if b0 in self else 0

        p += self['k0'].shape[0] if 'k0' in self else 0
        p += self['k1'].shape[0] if 'k1' in self else 0
        p += self['k2'].shape[0] if 'k2' in self else 0
        p += self['k3'].shape[0] if 'k3' in self else 0

        p += 1 if m0 in self else 0
        p += self[T0].shape[0] if T0 in self else 0

        p = p * self.rank
        p += self.samples * self.rank

        return p

    @property
    def observations(self):
        return self[OBSERVATIONS]

    @property
    def likelihood(self):
        return self[LOG_L][-1, :]

    @property
    def size(self):
        if not hasattr(self, '_size'):
            self._size = self[SIZE]

        return self._size

    @lazy_property
    def init(self):
        """Returns the maximum likelihood initialisation.
        """
        return np.argmax(self.likelihood)

    def get_init(self, init=None):
        """Returns initialization, if None :code:`get_init` returns the
        initialization with the largest log likelihood.
        """
        if init is None:
            init = self.init

        k = {}
        for key in [var for var in list(self.dset) if var.startswith('k')]:
            k[int(key[1])] = self.dset[key][..., init]

        initialization = Initialization(
            S0=self.dset[S0][..., init],
            a0=self.dset[a0][..., init],
            b0=self.dset[b0][..., init],
            ki=k,
            m0=self.dset[m0][..., init],
            T0=self.dset[T0][..., init],
            E0=self.dset[E0][..., init],
            rank=self[RANK],
            size=self[SIZE],
            objective=self[OBJECTIVE],
            epochs=self[EPOCHS],
            starter_learning_rate=self[STARTER_LEARNING_RATE],
            decay_learning_rate=self[DECAY_LEARNING_RATE],
            optimizer=self[OPTIMIZER],
            log_step=self[LOG_STEP],
            display_step=self[DISPLAY_STEP],
            observations=self[OBSERVATIONS],
            id=self[ID],
            init=init,
            seed=self[SEED],
            log_epochs=self.dset[LOG_EPOCHS][..., init],
            log_learning_rate=self.dset[LOG_LEARNING_RATE][..., init],
            log_L=self.dset[LOG_L][..., init],
            log_L1=self.dset[LOG_L1][..., init],
            log_L2=self.dset[LOG_L2][..., init],
            sample_indices=self.dset[SAMPLE_INDICES][..., init]
        )

        return initialization

    @property
    def summary_table(self):
        if not hasattr(self, '_summary'):
            df = pd.DataFrame({
                LOG_L1: self[LOG_L1][-1, list(self.icol.keys())].tolist(),
                LOG_L2: self[LOG_L2][-1, list(self.icol.keys())].tolist(),
                LOG_L: self.likelihood.tolist(),
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

            self._summary = df

        return self._summary

    def coefficient_table(self, cdim='b0', avg=False):
        """
        Returns a panda data frame with inferred coefficients for each signature
        and initialisation of the cluster.

        Parameters:
        cdim (string): name of the coefficient (eg. "b0", "a0" etc.)
        avg (boolean): returns computes the average of fitted coefficient over initialisations
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

    def normalize_counts(self, N, init=None, collapse=True):
        if init is None:
            init = self.init

        normed_mutations = []
        if collapse:
            N = TensorSignature.collapse_data(N).reshape(3, 3, -1, 96, 1)
        for s in range(self.rank):
            snv_counts = (self.S[..., s, init].reshape(-1, 1) @ self.E[s, ..., init].reshape(1,-1)).reshape([*self.S.shape[:-2], self.E.shape[-2]]) * N
            snv_counts = snv_counts.sum(axis=(0,1,2,3))
            other_counts = self.T[..., s, init].reshape(-1,1) @ self.E[s, ..., init].reshape(1,-1)
            other_counts = other_counts.sum(axis=0)
            normed_mutations.append(snv_counts+other_counts)

        Enormed = np.stack(normed_mutations)

        return Enormed


class Experiment(object):

    def __init__(self, path, pre_cluster=True):
        """
        Experiment class
        Experiment loads datasets dynamically.
        """
        self.dset = h5.File(path)
        self.data = set([])
        self.memo = dict()

        # walk through all experiment params
        self.dset.visititems(self.__visitor_func)

        if len(self.data) == 0:
            self.dset.visititems(self.__visitor_func_merged)

        if pre_cluster:
            for clu in tqdm(self):
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

    def __visitor_func(self, name, node):
        if isinstance(node, h5.Dataset):
            path = '/'.join(node.name.split('/')[:-1])
            self.data.add(path)

    def __visitor_func_merged(self, name, node):
        if isinstance(node, h5.Group):
            if len(node.attrs) != 0:
                self.data.add(name)

    def close(self):
        self.dset.close()

    def items(self):
        for dset in self.data:
            yield dset, self[dset]

    @property
    def summary_table(self, **kwargs):
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
    """
    Filter TensorSignature bootstrap samples and compute percentile based CIs.

    Params:
    """

    def __init__(self,
                 clu,
                 bootstrap,
                 cutoff=0.1,
                 lower=5,
                 cores=8,
                 upper=95,
                 init=None):
        self.clu = clu
        self.bootstrap = bootstrap
        self.cutoff = cutoff
        self.lower = lower
        self.upper = upper
        self.cores = cores
        if init is None:
            self.init = self.clu.init
        else:
            self.init = init

        # compute distances from each bootstrap sample to seed cluster
        boot_dic = defaultdict(list)
        for i in range(self.bootstrap['S'].shape[-1]):
            progress(i, self.bootstrap['S'].shape[-1], 'Iteration {}'.format(i))
            Sref = np.concatenate([self.clu.S[2, 2, 0, ..., self.init].reshape(-1, self.clu.rank), self.clu.T[..., self.init]])
            Sbot = np.concatenate([self.bootstrap['S'][2,2,0,...,i].reshape(-1, self.clu.rank), self.bootstrap['T'][...,i]])

            r, c, d = assign_signatures(Sref, Sbot)
            boot_dic['ref'].extend(r.tolist())
            boot_dic['boot'].extend(c.tolist())
            boot_dic['init'].extend([i]*len(d))
            boot_dic['tvd'].extend((1/2*np.linalg.norm(Sref-Sbot,ord=1,axis=0))[c])

        # convert it to dataframe
        self.boot_df = pd.DataFrame(boot_dic)
        # initialize dictionaries which store boundaries
        self.valid_init = defaultdict(dict)
        self.intervals = defaultdict(list)

    def _filter(self, var, sig):
        if var not in self.valid_init[sig]:
            self.valid_init[sig][var] = list()

            valid_init = self.boot_df[(self.boot_df.tvd < self.cutoff) & (self.boot_df.ref == sig)].init.tolist()
            valid_sig = self.boot_df[(self.boot_df.tvd < self.cutoff) & (self.boot_df.ref == sig)].boot.tolist()

            for i, (idx, init) in enumerate(zip(valid_sig, valid_init)):
                progress(i, len(valid_init), 'Sig {} ({})'.format(sig, init))
                if (var=='E') or (var=='E0'):
                    values = np.zeros(self.clu.samples)
                    values[:] = np.nan
                    values[self.bootstrap['sub'][..., init].astype(int)] = self.bootstrap[var][idx, ..., init]

                    self.valid_init[sig][var].append(values)
                else:
                    self.valid_init[sig][var].append(self.bootstrap[var][..., idx, init])

            # stack
            self.valid_init[sig][var] = np.stack(self.valid_init[sig][var], axis=-1)

        return self.valid_init[sig][var]

    def boundaries(self, var, sig=None):
        if sig is not None:
            if (var == 'E' or var == 'E0'):
                return np.stack([
                    np.nanpercentile(self._filter(var, sig), self.lower, axis=-1),
                    np.nanpercentile(self._filter(var, sig), self.upper, axis=-1)], axis=-1)

            return np.stack([np.percentile(self._filter(var, sig), self.lower, axis=-1),
                             np.percentile(self._filter(var, sig), self.upper, axis=-1)], axis=-1)
        else:
            if var not in self.intervals:
                for i in range(self.clu.rank):
                    if (var == 'E' or var == 'E0'):
                        self.intervals[var].append(
                            np.stack([
                                np.nanpercentile(self._filter(var, i), self.lower, axis=-1),
                                np.nanpercentile(self._filter(var, i), self.upper, axis=-1)], axis=-1))
                    else:
                        self.intervals[var].append(
                            np.stack([
                                np.percentile(self._filter(var, i), self.lower, axis=-1),
                                np.percentile(self._filter(var, i), self.upper, axis=-1)], axis=-1))
                self.intervals[var] = np.stack(self.intervals[var], axis=-2)

        return self.intervals[var]

    def yerr(self, var, func=lambda x: x):
        yerr = np.zeros([*self.clu[var][..., self.init].shape, 2])

        for i in range(self.clu.rank):
            # to make it more readable
            if (var == 'E' or var == 'E0'):
                lower = self.boundaries(var)[..., i, 0]
                upper = self.boundaries(var)[..., i, 1]
                mle = self.clu[var][i, ..., self.init]
            else:
                lower = self.boundaries(var)[..., i, 0]
                upper = self.boundaries(var)[..., i, 1]
                mle = self.clu[var][..., i, self.init]

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


def create_df(arr):
    d1, d2 = arr.shape
    print (d1, d2)
    df = pd.DataFrame({
        'signature': np.arange(d2).tolist() * d1,
        'val': arr.reshape(-1).tolist(),
        'dim': np.array([ [i]*d2 for i in range(d1) ]).reshape(-1).tolist()
    })

    return df


def ksstat(LT1, LT2):
    """
    computes the waterstein distance
    """

    left_tail = np.concatenate([LT1.reshape(-1), LT2.reshape(-1)])
    left_tail = left_tail[~np.isnan(left_tail)]
    return kstest(left_tail, 'uniform').statistic

def compute_left_tails(C, Chat, k, inv_norm=False):
    u = uniform.rvs(size=np.prod(C.shape)).reshape(C.shape)
    left_tail = u * nbinom.cdf(C, k, k/(Chat + k)) + (1-u) * nbinom.cdf(C-1, k, k/(Chat + k))

    if inv_norm:
        left_tail = norm.ppf(left_tail)

    return left_tail


def save_dict(data, out_path):
    with open(out_path, 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)


def load_dict(data):
    with open(data, 'rb') as fh:
        params = pickle.load(fh)

    return (data.split('/')[-1], params)


def load_dump(path):
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
    col1 = snv[[slice(None)]*(snv.ndim-3)+[0]+[slice(None)]*2]
    col2 = []
    for i, j in [(1, 1), (1, 0), (1, 2), (0, 1), (0, 0), (0, 2), (2, 1), (2, 0), (2, 2)]:
        col2.append(snv[[i, j]+[slice(None)]*(snv.ndim-5)+[1]+[slice(None)]*2])
    col2 = np.stack(col2).reshape(col1.shape)
    return col1 + col2
