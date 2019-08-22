

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


#from sklearn.metrics import silhouette_score
#from sklearn.metrics import silhouette_samples
#from sklearn.decomposition import NMF
from scipy.stats import nbinom
from scipy.stats import poisson
from scipy.stats import kstest
from scipy.stats import uniform
from collections import defaultdict
from multiprocessing import Pool
import pickle
import numpy as np
import pandas as pd
import h5py as h5
import os
import sys
import re
from tensorsignatures.config import *
from tensorsignatures.tensorsignatures import TensorSignature
#import seaborn as sns

def reshape_clustered_signatures(S):
    return np.moveaxis(S, [-1, -2], [0, 1]).reshape(S.shape[-1] * S.shape[-2], -1)

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

class Cluster(object):
    """
    Base cluster class takes 

    """
    def __init__(self, dset, **kwargs):
        self.dset = dset
        self.memo = {}
        self.seed = np.argmax(np.ma.array(self.dset['L'][()], mask=self.dset['L'][()]>=0))
        
        # cluster init
        self.S, self.T, self.E, self.icol = Cluster.cluster_signatures(
            dset['S'], dset['T'], dset['E'], self.seed)
        
        self.iter = self.S.shape[-1]
        self.rank = self.S.shape[-2]
        self.samples = self.E.shape[-2]

    def __len__(self):
        return self.iter

    def __iter__(self):
        for i in range(self.iter):
            yield i

    def __getitem__(self, item):
        if item in self.memo:
            return self.memo[item]
        elif item in list(self.dset):
            if item in ['T', 'S0', 'T0', 'a0', 'b0', 'm0', 'm1', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5']:
                return self.__sort_array(item, self.dset[item][()])
            elif item in ['L', 'L1', 'L2', 'tau', 'data_pts', 'log_L1', 'log_L2', 'log_learning_rate']:
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

    def __get_as_list(self, key, sign=1): 
        if key in self:
            if type(self[key]) == np.ndarray:
                return (sign * self[key]).tolist()
        
        return [np.nan] * self.iter

    def __sort_array(self, var, array):
        var_list = []
        
        for k, v in self.icol.items():
            var_list.append(array[... , v, k])
            
        self.memo[var] = np.stack(var_list, axis=array.ndim-1)
        
        return self.memo[var]

    @staticmethod
    def pre_cluster_signatures(p, S, T, E, I):
        S_clu, T_clu, E_clu, i_col = Cluster.cluster_signatures(S, T, E, I)        
        return (p, S_clu, T_clu, E_clu, i_col, None)
    
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

        if np.any(np.isnan(S_seed)) | np.any(np.isinf(S_seed)) | np.all(S_seed == 0):
            print("Warning: seed {} corrupted.".format(seed))
            return (None, None, None, None)

        if np.any(np.isnan(T_seed)) | np.any(np.isinf(T_seed)) | np.all(T_seed == 0):
            print("Warning: seed {} corrupted.".format(seed))
            return (None, None, None, None)    

        S_shape = S_seed.shape
        S_seed = np.concatenate([
            S_seed[2,2,0].reshape(-1, S_seed.shape[-1]),
            T_seed])

        S_list, T_list, E_list = [], [], []
        i_col = {}

        for i in range(S.shape[-1]):
            S_i = S[..., i]
            T_i = T[..., i]

            if np.any(np.isnan(S_i)) | np.any(np.isinf(S_i)) | np.all(S_i == 0):
                continue

            if np.any(np.isnan(T_i)) | np.any(np.isinf(T_i)) | np.all(T_i == 0):
                continue

            S_i = np.concatenate([
                S_i[2,2,0].reshape(-1, S_i.shape[-1]),
                T_i])
            E_i = E[..., i]

            ridx, cidx, _ = assign_signatures(S_seed, S_i)

            S_list.append(S[..., i][..., cidx])
            T_list.append(T[..., i][..., cidx])
            E_list.append(E[..., i][cidx, :])

            i_col[i] = cidx

        S_clu = np.stack(S_list, axis=S.ndim-1)
        T_clu = np.stack(T_list, axis=T.ndim-1)
        E_clu = np.stack(E_list, axis=2)

        return (S_clu, T_clu, E_clu, i_col)

    @property
    def parameters(self):
        if hasattr(self, '_params'):
            return self._params

        p = 4*95 # basic signature spectra
        p += self['a0'].shape[0] if 'a0' in self else 0 # biases for replication
        p += self['b0'].shape[0] if 'b0' in self else 0 # biases for transcription

        p += self['k0'].shape[0] if 'k0' in self else 0
        p += self['k1'].shape[0] if 'k1' in self else 0
        p += self['k2'].shape[0] if 'k2' in self else 0
        p += self['k3'].shape[0] if 'k3' in self else 0

        p += 1 if 'm1' in self else 0 # mixing factor
        p += self['T'].shape[0]-1 if 'T' in self else 0 # other covariates

        p = p * self.rank # multiply times number of signatures
        p += self.E.shape[-2] * self.rank # 2778 * x['rank'] # all fitted exposures

        self._params = p

        return self._params

    @property
    def likelihood(self):
        return self['L']

    @property
    def size(self):
        if not hasattr(self, '_size'):
            self._size = self['tau']

        return self._size

    @property
    def init(self):
        """
        Returns the maximum likelihood initialisation. 
        """
        return np.argmax(self.likelihood)

    @property
    def summary_table(self):
        if not hasattr(self, '_summary'):
            df = pd.DataFrame({
                'L1': self.__get_as_list('L1'),    
                'L2': self.__get_as_list('L2'),
                'L3': self.__get_as_list('L3'),
                'L': self.likelihood, #self.__get_as_list('L', -1),
                'size': self.__get_as_list('tau'),
                'rank': [self.rank] * self.iter,
                'init': np.arange(0, self.iter),
                'k': [self.parameters] * self.iter,    
                'n': list(map(int, self.__get_as_list('data_pts'))),    
                })
            df[AIC] = 2 * df['k'] - 2 * df['L']
            df[AIC_C] = df[AIC] + (2*df['k']**2 + 2 * df['k']) / (df['n'] - df['k'] - 1)
            df[BIC] = np.log(df['n']) * df['k'] - 2*df['L'] 

            self._summary = df
        
        return self._summary

    def log_table(self, resolution=100, skip_front=20, skip_back=200):
    
        end = self[EPOCHS] - skip_back * resolution
        
        df = pd.DataFrame({   
            EPOCHS : np.arange(resolution * skip_front, end, resolution).tolist() * self.iter,
            'init': np.array([[i] * int((end-resolution*skip_front)/resolution) for i in range(self.iter)]).reshape(-1),
            'L1': self['log_objective'][::resolution, :][skip_front:-skip_back,:].T.reshape(-1),
            'lrate': self['log_learning_rate'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
            'lr': self['log_lambda_r'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
            'lt': self['log_lambda_t'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
            'lc': self['log_lambda_c'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
            'la': self['log_lambda_a'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
            })
        
        return df

    def coefficient_table(self, cdim='b0', avg=False):
        """
        Returns a panda data frame with inferred coefficients for each signature 
        and initialisation of the cluster.
        
        Parameters:
        cdim (string): name of the coefficient (eg. "b0", "a0" etc.)
        avg (boolean): returns computes the average of fitted coefficient over initialisations
        """
        coeff_table = pd.DataFrame({
            'sig': np.array([[i] * self[cdim].shape[0] for i in range(self.rank)]).reshape(-1).tolist() * self.iter, 
            'dim': np.arange(self[cdim].shape[0]).tolist() * self[cdim].shape[1] * self[cdim].shape[2],
            'init': np.array([[i] * self.rank * self[cdim].shape[0] for i in range(self.iter)]).reshape(-1).tolist(),
            'val': self[cdim].T.reshape(-1).tolist()})

        if avg:
            coeff_table = coeff_table.groupby(['sig', 'dim']).agg({'val':[np.mean, np.std]}).reset_index()
            coeff_table.columns = [ ' '.join(col).strip() for col in coeff_table.columns ]

        return coeff_table

    def plot_signature(self, init=None, **kwargs):
        """
        Plots the signature spectra. If no integer for the initialisation
        is given the method selects the initialisation for which the likelihood        
        was maximised.
        """
        if init is None:
            init = self.init
        
        ax = plot_collapsed_signature(self.S[..., init], **kwargs)
        return ax

    def plot_big_spectra(self, init=None, **kwargs):
        if init is None:
            init = self.init

        T = self['T']
        plot_big_spectra(T[..., init], **kwargs)

    def normalize_counts(self, N, init=None, collapse=True):
        if init is None:
            init = self.init

        normed_mutations = []
        if collapse:
            N = TensorSignature.collapse_data(N).reshape(3,3,-1,96,1)
        for s in range(self.rank):
            snv_counts = (self.S[..., s, init].reshape(-1, 1) @ self.E[s, ..., init].reshape(1,-1)).reshape([*self.S.shape[:-2], self.E.shape[-2]]) * N
            snv_counts = snv_counts.sum(axis=(0,1,2,3))
            other_counts = self.T[..., s, init].reshape(-1,1) @ self.E[s, ..., init].reshape(1,-1)
            other_counts = other_counts.sum(axis=0)
            normed_mutations.append(snv_counts+other_counts)

        Enormed = np.stack(normed_mutations)
        
        return Enormed

class PreCluster(Cluster):
    def __init__(self, dset, S, T, E, icol, **kwargs):
        self.dset = dset
        self.memo = {}
        self.S = S
        self.T = T
        self.E = E
        self.icol = icol
        
        # added fast cluster option which sets the seed to -1
        self.seed = np.argmax(np.ma.array(self.dset['L'][()], mask=self.dset['L'][()]>=0))        
        self.iter = self.S.shape[-1]
        self.rank = self.S.shape[-2]
        self.samples = self.E.shape[-2]

class Singleton(Cluster):
    def __init__(self, dset, **kwargs):
        self.dset = Singleton.load_dict(dset)[1]
        self.S = self.dset['S'].reshape(*(*self.dset['S'].shape, 1))
        self.T = self.dset['T'].reshape(*(*self.dset['T'].shape, 1))
        self.E = self.dset['E'].reshape(*(*self.dset['E'].shape, 1))

        self.iter = self.S.shape[-1]
        self.rank = self.S.shape[-2]
        self.samples = self.E.shape[-2]                
    
    def __getitem__(self, item):
        if item in self.dset:
            if item in ['S', 'T', 'E', 'S0', 'T0', 'E0', 'a0', 'b0', 'm0', 'm1', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'tau']:
                return self.dset[item].reshape(*(*self.dset[item].shape, 1))
            else:
                return self.dset[item]
        else:
            raise KeyError('Could not find item.')
        
    @staticmethod
    def load_dict(data):
        with open(data, 'rb') as fh:
            params = pickle.load(fh)
        
        return (data.split('/')[-1], params)

class Experiment(object):

    def __init__(self, path, cores=8):
        """
        Experiment class 
        Experiment loads datasets dynamically.
        """
        self.dset = h5.File(path)
        self.data = set([])
        self.memo = dict()
        self.cores = cores

        # walk through all experiment params
        self.dset.visititems(self.__visitor_func)
        
        if len(self.data) == 0:
            self.dset.visititems(self.__visitor_func_merged)
        
        #self.__cluster()
            
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
    
    def cluster(self):
        """
        Uses the initializsation with largest likelihood as seed, and does not
        iterate over all initializations.
        """
        assert(self.cores > 0)
        data_sets = []

        for i, dset in enumerate(self.data):
            progress(i, len(self), dset)
            S = self.dset[dset]['S'][()]
            T = self.dset[dset]['T'][()]
            E = self.dset[dset]['E'][()]
            L = self.dset[dset]['L'][()]
            I = np.argmax(np.ma.array(L, mask=L >= 0))
            data_sets.append((dset, S, T, E, I))    
            
        if self.cores > 1:
            pool = Pool(self.cores)
            results = pool.starmap(Cluster.pre_cluster_signatures, data_sets) 
            pool.close()
        
        else:
            results = []
            for dset in data_sets:
                results.append(Cluster.pre_cluster_signatures(*dset))

        for r in results:
            self.memo[r[0]] = PreCluster(self.dset[r[0]], r[1], r[2], r[3], r[4])
            
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
            df[JOB_NAME] = k
            data_frames.append(df)

        self._summary = pd.concat(data_frames)

        return self._summary

class Bootstrap(object):
    """
    Filter TensorSignature bootstrap samples and compute percentile based CIs.

    Params:
    """

    def __init__(self, clu, bootstrap, cutoff=0.1, lower=5, cores=8, upper=95, init=None):
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
            if (var=='E') or (var=='E0'):
                return np.stack([np.nanpercentile(self._filter(var, sig), self.lower, axis=-1),
                                 np.nanpercentile(self._filter(var, sig), self.upper, axis=-1)], axis=-1)
            
            return np.stack([np.percentile(self._filter(var, sig), self.lower, axis=-1),
                             np.percentile(self._filter(var, sig), self.upper, axis=-1)], axis=-1)
        else:
            if var not in self.intervals:
                for i in range(self.clu.rank):
                    if (var=='E') or (var=='E0'):
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
        #print(yerr.shape)
        for i in range(self.clu.rank):
            # to make it more readable
            if (var=='E') or (var=='E0'):
                lower = self.boundaries(var)[...,i, 0]
                upper = self.boundaries(var)[...,i, 1]
                mle = self.clu[var][i,..., self.init]                
            else:
                lower = self.boundaries(var)[...,i,0]
                upper = self.boundaries(var)[...,i,1]
                mle = self.clu[var][..., i, self.init]
                
            #print(lower.shape,upper.shape,mle.shape)
            # select indices
            bounded = (lower <= mle) & (mle <= upper)
            positive = mle > 0
            negative = mle < 0

            bp = np.where(bounded & positive)
            bn = np.where(bounded & negative)
            
            if (var=='E') or (var=='E0'):
                yerr[i,...,1][bp] = func(upper[bp]) - func(mle[bp])
                yerr[i,...,0][bp] = func(mle[bp]) - func(lower[bp])
                yerr[i,...,1][bn] = func(upper[bn]) - func(mle[bn])
                yerr[i,...,0][bn] = func(mle[bn]) - func(lower[bn])

            else:
                yerr[...,i,1][bp] = func(upper[bp]) - func(mle[bp])
                yerr[...,i,0][bp] = func(mle[bp]) - func(lower[bp])
                yerr[...,i,1][bn] = func(upper[bn]) - func(mle[bn])
                yerr[...,i,0][bn] = func(mle[bn]) - func(lower[bn])

        return yerr


# def cluster_signatures(S, E, seed=None):
#     """
#     Expects that iterations are found in the last axis, and signatures in
#     the penultimate axis.
#     """

#     if seed is None:
#         seed = np.random.randint(S.shape[-1])

#     S_seed = S[..., seed]
#     if np.any(np.isnan(S_seed)) | np.any(np.isinf(S_seed)) | np.all(S_seed == 0):
#         print("Warning: seed {} corrupted.".format(seed))
#         return None, None, None
    
#     S_shape = S_seed.shape
#     S_seed = S_seed.reshape(-1, S_seed.shape[-1])
    
#     S_list, E_list = [], []
    
#     i_col = {}

#     for i in range(S.shape[-1]):
#         S_i = S[..., i]
        
#         if np.any(np.isnan(S_i)) | np.any(np.isinf(S_i)) | np.all(S_i == 0):
#             continue
        
#         S_i = S_i.reshape(-1, S_i.shape[-1])
#         E_i = E[..., i]

#         ridx, cidx, _ = assign_signatures(S_seed, S_i)
        
#         S_i = S_i.reshape(S_shape)
#         #S_clu[..., i] = S_i[..., cidx]
#         #E_clu[..., i] = E_i[cidx, :,]
#         S_list.append(S_i[..., cidx])
#         E_list.append(E_i[cidx, :])
        
#         i_col[i] = cidx
        
#     S_clu = np.stack(S_list, axis=S.ndim-1)
#     E_clu = np.stack(E_list, axis=2)

#     return (S_clu, E_clu, i_col)

# def silhouette_signatures(S, sample=False):
#     labels = [i for i in range(S.shape[-2])] * S.shape[-1]
#     S_reshaped = reshape_clustered_signatures(S) #np.moveaxis(S, [-1, -2], [0, 1]).reshape(S.shape[-1] * S.shape[-2], -1)
    
#     if sample:
#         return silhouette_samples(S_reshaped, labels, metric='cosine')

#     return silhouette_score(S_reshaped, labels, metric='cosine')

# def mp_cluster(p, S, E):
#     #print('Start working on {}'.format(p))
#     S_clu = None
#     E_clu = None
#     i_col = None

#     scores = np.zeros(S.shape[-1])
    
#     for seed in range(S.shape[-1]):
#         S_i, E_i, col_i = cluster_signatures(S, E, seed)
#         if (S_i is None) & (E_i is None):
#             scores[seed] = -np.inf
#         else:
#             scores[seed] = silhouette_signatures(S_i)

#         if np.argmax(scores) == seed:
            
#             S_clu = S_i
#             E_clu = E_i
#             i_col = col_i

#     return (p, S_clu, E_clu, i_col, scores)

# def mp_fast_cluster(p, S, E, I):
#     #print('Start working on {}'.format(p))
#     S_clu, E_clu, i_col = cluster_signatures(S, E, I)

#     return (p, S_clu, E_clu, i_col, None)

# class Cluster(object):
#     def __init__(self, dset, **kwargs):
#         self.dset = dset
#         self.memo = {}
#         self.scores = np.zeros(dset['S'].shape[-1])
        
#         for seed in range(dset['S'].shape[-1]):
#             S_clu, E_clu, _ = cluster_signatures(dset['S'], dset['E'], seed)
#             if (S_clu is None) & (E_clu is None):
#                 self.scores[seed] = -np.inf
#             else:
#                 self.scores[seed] = silhouette_signatures(S_clu)

#         self.seed = np.argmax(self.scores)
#         self.S, self.E, self.icol = cluster_signatures(dset['S'], dset['E'], self.seed)
        
#         self.iter = self.S.shape[-1]
#         self.rank = self.S.shape[-2]
#         self.samples = self.E.shape[-2]

#     def __len__(self):
#         return self.iter

#     def __iter__(self):
#         for i in range(self.iter):
#             yield i

#     def __getitem__(self, item):
#         if item in self.memo:
#             return self.memo[item]
#         elif item in list(self.dset):
#             if item in ['T', 'a0', 'b0', 'm0', 'm1', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5']:
#                 return self.__sort_array(item, self.dset[item][()])
#             elif item in ['L', 'L1', 'L2', 'tau', 'data_pts', 'log_L1', 'log_L2', 'log_learning_rate']:
#                 return self.dset[item][()][..., list(self.icol.keys())]
#             else:
#                 return self.dset[item][()]

        
#         if item in list(self.dset.attrs):
#             return self.dset.attrs[item]
        
#         raise KeyError('Could not find item.')

#     def __contains__(self, item):
#         if (item in list(self.dset)) or (item in list(self.dset.attrs)):
#             return True
#         else:
#             False

#     def __get_as_list(self, key, sign=1): 
#         if key in self:
#             if type(self[key]) == np.ndarray:
#                 return (sign * self[key]).tolist()
        
#         return [np.nan] * self.iter
#         #return (sign * self[key]).tolist() if (key in self) & (type(self[key]) == np.ndarray) else [np.nan] * self.iter

#     def __sort_array(self, var, array):
#         var_list = []
        
#         for k, v in self.icol.items():
#             var_list.append(array[... , v, k])
            
#         self.memo[var] = np.stack(var_list, axis=array.ndim-1)
        
#         return self.memo[var]

#     @property
#     def parameters(self):
#         if hasattr(self, '_params'):
#             return self._params

#         p = 4*95 # basic signature spectra
#         p += self['a0'].shape[0] if 'a0' in self else 0 # biases for replication
#         p += self['b0'].shape[0] if 'b0' in self else 0 # biases for transcription

#         p += self['k0'].shape[0] if 'k0' in self else 0
#         p += self['k1'].shape[0] if 'k1' in self else 0
#         p += self['k2'].shape[0] if 'k2' in self else 0
#         p += self['k3'].shape[0] if 'k3' in self else 0

#         p += 1 if 'm1' in self else 0 # mixing factor
#         p += self['T'].shape[0]-1 if 'T' in self else 0 # other covariates

#         p = p * self.rank # multiply times number of signatures
#         p += self.E.shape[-2] * self.rank # 2778 * x['rank'] # all fitted exposures

#         self._params = p

#         return self._params

#     @property
#     def likelihood(self):
#         return self['L']

#     @property
#     def size(self):
#         if not hasattr(self, '_size'):
#             self._size = self['tau']

#         return self._size

#     @property
#     def init(self):
#         """
#         Returns the maximum likelihood initialisation. 
#         """
#         return np.argmax(self.likelihood)

#     @property
#     def summary_table(self):
#         if not hasattr(self, '_summary'):
#             df = pd.DataFrame({
#                 'L1': self.__get_as_list('L1'),    
#                 'L2': self.__get_as_list('L2'),
#                 'L3': self.__get_as_list('L3'),
#                 'L': self.likelihood, #self.__get_as_list('L', -1),
#                 'size': self.__get_as_list('tau'),
#                 #'P_ts': self.__get_as_list('P_ts'),
#                 #'P_rt': self.__get_as_list('P_rt'),
#                 #'P_c': self.__get_as_list('P_c'),
#                 #'P_a': self.__get_as_list('P_a'),
#                 #'wd': self.__get_as_list('WD'),
#                 #'ks': self.__get_as_list('KS'),
#                 'rank': [self.rank] * self.iter,
#                 'init': np.arange(0, self.iter),
#                 'k': [self.parameters] * self.iter,    
#                 'n': list(map(int, self.__get_as_list('data_pts'))),    
#                 })
#             df[AIC] = 2 * df['k'] - 2 * df['L']
#             df[AIC_C] = df[AIC] + (2*df['k']**2 + 2 * df['k']) / (df['n'] - df['k'] - 1)
#             df[BIC] = np.log(df['n']) * df['k'] - 2*df['L'] 

#             self._summary = df
        
#         return self._summary

#     def log_table(self, resolution=100, skip_front=20, skip_back=200):
    
#         end = self[EPOCHS] - skip_back * resolution
        
#         df = pd.DataFrame({   
#             EPOCHS : np.arange(resolution * skip_front, end, resolution).tolist() * self.iter,
#             'init': np.array([[i] * int((end-resolution*skip_front)/resolution) for i in range(self.iter)]).reshape(-1),
#             'L1': self['log_objective'][::resolution, :][skip_front:-skip_back,:].T.reshape(-1),
#             'lrate': self['log_learning_rate'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
#             'lr': self['log_lambda_r'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
#             'lt': self['log_lambda_t'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
#             'lc': self['log_lambda_c'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
#             'la': self['log_lambda_a'][::resolution, :][skip_front:-skip_back, :].T.reshape(-1),
#             })
        
#         return df

#     def coefficient_table(self, cdim='b0', avg=False):
#         """
#         Returns a panda data frame with inferred coefficients for each signature 
#         and initialisation of the cluster.
        
#         Parameters:
#         cdim (string): name of the coefficient (eg. "b0", "a0" etc.)
#         avg (boolean): returns computes the average of fitted coefficient over initialisations
#         """
#         coeff_table = pd.DataFrame({
#             'sig': np.array([[i] * self[cdim].shape[0] for i in range(self.rank)]).reshape(-1).tolist() * self.iter, 
#             'dim': np.arange(self[cdim].shape[0]).tolist() * self[cdim].shape[1] * self[cdim].shape[2],
#             'init': np.array([[i] * self.rank * self[cdim].shape[0] for i in range(self.iter)]).reshape(-1).tolist(),
#             'val': self[cdim].T.reshape(-1).tolist()})

#         if avg:
#             coeff_table = coeff_table.groupby(['sig', 'dim']).agg({'val':[np.mean, np.std]}).reset_index()
#             coeff_table.columns = [ ' '.join(col).strip() for col in df.columns ]

#         return coeff_table


#     def plot_silhouette_signature(self):
#         plot_silhouette_signature(self.S)

#     def plot_silhouette_iteration(self, **kwargs):
#         plot_silhouette_iteration(self.scores, **kwargs)

#     def plot_signature(self, init=None, **kwargs):
#         """
#         Plots the signature spectra. If no integer for the initialisation
#         is given the method selects the initialisation for which the likelihood        
#         was maximised.
#         """
#         if init is None:
#             init = self.init
        
#         ax = plot_multdim_sig(self.S[..., init], **kwargs)
#         return ax
    
#     def plot_collapsed_signature(self, init=None, **kwargs):
#         """
#         Plots the signature spectra. If no integer for the initialisation
#         is given the method selects the initialisation for which the likelihood        
#         was maximised.
#         """
#         if init is None:
#             init = self.init
        
#         ax = plot_collapsed_signature(self.S[..., init], **kwargs)
#         return ax
    
#     def plot_tSNE_signature(self):
#         plot_tSNE_signature(self.S)

#     def plot_likelihood_across_tr(self):
#         likelihood_across_transcription_replication(self['Lij_1'])
    
#     def plot_likelihood_across_sub(self):
#         likelihood_across_subs(self['Lij_1'])

#     def plot_big_spectra(self, init=None, **kwargs):
#         if init is None:
#             init = self.init

#         T = self['T']
#         plot_big_spectra(T[..., init], **kwargs)

#     def plot_activties(self, init=None, **kwargs):
#         if init is None:
#             init = self.init

#         F = self.get_sorted_parameter('F')
#         plot_activities(F[..., init], **kwargs)

#     def plot_ecdf(self, init=None, ax=None):
#         if init is None:
#             init = self.init

#         sample = self['khat'][..., init].reshape(-1)
#         ecdf = sm.distributions.ECDF(sample)
#         x = np.linspace(min(sample), max(sample))
#         y = ecdf(x)

#         if ax is None:
#             fig, ax = plt.subplots(1, 1)

#         ax.step(x, y, label='ECDF')
#         tx = np.arange(0, int(np.max(self['khat'][..., self.init].reshape(-1))))
#         ty = stats.gamma.cdf(tx, a=self['alpha'][..., self.init][0], scale=self['beta'][..., self.init][0])
#         ax.plot(tx, ty, label='theoretical CDF')

#         return ax

# class PreCluster(Cluster):
#     def __init__(self, dset, S, E, icol, scores, **kwargs):
#         self.dset = dset
#         self.memo = {}
#         self.S = S
#         self.E = E
#         self.icol = icol
        
#         self.scores = scores
#         # added fast cluster option which sets the seed to -1
#         self.seed = np.argmax(self.scores) if self.scores is not None else -1 
        
#         self.iter = self.S.shape[-1]
#         self.rank = self.S.shape[-2]
#         self.samples = self.E.shape[-2]

# class Singleton(Cluster):
#     def __init__(self, dset, **kwargs):
#         self.dset = load_dict(dset)[1]
#         self.S = self.dset['S']
#         self.E = self.dset['E']
                
#         self.iter = 1
#         self.rank = self.S.shape[-1]
#         self.samples = self.E.shape[-2]
    
#     def __getitem__(self, item):
#         if item in self.dset:
#             if item in ['T', 'a0', 'b0', 'm0', 'm1', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5']:
#                 return self.dset[item].reshape(*(*self.dset[item].shape, 1))
#             else:
#                 return self.dset[item][()]
#         else:
#             raise KeyError('Could not find item.')

# class FastCluster(Cluster):

#     def __init__(self, dset, **kwargs):
#         self.dset = dset
#         self.memo = {}
#         S = dset['S'][()]
#         E = dset['E'][()]
#         L = dset['L'][()]
#         m = L >= 0
#         I = np.argmax(np.ma.array(L, mask=m))
        
#         self.S, self.E, self.icol = cluster_signatures(dset['S'], dset['E'], I)
        
#         self.iter = self.S.shape[-1]
#         self.rank = self.S.shape[-2]
#         self.samples = self.E.shape[-2]    


# class Experiment(object):

#     def __init__(self, experiment, raw=False, merged=False, fast=False):
#         """
#         Experiment loads datasets dynamically.
#         """
#         self.expr = experiment
#         self.data = set([])
#         self.memo = dict()
#         self.fast = fast
#         self.raw = False
#         self.merged = merged

#         # walk through all experiment params
#         experiment.visititems(self.__visitor_func)
    
#     def __len__(self):
#         return len(self.data)

#     def __iter__(self):
#         for dset in self.data:
#             yield dset

#     def __getitem__(self, path):
#         assert(path in self.data), 'Sorry, did not find requested dataset.'
#         if path in self.memo:
#             return self.memo[path]
#         else:
#             if self.raw:
#                 self.memo[path] = self.expr[path]
#             else:
#                 if self.fast:
#                     self.memo[path] = FastCluster(self.expr[path])
#                 else:
#                     self.memo[path] = Cluster(self.expr[path])

#             return self.memo[path]

#     def __visitor_func(self, name, node):
#         if isinstance(node, h5.Dataset):
#             path = '/'.join(node.name.split('/')[:-1])
#             self.data.add(path)

#         if self.merged:
#             if isinstance(node, h5.Group):
#                 if len(node.attrs) != 0:
#                     self.data.add(name)
    
#     def items(self):
#         for dset in self.data:
#             yield dset, self[dset]

#     def pre_cluster(self, cores=8):
#         """Preclusters all datasets."""
#         data_sets = []

#         for i, dset in enumerate(self.data):
#             progress(i, len(self), dset)
#             S = self.expr[dset]['S'][()]
#             E = self.expr[dset]['E'][()]
#             data_sets.append((dset, S, E))

#         pool = Pool(cores)
#         results = pool.starmap(mp_cluster, data_sets)

#         for r in results:
#             self.memo[r[0]] = PreCluster(self.expr[r[0]], r[1], r[2], r[3], r[4])

#         pool.close()

#     def fast_cluster(self, cores=8, signature_tensor='S'):
#         """
#         Uses the initializsation with largest likelihood as seed, and does not
#         iterate over all initializations.
#         """
#         assert(cores>0)
#         data_sets = []

#         for i, dset in enumerate(self.data):
#             progress(i, len(self), dset)
#             S = self.expr[dset][signature_tensor][()]
#             E = self.expr[dset]['E'][()]
#             L = self.expr[dset]['L'][()]
#             m = L >= 0
#             I = np.argmax(np.ma.array(L, mask=m))

#             data_sets.append((dset, S, E, I))        
#         if cores > 1:
#             pool = Pool(cores)
#             results = pool.starmap(mp_fast_cluster, data_sets)
            
#             pool.close()
        
#         else:
#             results = []
#             for dset in data_sets:
#                 results.append(mp_fast_cluster(*dset))

#         for r in results:
#             self.memo[r[0]] = PreCluster(self.expr[r[0]], r[1], r[2], r[3], r[4])

#     @property
#     def summary_table(self, **kwargs):
#         if hasattr(self, '_summary'):
#             return self._summary

#         data_frames = []

#         for k, v in self.items():
#             df = v.summary_table
#             df[JOB_NAME] = k
#             data_frames.append(df)

#         self._summary = pd.concat(data_frames)

#         return self._summary

#
# load data functions
#
    
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

def compute_tnc_freq(genome_type='whole_genome'):
    """
    genome_type can be "whole_genome" or "exome"
    """
    with h5.File(PCAWG_TGCA) as fh:
        if genome_type == 'whole_genome':
            M0 = fh['TNC/WG'][()].reshape(3,3,2,96)
        elif genome_type == 'exome':
            M0 = fh['TNC/AGILENT'][()].reshape(3,3,2,96)
    
    M = (M0/M0.sum(axis=3, keepdims=True) * M0.sum()/M0.sum(axis=(0,1,2), keepdims=True)).reshape(3,3,1,2,96,1)
    
    return M

def compute_snv_freq(counts):
    """
    Computes the SNV frequencies for a given count matrix.
    """
    counts = counts.sum(axis=2)
    N0_pyr = (np.sum(counts, axis=(2,3,4)) / np.sum(counts)).reshape(3, 3, 1, 1, 1, 1)
    N0_pur = N0_pyr[[1, 0, 2],:,:][:,[1,0,2],:]
    N = np.concatenate([N0_pyr, N0_pur], axis=3)
    return N

def load_counts():
    subs = h5.File(PCAWG_SNV)
    counts = subs['TRC'][()]
    counts = counts.transpose(3, 2, 1, 4, 0)
    counts = counts.reshape(3,3,2,2,96,2703)
    counts = counts.sum(axis=2) 

    N = counts.sum((2,3), keepdims=True)/counts.sum((0,1,2,3),keepdims=True)
    N = np.concatenate([N, N[[2, 1, 0],:,:,:,:][:,[2,1,0],:,:,:]], axis=2)

    T = pd.read_csv(PCAWG_TNC, sep="\t")
    TNC = np.transpose(T.as_matrix().reshape(3,3,2,96) + 0.0, (1,0,2,3))
    M = TNC / TNC.sum(axis=(3),keepdims=True) * TNC.sum(keepdims=True) / TNC.sum(axis=(0,1,2),keepdims=True)
    M = M.reshape(3,3,2,96,1)

    return counts, N, M

def nans(shape, dtype=np.float32):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def load_pcawg_tcga():
    data = h5.File(PCAWG_TGCA)
    pcawg_snv = data['PCAWG/SNV'][()]
    tcga_snv = data['TCGA/SNV'][()]
    
    pcawg_ncl = pcawg_snv.sum(axis=2)
    tcga_ncl = tcga_snv.sum(axis=2)

    # set the clustered dimension of the TCGA SNV array to NA
    tcga_nan = nans(tcga_ncl.shape)
    tcga_fin = np.concatenate([tcga_ncl.reshape(3,3,1,2,96,-1), tcga_nan.reshape(3,3, 1, 2, 96, -1)], axis=2)
    snv = np.concatenate([pcawg_snv, tcga_fin], axis=5)
    
    # loads all other covariates
    other = data['MERGED/COMBINED'][()].T

    # some changes
    # compute trinucleotide frequencies
    M1 = data['TNC/WG'][()].reshape(3,3,2,96)
    M1 = (M1/M1.sum(axis=3, keepdims=True) * M1.sum()/M1.sum(axis=(0,1,2), keepdims=True)).reshape(3,3,1,2,96,1)
    M2 = data['TNC/AGILENT'][()].reshape(3, 3, 2, 96)
    M2 = (M2/M2.sum(axis=3, keepdims=True) * M2.sum()/M2.sum(axis=(0,1,2) , keepdims=True)).reshape(3, 3, 1, 2, 96, 1)
    # merge the in a big array
    M1 = np.concatenate([M1] * pcawg_snv.shape[-1], axis=5)
    M2 = np.concatenate([M2] * tcga_snv.shape[-1], axis=5)
    M = np.concatenate([M1, M2], axis=5)

    
    N1 = (np.sum(pcawg_ncl, axis=(2,3,4)) / np.sum(pcawg_ncl)).reshape(3, 3, 1, 1, 1, 1)
    N1_pyr = np.concatenate([N1] * pcawg_ncl.shape[-1], axis=5)
    N1_pur = np.concatenate([N1[[1, 0, 2],:,:][:,[1,0,2],:]] * pcawg_ncl.shape[-1], axis=5)
    N1 = np.concatenate([N1_pyr, N1_pur], axis=3)

    N2 = (np.sum(tcga_ncl, axis=(2,3,4)) / np.sum(tcga_ncl)).reshape(3, 3, 1, 1, 1, 1)
    N2_pyr = np.concatenate([N2] * tcga_ncl.shape[-1], axis=5)
    N2_pur = np.concatenate([N2[[1,0,2],:,:][:,[1,0,2],:]] * tcga_ncl.shape[-1], axis=5)
    N2 = np.concatenate([N2_pyr, N2_pur], axis=3)

    N = np.concatenate([N1, N2], axis=5)

    return snv, other, M, N

def moritz_data():
    with h5.File(MORITZ_SNV, 'r') as f:
        countsTRC = f['allSubsTransRepClust'][()]
    countsTRC = countsTRC.transpose(3,2,1,4,0).reshape(3,3,3,2,96,2778)
    countsTRC.shape
    countsTR = countsTRC.sum(axis=2)

    tab = pd.read_csv(MORITZ_TNC, sep="\t")
    tncTR=np.transpose(tab.as_matrix().reshape(3,3,2,96) + 0.0, (1,0,2,3)) # python has reverse indexing

    tab = pd.read_csv(MORITZ_OTHER, sep="\t")
    mutations = list(tab.columns)
    countsO=tab.as_matrix()+0.0
    countsO=countsO[:,96:(475-145)].transpose() ## Remove SNVs and CN
    mutations = list(tab)
    #import pdb; pdb.set_trace()

    N = (np.sum(countsTR, axis=(2,3,4)) / np.sum(countsTR, axis=(0,1,2,3,4))).reshape(3,3,1,1,1)
    N = np.concatenate([N,N[[1,0,2],:,:,:,:][:,[1,0,2],:,:,:]], axis=2).reshape(3,3,1,2,1,1)
    M = tncTR / tncTR.sum(axis=(3),keepdims=True) * tncTR.sum(keepdims=True) / tncTR.sum(axis=(0,1,2),keepdims=True)
    M = M.reshape(3,3,1,2,96,1)

    return countsTRC, countsO, M, N


def load_pcawg():
    with h5.File(PCAWG_TGCA) as data:
        pcawg_snv = data['PCAWG/SNV'][()]
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T
        M1 = data['TNC/WG'][()].reshape(3,3,2,96)
    pcawg_ncl = pcawg_snv.sum(axis=2)

    other  = np.concatenate([mnv, indel, sv], axis=0)

    
    M = (M1/M1.sum(axis=3, keepdims=True) * M1.sum()/M1.sum(axis=(0,1,2), keepdims=True)).reshape(3,3,1,2,96,1)
    
    N0 = (np.sum(pcawg_ncl, axis=(2,3,4)) / np.sum(pcawg_ncl)).reshape(3, 3, 1, 1, 1, 1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=3)
    
    N = (M*N)/2
    print('N={}'.format(N.sum()))
    
    return pcawg_snv, other, N

def load_pcawg_chrom():
    with h5.File(PCAWG_CHROM_SNV) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3,3,16,2,96,-1)
    pcawg_ncl = pcawg_snv.sum(axis=2)

    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T
        M1 = data['TNC/WG'][()].reshape(3,3,2,96)

    other  = np.concatenate([mnv, indel, sv], axis=0)

    TNC = pd.read_csv(PCAWG_CHRM_TNC, index_col=0)
    M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    #M = (M0/M0.sum()).reshape(3,3,16,2,96,1)

    #M = (M1/M1.sum(axis=3, keepdims=True) * M1.sum()/M1.sum(axis=(0,1,2), keepdims=True)).reshape(3,3,1,2,96,1)
    
    N0 = (np.sum(pcawg_snv, axis=(3,4,5)) / np.sum(pcawg_snv)).reshape(3, 3, 16, 1, 1, 1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=3)
    #N = np.ones_like(M)

    return pcawg_snv, other, M, N

def load_pcawg_chrom_clust():
    with h5.File(PCAWG_CHROM_CLUST) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3, 3, 16, 2, 2, 96, -1)
    pcawg_ncl = pcawg_snv.sum(axis=3)

    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T
        M1 = data['TNC/WG'][()].reshape(3,3,2,96)

    other  = np.concatenate([mnv, indel, sv], axis=0)

    TNC = pd.read_csv(PCAWG_CHRM_TNC, index_col=0)
    M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    #M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    M = (M0/M0.sum()).reshape(3,3,16,2,96,1)
    M = M.reshape(3,3,16,2,1,96,1)
    
    #M = (M1/M1.sum(axis=3, keepdims=True) * M1.sum()/M1.sum(axis=(0,1,2), keepdims=True)).reshape(3,3,1,2,96,1)
    
    #N0 = (np.sum(pcawg_snv, axis=(4,5,6)) / np.sum(pcawg_snv)).reshape(3, 3, 16, 1, 1, 1)
    #N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=3)
    N = np.ones_like(M)
    N = N.reshape(3,3,16,1,2,96, 1)
    
    return pcawg_snv, other, M, N


def load_pcawg_chrom_clust2():
    with h5.File(PCAWG_CHROM_CLUST) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3, 3, 16, 2, 2, 96, -1)
    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T
        M1 = data['TNC/WG'][()].reshape(3,3,2,96)

    other  = np.concatenate([mnv, indel, sv], axis=0)

    TNC = pd.read_csv(CHROM_CLUST_TNC, index_col=0)
    tnc192 = pd.read_csv(CHROM_CLUST_TNC, index_col=0)
    M1 = tnc192.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    M2 = M1/M1.sum(axis=(4), keepdims=True) / M1.sum(axis=(0,1,2,3), keepdims=True) * M1.sum(keepdims=True) 
    
    #M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    #M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    
    
    N0 = (pcawg_snv.sum(axis=(3,4,5,6))/ pcawg_snv.sum()).reshape(3,3,16,1,1,1,1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=4)
    
    N = M2.reshape(3,3,16,1,2,96,1) * N / 2
    #N = np.array([1])
    print("SUM N={}".format(N.sum()))
    
    return pcawg_snv, other, N

def load_pcawg_nucleosome():
    with h5.File(PCAWG_NUC) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3, 3, 2, 2, 2, 96, -1)
    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T

    other  = np.concatenate([mnv, indel, sv], axis=0)

    tnc192 = pd.read_csv(PCAWG_NUC_TNC, index_col=0)
    M1 = tnc192.as_matrix().T.reshape(9, 2, 2, 96).reshape(3,3,2,2,96)
    M2 = M1/M1.sum(axis=(4), keepdims=True) / M1.sum(axis=(0,1,2,3), keepdims=True) * M1.sum(keepdims=True) 
    
    #M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    #M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    
    
    N0 = (pcawg_snv.sum(axis=(3,4,5,6))/ pcawg_snv.sum()).reshape(3,3,2,1,1,1,1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=4)
    
    N = M2.reshape(3,3,2,1,2,96,1) * N / 2
    #N = np.array([1])
    print("SUM N={}".format(N.sum()))
    
    return pcawg_snv, other, N


def load_pcawg_min_maj():
    with h5.File(PCAWG_MINMAJ) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3, 3, 3, 2, 2, 96, -1)
    # resort layers * + -
    pcawg_snv = np.concatenate([
        pcawg_snv[:,:,2,:,:,:,:].reshape(3,3,1,2,2,96,-1),
        pcawg_snv[:,:,0,:,:,:,:].reshape(3,3,1,2,2,96,-1),
        pcawg_snv[:,:,1,:,:,:,:].reshape(3,3,1,2,2,96,-1)], axis=2)

    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T

    other  = np.concatenate([mnv, indel, sv], axis=0)

    tnc192 = pd.read_csv(PCAWG_MINMAJ_TNC, index_col=0)
    M1 = tnc192.as_matrix().T.reshape(9, 3, 2, 96).reshape(3,3,3,2,96)
    M2 = M1/M1.sum(axis=(4), keepdims=True) / M1.sum(axis=(0,1,2,3), keepdims=True) * M1.sum(keepdims=True) 
    
    #M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    #M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    
    
    N0 = (pcawg_snv.sum(axis=(3,4,5,6))/ pcawg_snv.sum()).reshape(3,3,3,1,1,1,1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=4)
    
    M = M2.reshape(3,3,3,1,2,96,1) * N / 2
    #N = np.array([1])
    print("SUM N={}".format(N.sum()))
    
    return pcawg_snv, other, N

def load_pcawg_min_maj_lin():
    with h5.File(PCAWG_MINMAJLIN) as data:
        pcawg_snv = data['SNV'][()].T

    pcawg_snv = pcawg_snv.reshape(3, 3, 4, 2, 2, 96, -1)
    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T

    other  = np.concatenate([mnv, indel, sv], axis=0)

    tnc192 = pd.read_csv(PCAWG_MINMAJLIN_TNC, index_col=0)
    M1 = tnc192.as_matrix().T.reshape(9, 4, 2, 96).reshape(3,3,4,2,96)
    M2 = M1/M1.sum(axis=(4), keepdims=True) / M1.sum(axis=(0,1,2,3), keepdims=True) * M1.sum(keepdims=True) 
    
    #M0 = TNC.as_matrix().T.reshape(9, 16, 2, 96).reshape(3,3,16,2,96)
    #M = (M0/M0.sum(axis=4, keepdims=True)*M0.sum()/M0.sum(axis=(0,1,2,3))).reshape(3,3,16,2,96,1)
    
    
    N0 = (pcawg_snv.sum(axis=(3,4,5,6))/ pcawg_snv.sum()).reshape(3,3,4,1,1,1,1)
    N = np.concatenate([N0, N0[[1, 0, 2],:,:][:,[1,0,2],:]], axis=4)
    
    N = M2.reshape(3,3,4,1,2,96,1) * N / 2
    #N = np.array([1])
    print("SUM N={}".format(N.sum()))
    
    return pcawg_snv, other, N


def load_pcawg_8dim():
    with h5.File(PCAWG_8DIM) as data:
        snv = data['SNV'][()].T

    snv = snv.reshape(3,3,16,4,2,2,96,-1)
    
    with h5.File(PCAWG_TGCA) as data:
        mnv = data['PCAWG/MNV'][()].T
        indel = data['PCAWG/INDEL'][()].T
        sv = data['PCAWG/SV'][()].T

    other  = np.concatenate([mnv, indel, sv], axis=0)

    tnc = pd.read_csv(PCAWG_8DIM_TNC, index_col=0)
    # rearrangement to get numbers into the right dimension
    M0 = tnc.as_matrix().T.reshape(9, 16*4, 2, 96) 
    M1 = M0.reshape(3, 3, 16*4, 2, 96)
    M2 = M1.reshape(3, 3, 4, 16, 2, 96)
    M3 = M2.swapaxes(3, 2) # has now shape (3, 3, 16, 4, 2, 96)
    # add genome wide relative freq
    tnc96 = M0.sum(axis=(0,1,2))/M0.sum()
    M3 += tnc96.reshape(1,1,1,1,1,96)
    M4 = M3/M3.sum(axis=5, keepdims=True) / M3.sum(axis=(0,1,2,3,4), keepdims=True) * M3.sum()
    M5 = M4.reshape(3, 3, 16, 4, 1, 2, 96, 1)
    
    
    N0 = (snv.sum(axis=(4,5,6,7)) / snv.sum()).reshape(3,3,16,4,1) # shape (3, 3, 16, 4, 1)
    N1 = np.concatenate([N0, N0[[1,0,2],:,:][:,[1,0,2],:,:]], axis=4)
    N2 = N1.reshape(3,3,16,4,1,2,1,1)
    N = (N2*M5)/2

    print("SUM N={}".format(N.sum()))
    
    return snv, other, N


def load_pcawg_8dim_merged():
    snv, other, N = load_pcawg_8dim()
    N = np.concatenate([N[:,:,[0,3,6,10,11,12],:,:,:,:,:].sum(axis=2).reshape(3,3,1,4,1,2,96,1), N[:,:,[1,2,4,5,7,8,9,13,14,15],:,:,:,:,:]], axis=2)
    N = np.concatenate([N.reshape(3,3,-1,2,96,1)]*2, axis=2)
    snv = np.concatenate([snv[:,:,[0,3,6,10,11,12],:,:,:,:,:].sum(axis=2).reshape(3,3,1,4,2,2,96,2778), snv[:,:,[1,2,4,5,7,8,9,13,14,15],:,:,:,:]], axis=2)
    return snv, other, N


def save_dict(data, out_path):
    with open(out_path, 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)

def load_dict(data):
    with open(data, 'rb') as fh:
        params = pickle.load(fh)

    return (data.split('/')[-1], params)

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def progress(iteration, total, log_string):
    """
    Gives information about the how far the process has progressed.
    -----
    arguments
    """
    sys.stdout.write('\rProgress {iter}/{total} {percent} % {log_string}'.format(
                iter=iteration,
                total=total,
                percent=int(iteration / total * 100),
                log_string=log_string
                ),)
    sys.stdout.flush()

    return 0

def create_file_name(params):
    assert params[FILENAME][0] == 'J', 'A filename has to start with the job name'
    PARSER = dict(
        R=RANK, 
        LR=STARTER_LEARNING_RATE, 
        I=ITERATION,
        L=LAMBDA,
        J=JOB_NAME,
        K=DISPERSION,
        #A=ALPHA,
        #B=BETA,
        LRT=LAMBDA_R,
        LTS=LAMBDA_T,
        LC=LAMBDA_C,
        LA=LAMBDA_A
    )
    
    fname = ''
    for exp in params[FILENAME].split('_'):
        if exp == 'J':
            fname += params[PARSER[exp]]
        else:
            fname += '_' + exp + '=' + str(params[PARSER[exp]])
    
    return fname

def collapse_data(snv):
    col1 = snv[[slice(None)]*(snv.ndim-3)+[0]+[slice(None)]*2]
    col2 = []
    for i, j in [(1, 1), (1, 0), (1, 2), (0, 1), (0, 0), (0, 2), (2, 1), (2, 0), (2, 2)]:
        col2.append(snv[[i, j]+[slice(None)]*(snv.ndim-5)+[1]+[slice(None)]*2])
    col2 = np.stack(col2).reshape(col1.shape)
    return col1 + col2

