#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats as stats
import h5py as h5
from tensorsignatures.config import *
from tensorsignatures.util import *
import subprocess
import os

BMIN, BMAX = -.5, .5
AMIN, AMAX = -.5, .5
KMIN, KMAX = -2, 2
KSIZ = 2


class TensorSignatureData(object):

    def __init__(self, seed, rank, samples=100, tau=50, mutations=1000,
                 verbose=True, dim=[2], **kwargs):
        """
        Simulates data for TensorSignatures.

        Parameters
        - seed                 seed for data instantiation (int)
        - signatures           signatures preselected or random (list or int)
        - noise                noise changes in spectra (None or float (0,1))
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.verbose = verbose
        self.rank = rank
        self.gen = samples
        self.tau = tau
        self.mut = mutations
        self.dim = dim

        self.idx = np.random.choice(np.arange(40), replace=False, size=self.rank)
        self.S0 = np.loadtxt(SIMULATION)[:, self.idx]
        self.T0 = np.loadtxt(OTHER)[:, self.idx]

        self.S1
        self.B
        self.A
        self.M
        self.K


        self.S = self.S1 * self.B * self.A * self.M

        for i, k in self.K.items():
            self.S = self.S * k
        #self.S /= self.S.sum(axis=(0,1,2,3,))
        #self.S /= self.S.sum(axis=(2,4), keepdims=True)
        #self.S3 = self.S2 / self.S2.sum(axis=(0,1,2,3))
        #self.S = self.S3 * self.M

        #self.S = self.S1 * self.M
        self.T = self.T0 * (1-self.M.reshape(-1, self.rank))
        # self.E
        # additional code
        # self.dim_snv = self.S.reshape(-1, self.rank).shape[0]
        # tmp = np.concatenate([self.S.reshape(-1, self.rank), self.T])
        # tmp /= tmp.sum(0)
        # self.S = tmp[:self.dim_snv,:].reshape(3,3,KSIZ+1,96,self.rank)
        # self.T = tmp[self.dim_snv:,:]

    def __getitem__(self, item):
        if not hasattr(self, '_var'):
            self._var = {
                'b0': self.b0,
                'a0': self.a0,
                'm1': self.m1,
                'E': self.E,
                **self._k}
        return self._var[item]

    def __add_noise(self, signatures, noise_strengh):
        p, r = signatures.shape
        S = []
        for r_i in range(r):
            S_i = signatures[:, r_i] \
                + np.random.uniform(-signatures[:, r_i], signatures[:, r_i]) \
                * noise_strengh
            S_i = S_i / S_i.sum()
            S.append(S_i)

        return np.stack(S, axis=1)

    @property
    def S1(self, noise=None):
        if not hasattr(self, '_S1'):
            pppu, pmpu, pppy, pmpy = self.S0, self.S0, self.S0, self.S0

            self._S1 = np.stack([
                pppy, pmpu, 1 / 2 * (pppy + pmpu),
                pmpy, pppu, 1 / 2 * (pmpy + pppu),
                1/2 * (pppy + pmpy), 1/2 * (pmpu + pppu), 1/4 * (pppu + pmpy + pmpu + pppy)
                ]).reshape(3, 3, *[ 1 for j in enumerate(self.dim) ], 96, self.rank)

            #self._S1 = self._S1/self._S1.sum(axis=(0,1,2,3,4))

            if self.verbose:
                print(self._S1.shape)


        return self._S1

    @property
    def B(self):
        if not hasattr(self, '_B'):
            self.b0 = np.random.uniform(BMIN, BMAX, size=2*self.rank).reshape(2, self.rank)
            self._B = np.exp(np.stack([
                self.b0[1,:]+self.b0[0,:], self.b0[0,:]-self.b0[1,:], self.b0[0,:],
                -self.b0[0,:]+self.b0[1,:], -self.b0[0,:]-self.b0[1,:], -self.b0[0,:],
                self.b0[1,:], -self.b0[1,:], np.zeros((self.rank))
            ]).reshape(3, 3, *[ 1 for j in enumerate(self.dim) ], 1, self.rank))


            if self.verbose:
                print(self._B.shape)
        return self._B

    @property
    def A(self):
        """
        Initialize amplitudes.
        """
        if not hasattr(self, '_A'):
            self.a0 = np.random.uniform(AMIN, AMAX, size=2*self.rank).reshape(2, self.rank)
            a1 = np.exp(np.concatenate([self.a0, self.a0, np.zeros([2, self.rank])], axis=0).reshape(3, 2, self.rank))
            a2 = a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :] # outer product
            self._A = a2.reshape(3, 3, *[ 1 for j in enumerate(self.dim) ], 1, self.rank)
            if self.verbose:
                print(self._A.shape)
        return self._A

    @property
    def M(self):
        """
        Initialize amplitudes.
        """
        if not hasattr(self, '_M'):
            self.m1 = np.random.uniform(0, 1, size=self.rank).reshape(1, self.rank)
            self._M = self.m1.reshape(1, 1, *[ 1 for j in enumerate(self.dim) ], 1, self.rank)
            if self.verbose:
                print(self._M.shape)
        return self._M

    @property
    def K(self):
        """
        Initialize amplitudes.
        """
        if not hasattr(self, '_K'):
            self._k = {}
            self._K = {}

            for i, size in enumerate(self.dim):
                ki = np.random.uniform(KMIN, KMAX, size=self.rank*(size-1)).reshape(size-1, self.rank)
                self._k['k{}'.format(i)] = ki

                Ki = np.exp(
                    np.concatenate(
                        [np.zeros([1, self.rank]), ki], axis=0).reshape(-1, self.rank)
                ).reshape(1, 1, *[ size if j == i else 1 for j, size in enumerate(self.dim) ], 1, self.rank)

                self._K['k{}'.format(i)] = Ki

                if self.verbose:
                    print(Ki.shape)

        return self._K

    @property
    def E(self):
        if not hasattr(self, '_E'):
            if self.mut == 'log_normal':
                self.Ej = np.exp(np.random.normal(8.63, 1.43, size=self.samples)).reshape(1, -1)
            else:
                self.Ej = self.mut #/ (self.S.sum(axis=(0,1,2,3,))+self.T.sum(0)).mean()
            E = np.random.uniform(size=self.rank * self.gen).reshape(self.rank, self.gen)
            self._E = E/E.sum(0) * self.Ej * 1/(self.S.sum(axis=tuple([j for j in range(self.S.ndim-1)]))+self.T.sum(0)).reshape(self.rank,1)

        return self._E

    @property
    def C1(self):
        if not hasattr(self, '_C1'):
            self._C1 = (self.S.reshape(-1, self.rank) @ self.E).reshape(3, 3, *self.dim, 96, self.gen)

        return self._C1

    @property
    def C2(self):
        if not hasattr(self, '_C2'):
            self._C2 = (self.T @ self.E).reshape(234, self.gen)

        return self._C2

    def snv(self, init=0):
        self._snv = stats.nbinom.rvs(self.tau, self.tau/(self.C1 + self.tau), random_state=init)
        return self._snv

    def other(self, init=0):
        self._other = stats.nbinom.rvs(self.tau, self.tau/(self.C2 + self.tau), random_state=init)
        return self._other

    def plot_signatures(self):
        plot_collapsed_signature(self.S.reshape(3, 3, -1, 96, self.rank))
        #ts.plot_collapsed_signature(self.S)

    def plot_biases(self):
        coefficients = {
            'sig': np.arange(self.rank).tolist(),
            'mix': self.m1.reshape(-1).tolist(),
            'b00': self.b0[0, :].reshape(-1).tolist(),
            'b01': self.b0[1, :].reshape(-1).tolist(),
            'a00': self.a0[0, :].reshape(-1).tolist(),
            'a01': self.a0[1, :].reshape(-1).tolist(),
        }
        ax = sns.factorplot(x='variable', y='value', hue='sig',data=pd.DataFrame(coefficients).melt('sig'), kind='bar')
        return ax

    def save_init(self, path, init):
        fh = h5.File(path, 'w')
        dset = fh.create_dataset('SNV', data=self.snv(init=init))
        dset.attrs['seed'] = self.seed
        dset.attrs['rank'] = self.rank
        dset.attrs['samples'] = self.gen
        dset.attrs['mutations'] = self.mut
        dset.attrs['init'] = init
        dset.attrs['path'] = path

        fh.create_dataset('OTHER', data=self.other(init=init))
        #fh.create_dataset('N', data=self.N)
        fh.close()

    def coefficient_table(self, clu, cdim, as_df=True, unscaled=False, only_init=False):
        dic = {
            'sig': [],
            'dim': [],
            'init': [],
            'true':[],
            'pred':[],
            'var':[]
        }

        if only_init:
            g, h = clu.init, clu.init+1
        else:
            g, h = 0, clu[cdim].shape[-1]

        for i in range(g, h):
            r, c, d = self.map_signatures(clu, i)

            if cdim == 'E':
                if unscaled:
                    dic['true'].extend(self[cdim][r,...].reshape(-1))
                    dic['pred'].extend(clu.E[c,..., i].reshape(-1))
                else:
                    #dic['true'].extend((self[cdim][r,...] * (self.S.sum(axis=tuple([j for j in range(self.S.ndim-1)]))+self.T.sum(0)).reshape(self.rank,1)).reshape(-1))
                    #dic['pred'].extend((clu.E[c,..., i] *(self.S.sum(axis=tuple([j for j in range(self.S.ndim-1)]))+self.T.sum(0)).reshape(self.rank,1)).reshape(-1))
                    dic['true'].extend((self[cdim][r,...] * (self.S.sum(axis=tuple([j for j in range(self.S.ndim-1)]))+self.T.sum(0)).reshape(self.rank,1)).reshape(-1))
                    dic['pred'].extend((clu.E[c,..., i] * (clu.S[...,c,i].sum(axis=tuple([j for j in range(clu.S[...,c,i].ndim-1)]))+clu['T'][...,c,i].sum(0)).reshape(clu.rank,1)).reshape(-1))

                dic['dim'].extend(np.concatenate([[i] for i in range(self[cdim].shape[1])]*self.rank).reshape(-1).tolist())
                dic['init'].extend([i] * self.rank * self[cdim].shape[1])
                dic['sig'].extend(np.concatenate([[i] * self[cdim].shape[1] for i in range(self.rank)]).reshape(-1).tolist())
                dic['var'].extend([cdim] * self[cdim].shape[1] * self.rank)

            else:
                dic['pred'].extend(clu[cdim][..., c, i].reshape(-1))
                dic['true'].extend(self[cdim][..., r].reshape(-1))
                dic['dim'].extend(np.concatenate([[i]*self.rank for i in range(self[cdim].shape[0])]).reshape(-1).tolist())
                dic['init'].extend([i] * self.rank * self[cdim].shape[0])
                dic['sig'].extend([i for i in range(self.rank)] * self[cdim].shape[0])
                dic['var'].extend([cdim] * self[cdim].shape[0] * self.rank)

        if as_df:
            return pd.DataFrame(dic)
        return dic

    def average_coefficients(self, clu, cdim):
        coefficients = pd.DataFrame(self.coefficient_table(clu, cdim))
        average_coefficients = coefficients.groupby(['sig', 'dim', ]).agg({
            'pred': ['mean', 'std'],
            'true': ['mean', 'std']
        }).reset_index()

        average_coefficients.columns = [' '.join(col).strip() for col in average_coefficients.columns.values]
        return average_coefficients

    def map_signatures(self, clu, init=None):
        if init is None:
            init = clu.init

        S = clu.S[..., init]
        T = clu['T'][..., init]

        ridx, cidx, dist = assign_signatures(
            np.concatenate([self.S.reshape(-1, self.rank), self.T]),
            np.concatenate([S.reshape(-1, self.rank), T]))


        return ridx, cidx, dist

def arg():
    import argparse
    description = """A description"""
    epilog= 'Designed by H.V.'
    # Initiate a ArgumentParser Class
    parser = argparse.ArgumentParser(description = description, epilog = epilog)

    # Call add_options to the parser
    parser.add_argument('seed', help='seed', type=int)
    parser.add_argument('init', help='init', type=int)
    parser.add_argument('output', help='output file')

    parser.add_argument('-r', type = int, help = 'rank (default=5)', default=5)
    parser.add_argument('-m', type = int, help = 'mutations (default=100)', default=100)
    parser.add_argument('-k', type = int, help = 'tau (default=50)', default=50)
    parser.add_argument('-s', type = int, help = 'samples (default=100)', default=100)
    parser.add_argument('-dim',
        type=int,
        nargs='+',
        help='additional dims (default = 2)',
        default=[2])
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'verbose mode')

    return parser


def main():
    import sys
    parser = arg()
    args = parser.parse_args(sys.argv[1:])

    data = TensorSignatureData(args.seed, args.r, mutations=args.m, samples=args.s, tau=args.k, dim=args.dim)

    data.save_init(args.output, args.init)


if __name__ == "__main__":
    main()




