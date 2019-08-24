#!/usr/bin/env python

import sys
import os
import warnings; warnings.filterwarnings("ignore")
import argparse
import textwrap
import datetime
import tensorflow as tf
import numpy as np
import h5py as h5
from tensorsignatures.config import *
from tensorsignatures.tensorsignatures import *
from tensorsignatures.util import *
from scipy.stats import nbinom


class TensorSignatureBootstrap(TensorSignature):
    def __init__(self, snv, other, N, clu, **kwargs):
        self.clu = clu
        self.init = kwargs.get('init', self.clu.init)
        self.rank = clu.rank
        self.samples = clu.samples #int(np.floor(clu.samples * kwargs.get('sub_sampling', False))) if kwargs.get('sub_sampling', False) else clu.samples
        self.collapse = kwargs.get('collapse', True)
        #self.sub = np.random.choice(np.arange(clu.samples), self.samples, replace=False) if kwargs.get('sub_sampling', False) else slice(None)

        self.verbose = kwargs.get('verbose', True)
        self.size = self.clu['tau'][..., self.init]
        # keep data
        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv)
        else:
            self.snv = snv

        self.other = other
        if N is not None:
            if self.collapse:
                self.N = TensorSignature.collapse_data(N).reshape(3, 3,-1, 96, 1)
            else:
                self.N = N.reshape(3, 3, -1, 96, 1)
        else:
            self.N = None

        self.clu_dim = sorted([ var for var in list(self.clu.dset) if var.startswith('k') ])
        self.card = [ self.clu[var].shape[0]+1 for var in self.clu_dim ]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(np.arange(self.card_prod), self.card)

        self.dtype = tf.float32
        self.p = kwargs.get('p', 96)
        self.tau = tf.constant(self.size, dtype=tf.float32)
        self.epochs = kwargs.get(EPOCHS, 5000)

        # initialize variables
        self.M
        self.S
        self.E
        self.T
        self.Chat1
        self.Chat2
        self.C1
        self.C2
        self.L1
        self.L2
        self.L

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.clu[STARTER_LEARNING_RATE]
        self.decay_learning_rate = self.clu[DECAY_LEARNING_RATE] # kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = self.clu[OPTIMIZER]
        self.minimize

    def fit_bootstrap_sample(self):
        #tf.reset_default_graph()

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


        for i in range(self.epochs):
            if i%1000==0:
                print('step', i, self.sess.run(self.L))
            _ = self.sess.run(self.minimize)

        self.bootstrap = self.get_tensors(self.sess)
        self.sess.close()

    @define_scope
    def A(self):
        self.a0 = tf.Variable(self.clu['a0'][..., self.init], name='a0')
        a1 = tf.exp(tf.reshape(tf.concat([self.a0, self.a0, tf.zeros([2, self.rank])], axis=0), (3, 2, self.rank)))
        a2 = a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :] # outer product
        self._A = tf.reshape(a2, [3, 3, 1, 1, self.rank])
        if self.verbose:
            print('A:', self._A.shape)

        return self._A

    @define_scope
    def B(self):
        self.b0 = tf.Variable(self.clu['b0'][..., self.init], name='b0') # Pyrimidine - Purine bias x 2 (TS, RS) X s
        self._B = tf.exp(tf.reshape(
            tf.stack([
                self.b0[0,:]+self.b0[1,:], self.b0[0,:]-self.b0[1,:], self.b0[0,:],
                self.b0[1,:] - self.b0[0,:], -self.b0[1,:] - self.b0[0,:], -self.b0[0,:],
                self.b0[1,:], -self.b0[1,:], tf.zeros(self.b0[0,:].shape)]),
            [3, 3, 1, 1, self.rank]))
        if self.verbose:
            print('B:', self._B.shape)

        return self._B

    @define_scope
    def S1(self):
        # basic parameters [+/+, +/-] x [Pyr, Pur] x 96-1 x s
        self.S0 = tf.Variable(self.clu['S0'][..., self.init], name='S0')
        S1 = tf.nn.softmax(tf.concat([self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2), dim=2, name='S1')
        self._S1 = tf.reshape(
            tf.stack([
                S1[0, 0, :, :], S1[1, 0, :, :], 0.5 * tf.reduce_sum(S1[:, 0, :, :], axis=0),
                S1[1, 1, :, :], S1[0, 1, :, :], 0.5 * tf.reduce_sum(S1[:, 1, :, :], axis=0),
                0.5 * (S1[0, 0, :, :] + S1[1, 1, :, :]), 0.5 * (S1[1, 0, :, :] + S1[0, 1, :, :]), 0.25 * (tf.reduce_sum(S1, axis=(0, 1)))]),
            [3, 3, 1, self.p, self.rank])

        return self._S1

    @define_scope
    def E(self):
        self.E0 = tf.Variable(self.clu['E0'][..., self.init], name='E0')
        self._E = tf.exp(self.E0, name='E')
        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def K(self):
        self._clu_var = {}
        self._cbiases = {}

        for clu in self.clu_dim:
            #print(clu)
            k = int(clu[1:])
            v = tf.Variable(self.clu[clu][..., self.init], name='k{}'.format(k))

            self._clu_var[k] = v
            self._cbiases[k] = tf.concat([tf.zeros([1, self.rank], dtype=self.dtype), v], axis=0)

            if self.verbose:
                print('k{}:'.format(k), self._cbiases[k].shape)

        final_tensor = []
        for r in range(self.idex.shape[0]):
            current_term = []
            for c in range(self.idex.shape[1]):
                current_term.append(self._cbiases[c][self.idex[r, c].astype(int), :])

            final_tensor.append(tf.reduce_sum(tf.stack(current_term), axis=0))

        self._K = tf.exp(tf.reshape(tf.stack(final_tensor), (1, 1, -1, 1, self.rank)))
        if self.verbose:
            print('K:', self._K.shape)
        return self._K

    @define_scope
    def M(self):
        self.m0 = tf.Variable(self.clu['m0'][..., self.clu.init], name='m0')
        self.m1 = tf.sigmoid(self.m0, name='m1')
        self._M = tf.reshape(self.m1, (1, 1, 1, 1, self.rank))
        if self.verbose:
            print('m:', self._M.shape)
        return self._M

    @define_scope
    def S(self):
        self._S = self.S1 * self.A * self.B * self.K * self.M
        if self.verbose:
            print('S:', self._S.shape)
        return self._S

    @define_scope
    def T(self):
        self.T0 = tf.Variable(self.clu['T0'][..., self.clu.init], name='T0')
        T1 = tf.nn.softmax(tf.concat([self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0), dim=0, name='T')
        self._T = T1 * (1-tf.reshape(self.M, (1, self.rank)))
        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def Chat1(self):
        self._Chat1 = tf.reshape(tf.matmul(tf.reshape(self.S, (-1, self.rank)), self.E), [3, 3, -1, self.p, self.samples])
        self._Chat1 *= (self.N.astype('float32') + 1e-6) #self.N

        if self.verbose:
            print('Chat1:', self._Chat1.shape)

        return self._Chat1

    @define_scope
    def Chat2(self):
        self._Chat2 = tf.matmul(self.T, self.E)
        if self.verbose:
            print('Chat2:', self._Chat2.shape)

        return self._Chat2

    @define_scope
    def C1(self):
        snv = nbinom.rvs(self.size, self.size/(self.size+self.snv.reshape(3, 3, -1, self.p, self.samples)))
        self._C1 = tf.constant(snv, dtype=self.dtype)
        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)

        other = nbinom.rvs(self.size, self.size/(self.size+self.other))
        self._C2 = tf.constant(other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)
        return self._C2

class TensorSignatureBootstrapDenovo(TensorSignature):
    """
    Tensor Signature that performs a denovo fitting on a resampled dataset.
    """
    def __init__(self, snv, other, N=None, **kwargs):
        super().__init__(snv, other, N, **kwargs)

    @define_scope
    def C1(self):
        print('Subsampling C1.')
        snv = nbinom.rvs(self.size, self.size/(self.size+self.snv.reshape(3, 3, -1, self.p, self.samples)))
        self._C1 = tf.constant(snv, dtype=self.dtype)
        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):

        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)
        print('Subsampling C2')
        other = nbinom.rvs(self.size, self.size/(self.size+self.other))
        self._C2 = tf.constant(other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)
        return self._C2

class TensorSignatureBootstrapSubsample(TensorSignatureBootstrap):
    """
    Tensor Signature that performs a denovo fitting on a resampled dataset.
    """
    def __init__(self, clu, snv, other, sub, N=None, **kwargs):
        assert(sub > 0 and sub < 1)

        self.clu = clu
        self.init = kwargs.get('init', self.clu.init)
        self.rank = clu.rank

        self.samples = int(np.floor(clu.samples * sub))
        self.sub = np.random.choice(np.arange(clu.samples), self.samples, replace=False)

        self.verbose = kwargs.get('verbose', True)
        self.size = self.clu.dset['tau'][()][..., self.init]

        self.snv = self.collapse_data(snv[..., self.sub])
        self.other = other[..., self.sub]
        self.N = self.collapse_data(np.concatenate([N]*2, axis=-4).reshape(3, 3, -1, 2, 96, 1))

        self.clu_dim = sorted([ var for var in list(self.clu.dset) if var.startswith('k') ])
        self.card = [ self.clu.dset[var][()].shape[0]+1 for var in self.clu_dim ]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(np.arange(self.card_prod), self.card)

        self.dtype = tf.float32
        self.p = kwargs.get('p', 96)
        self.tau = tf.constant(self.size, dtype=tf.float32)

        # initialize variables
        self.M
        self.S
        self.E
        self.T
        self.Chat1
        self.Chat2
        self.C1
        self.C2
        self.L1
        self.L2
        self.L

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.clu[STARTER_LEARNING_RATE]
        self.decay_learning_rate = self.clu[DECAY_LEARNING_RATE] # kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = self.clu[OPTIMIZER]
        self.minimize

    @define_scope
    def E(self):
        self.E0 = tf.Variable(self.clu.dset['E0'][()][..., self.sub, self.init], name='E0')
        self._E = tf.exp(self.E0, name='E')
        if self.verbose:
            print('E:', self._E.shape)

        return self._E


class TensorSignatureBootT(TensorSignatureBootstrap):
    """
    Tensor Signature that performs a denovo fitting on a resampled dataset.
    """
    def __init__(self, clu, snv, other, N=None, **kwargs):
        assert(kwargs.get('sub', 1) > 0 and kwargs.get('sub', 1) <= 1)

        self.clu = clu
        self.init = self.clu.init
        self.rank = clu.rank
        self.objective = self.clu[OBJECTIVE]
        self.collapse = kwargs.get('collapse', True)

        self.verbose = kwargs.get('verbose', True)
        self.size = self.clu['tau'][..., self.init]

        self.samples = int(np.floor(clu.samples * kwargs.get('sub', 1)))
        self.sub = np.random.choice(np.arange(clu.samples), self.samples, replace=False) if kwargs.get('sub', 1) < 1 else np.arange(self.samples)


        if self.verbose:
        	print('Samples {} ({})'.format(self.samples, kwargs.get('sub', 1)))

        # keep data
        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv[..., self.sub])
        else:
            self.snv = snv[..., self.sub]

        self.other = other[..., self.sub]
        if N is not None:
            if self.collapse:
                self.N = TensorSignature.collapse_data(N).reshape(3, 3,-1, 96, 1)
            else:
                self.N = N.reshape(3, 3,-1, 96, 1)
        else:
            self.N = None

        self.clu_dim = sorted([ var for var in list(self.clu.dset) if var.startswith('k') ])
        self.card = [ self.clu.dset[var][()].shape[0]+1 for var in self.clu_dim ]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(np.arange(self.card_prod), self.card)

        self.dtype = tf.float32
        self.p = kwargs.get('p', 96)
        self.q = other.shape[0]
        self.tau = tf.constant(self.size, dtype=tf.float32)
        self.epochs = kwargs.get(EPOCHS, 5000)

        # initialize variables
        self.M
        self.S
        self.E
        self.T
        self.Chat1
        self.Chat2
        self.C1
        self.C2
        self.L1
        self.L2
        self.L

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.clu[STARTER_LEARNING_RATE]
        self.decay_learning_rate = self.clu[DECAY_LEARNING_RATE] # kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = self.clu[OPTIMIZER]
        self.minimize

    # Override this from TensorSignatureBootstrap and
    # initialize with fresh random values because
    # of faulty confidence intervals in T

    @define_scope
    def T(self):

        self.T0 = tf.Variable(tf.truncated_normal([self.q-1, self.rank], dtype=self.dtype), name='T0')
        T1 = tf.nn.softmax(tf.concat([self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0), dim=0, name='T')
        self._T = T1 * (1-tf.reshape(self.M, (1, self.rank)))
        if self.verbose:
            print('T (reinitialized):', self._T.shape)

        return self._T

    @define_scope
    def E(self):
        self.E0 = tf.Variable(self.clu['E0'][..., self.sub, self.init], name='E0')
        self._E = tf.exp(self.E0, name='E')
        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def C1(self):
        print('-> no subsampling')
        self._C1 = tf.constant(self.snv.reshape(3, 3, -1, self.p, self.samples), dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        print('-> no subsampling')
        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)
        self._C2 = tf.constant(self.other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)
        return self._C2

class TensorSignatureRandomize(TensorSignatureBootstrap):
    """
    Randomizes a fraction of values in S0 and T0.
    """
    def __init__(self, snv, other, N, clu, **kwargs):
        assert(kwargs.get('sub', 1) > 0 and kwargs.get('sub', 1) <= 1)

        self.clu = clu
        self.init = self.clu.init
        self.rank = clu.rank
        self.objective = self.clu[OBJECTIVE]
        self.collapse = kwargs.get('collapse', True)

        self.verbose = kwargs.get('verbose', True)
        self.size = self.clu['tau'][..., self.init]

        self.samples = int(np.floor(clu.samples * kwargs.get('sub', 1)))
        self.sub = np.random.choice(np.arange(clu.samples), self.samples, replace=False) if kwargs.get('sub', 1) < 1 else np.arange(self.samples)

        if self.verbose:
        	print('Samples {} ({})'.format(self.samples, kwargs.get('sub', 1)))

        # keep data
        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv[..., self.sub])
        else:
            self.snv = snv[..., self.sub]

        self.other = other[..., self.sub]
        if N is not None:
            if self.collapse:
                self.N = TensorSignature.collapse_data(N).reshape(3, 3,-1, 96, 1)
            else:
                self.N = N.reshape(3, 3,-1, 96, 1)
        else:
            self.N = None

        self.clu_dim = sorted([ var for var in list(self.clu.dset) if var.startswith('k') ])
        self.card = [ self.clu.dset[var][()].shape[0]+1 for var in self.clu_dim ]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(np.arange(self.card_prod), self.card)

        self.dtype = tf.float32
        self.p = kwargs.get('p', 96)
        self.q = other.shape[0]
        self.tau = tf.constant(self.size, dtype=tf.float32)
        self.epochs = kwargs.get(EPOCHS, 5000)
        self.frac = kwargs.get('frac', .1)

        if self.verbose:
        	print('Distortion {}'.format(self.frac))

        # initialize variables
        self.M
        self.S
        self.E
        self.T
        self.Chat1
        self.Chat2
        self.C1
        self.C2
        self.L1
        self.L2
        self.L

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.clu[STARTER_LEARNING_RATE]
        self.decay_learning_rate = self.clu[DECAY_LEARNING_RATE] # kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = self.clu[OPTIMIZER]
        self.minimize

    # Override this from TensorSignatureBootstrap and
    # initialize with fresh random values because
    # of faulty confidence intervals in T
    @define_scope
    def S1(self):
        S0 = self.clu['S0'][..., self.init]
        #print(S0)
        indices = np.random.choice(np.arange(np.prod(S0.shape)), replace=False, size=int(np.floor(np.prod(S0.shape)*self.frac)))
        array_min, array_max = np.min(S0), np.max(S0)
        S0_mut = np.random.uniform(low=array_min, high=array_max, size=int(np.floor(np.prod(S0.shape)*self.frac))) # sample random values
        S0[np.unravel_index(indices, S0.shape)] = S0_mut
        #print('after')
        #print(S0)

        self.S0 = tf.Variable(S0, name='S0') # basic parameters [+/+, +/-] x [Pyr, Pur] x 96-1 x s
        S1 = tf.nn.softmax(tf.concat([self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2), dim=2, name='S1') # pad 0
        self._S1 = tf.reshape(
            tf.stack([
                S1[0, 0, :, :], S1[1, 0, :, :], 0.5 * tf.reduce_sum(S1[:, 0, :, :], axis=0),
                S1[1, 1, :, :], S1[0, 1, :, :], 0.5 * tf.reduce_sum(S1[:, 1, :, :], axis=0),
                0.5 * (S1[0, 0, :, :] + S1[1, 1, :, :]), 0.5 * (S1[1, 0, :, :] + S1[0, 1, :, :]), 0.25 * (tf.reduce_sum(S1, axis=(0, 1)))]),
            [3, 3, 1, self.p, self.rank])

        return self._S1

    @define_scope
    def T(self):
        T0 = self.clu['T0'][..., self.clu.init]
        #print(T0)
        indices = np.random.choice(np.arange(np.prod(T0.shape)), replace=False, size=int(np.floor(np.prod(T0.shape)*self.frac)))
        array_min, array_max = np.min(T0), np.max(T0)
        T0_mut = np.random.uniform(low=array_min, high=array_max, size=int(np.floor(np.prod(T0.shape)*self.frac))) # sample random values
        T0[np.unravel_index(indices, T0.shape)] = T0_mut
        #print('after')
        #print(T0)

        self.T0 = tf.Variable(T0, name='T0')
        T1 = tf.nn.softmax(tf.concat([self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0), dim=0, name='T')
        self._T = T1 * (1 - tf.reshape(self.M, (1, self.rank)))
        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def E(self):
        self.E0 = tf.Variable(self.clu['E0'][..., self.sub, self.init],
                              name='E0')
        self._E = tf.exp(self.E0, name='E')
        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def C1(self):
        # print('-> no subsampling')
        self._C1 = tf.constant(self.snv.reshape(3, 3, -1, self.p, self.samples),
                               dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        # print('-> no subsampling')
        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)
        self._C2 = tf.constant(self.other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)
        return self._C2

def arg():
    # Initiate a ArgumentParser Class
    parser = argparse.ArgumentParser(
        prog="train.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(DESCRIPTION))
    job_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    parser.add_argument(INPUT,
        metavar='STRING',
        type=str,
        help='path to h5 index')
    parser.add_argument(OUTPUT,
        metavar='DIR',
        type=str,
        help='output dir (writes ckpt and log folder to it)')
    parser.add_argument('h5',
        metavar='STR',
        type=str,
        help='path to h5 file which contains the initial fit',
        #choices=OBJECTIVE_CHOICE,
        )
    parser.add_argument('project',
        metavar='STR',
        type=str,
        help='project identifier',
        #choices=OBJECTIVE_CHOICE,
        )
    parser.add_argument('init',
        metavar='INT',
        type=int,
        nargs='+',
        help='start init end init (example 0 50 creates bootstrapsample 0-50)'
        )
    parser.add_argument('-v', '--' + VERBOSE,
        action='store_true',
        help='verbose mode')
    parser.add_argument('-norm',
        action='store_true',
        help='use normalisation constant M')
    parser.add_argument('-j',  '--' + JOB_NAME,
        metavar='STRING',
        type=str,
        help='job name',
        default=job_name)
    parser.add_argument('-i', '--' + ITERATION,
        metavar='INT',
        type=int,
        nargs='+',
        help='iteration (default = -1 MLE)',
        default=-1)
    parser.add_argument('-s', '--' + SEED,
        metavar='INT',
        type=int,
        help='seed',
        default=None)
    parser.add_argument('-sub',
        metavar='FLOAT',
        type=float,
        help='select a float > 0 && float < 1',
        default=False)
    parser.add_argument('-fn', '--' + FILENAME,
        metavar='STRING',
        type=str,
        help='enter string to save the job (default J_R_K_I)',
        default='J_R_K_I')
    parser.add_argument('-ds', '--' + DISPLAY_STEP,
        metavar='INT',
        type=int,
        help='progress updates (default = 1000)',
        default=100)

    params = parser.add_argument_group('model parameters')
    params.add_argument('-k', '--' + DISPERSION,
        metavar='FLOAT',
        type=float,
        help='dispersion factor (default = 50)',
        default=50)
    params.add_argument('-la', '--' + LAMBDA_A,
        metavar='INT',
        type=int,
        help='sigma of gaussian on signature activities (sigma = sqrt(1/lambda))',
        default=0)
    params.add_argument('-lts', '--' + LAMBDA_T,
        metavar='INT',
        type=int,
        help='sigma of gaussian prior on transcription',
        default=0)
    params.add_argument('-lrt', '--' + LAMBDA_R,
        metavar='INT',
        type=int,
        help='sigma of gaussian prior on transcription',
        default=0)
    params.add_argument('-lc', '--' + LAMBDA_C,
        metavar='INT',
        type=int,
        help='sigma of gaussian prior clustering coefficient',
        default=0)

    init = parser.add_argument_group('initialization parameters')
    init.add_argument('-op', '--' + OPTIMIZER,
                      metavar='STRING',
                      type=str,
                      default='ADAM',
                      help='optimizer (default ADAM)',
                      choices=OPTIMIZER_CHOICE)
    init.add_argument('-ep', '--' + EPOCHS,
        metavar='INT',
        type=int,
        default=5000,
        help='number of epochs')
    init.add_argument('-lr', '--' + STARTER_LEARNING_RATE,
        metavar='FLOAT',
        type=float,
        default=0.1,
        help='starter learning rate')
    init.add_argument('-ld', '--' + DECAY_LEARNING_RATE,
        metavar='STRING',
        type=str,
        default='exponential',
        help='learning rate (default exponential)',
        choices=DECAY_LEARNING_RATE_CHOICE)


    return parser

def main():
    import sys
    parser = arg()
    args = parser.parse_args(sys.argv[1:])
    params = vars(args)

    # load dataset
    if params[INPUT] == 'PCAWG_TCGA':
        snv, other, M, N = load_pcawg_tcga()
    elif params[INPUT] == 'PCAWG' or params[INPUT] == 'PCAWG_ACT':
        snv, other, N = load_pcawg()
    elif params[INPUT] ==  'PCAWG_CHROM':
        snv, other, M, N = load_pcawg_chrom()
    elif params[INPUT] ==  'MORITZ':
        snv, other, M, N = moritz_data()
    elif params[INPUT] == 'CHROMCLUST':
        snv, other, N = load_pcawg_chrom_clust2()
        print('clustered dim sum {}'.format(snv[:,:,:,1,:,:,:].sum()))
    elif params[INPUT] == 'NUCLEOSOME':
        snv, other, N = load_pcawg_nucleosome()
        #N = np.concatenate([N]*2, axis=-4).reshape(3,3,-1,2,96,1)
        print('clustered dim sum {}'.format(snv[:,:,:,1,:,:,:].sum()))
    elif params[INPUT] == 'MINMAJ':
        snv, other, N = load_pcawg_min_maj()
        print('clustered dim sum {}'.format(snv[:,:,:,1,:,:,:].sum()))
    elif params[INPUT] == 'MINMAJLIN':
        snv, other, N = load_pcawg_min_maj_lin()
        print('clustered dim sum {}'.format(snv[:,:,:,1,:,:,:].sum()))
    elif params[INPUT] == '8DIM':
        snv, other, N = load_pcawg_8dim()

    else:
        with h5.File(params[INPUT]) as fh:
            snv = fh['SNV'][()]
            other = fh['OTHER'][()]
            N = fh['N'][()]

    params['data_pts'] = np.array(np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other)))

    E = Experiment(h5.File(params['h5']), fast=True, merged=True)
    clu = E[params['project']]


    for i in np.arange(params['init'][0], params['init'][1]+1):
        if params[VERBOSE]:
        	pass
            #print('Bootstrap initialization {}/{}'.format(i, params[ITERATION][1]))
        params_copy = dict(params)
        params_copy[ITERATION] = i

        tf.reset_default_graph()

        if params['sub']:
                model = TensorSignatureSubsampleRandom(clu, snv, other, N, **params)
        else:
        	model = TensorSignatureBootstrap(
        		snv=snv,
        		other=other,
        		N=N,
        		clu=clu)

        model.fit_bootstrap_sample()
        data = {**params_copy, **model.bootstrap}
        #if (params[OBJECTIVE] == 'nbconst') | (params[OBJECTIVE] == 'nbvar') | (params[OBJECTIVE] == 'nbgamma'):
        #    post_processing(results, snv, other)

        fname = params['project'].lstrip("/").split('/')[0] + "_" + params['project'].lstrip("/").split('/')[1] + "_I={}".format(i)
        save_dict(data, os.path.join(params_copy[OUTPUT], fname+'.pkl'))



if __name__ == "__main__":
    main()
