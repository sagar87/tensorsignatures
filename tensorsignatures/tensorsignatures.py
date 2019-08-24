# -*- coding: utf-8 -*-
"""
.. module:: tensorsignatures
   :synopsis: TensorSignatures main module
.. moduleauthor:: Harald Vohringer <github.com/sagar87>
"""


import sys
import os
import tensorflow as tf
import numpy as np
import h5py as h5
from tensorsignatures.config import *
import functools

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class TensorSignature(object):
    r"""Heart of TensorSignatures.

    Args:
        snv: Input SNV tensor with shape :math:`\mathbf{C}^{\text{SNV}}\in
            \mathbb{R}^{3\times 3\times t_1\times\dots\times t_m\times p
            \times n}`. TensorSignature expects the first two dimension to
            represent transcription and replication, while the last two
            dimensions contain the mutation types :math:`p` and samples
            :math:`n`. Dimensions :math:`t_1,\dots, t_m` may represent arbtrary
            genomic dimensions.
        other: Mutation count matrix with :math:`q` mutation types and
            :math:`n` samples.
        N:  Optional normalization tensor :math:`\mathbf{N}\in
            \mathbb{R}^{3\times 3\times t_1\times\dots\times t_m\times p
            \times 1}` containing trinucleotide frequencies for each genomic
            state.
        objective: likelihood function with which mutation counts are
            modeled. Currently, the negative binomial or poisson are supported.
    """

    def __init__(self, snv, other, N=None, objective='nbconst',
        dtype=tf.float32, verbose=True, seed=None, **kwargs):
        assert(len(snv.shape) >= 5)
        self.verbose = verbose
        self.seed = seed
        self.dtype = dtype
        self.objective = objective
        self.collapse = kwargs.get('collapse', True)
        #TODO: include this field (?)
        #self.data_pts = np.array(np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other)))

        # hyperparams
        self.rank = kwargs.get(RANK, 16)
        self.samples = snv.shape[-1]
        self.size = kwargs.get(DISPERSION, 25)

        # dimensions
        self.p = snv.shape[-2]
        self.q = other.shape[0]

        # intialize C1 and C2
        self.tau = tf.constant(kwargs.get(DISPERSION, 25), dtype=self.dtype)

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
                self.N = N.reshape(3, 3,-1, 96, 1)
        else:
            self.N = None

        # clustering dims
        self.c = len(self.snv.shape)-4
        self.card = list(self.snv.shape)[2:-2] # cardinality of clustered dimensions
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(np.arange(self.card_prod), self.card)

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
        self.starter_learning_rate = kwargs.get(STARTER_LEARNING_RATE, 0.1)
        self.decay_learning_rate = kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = kwargs.get(OPTIMIZER, 'ADAM')
        self.minimize

    @staticmethod
    def collapse_data(snv):
        col1 = snv[[slice(None)]*(snv.ndim-3)+[0]+[slice(None)]*2]
        col2 = []
        for i, j in [(1, 1), (1, 0), (1, 2), (0, 1), (0, 0), (0, 2), (2, 1), (2, 0), (2, 2)]:
            col2.append(snv[[i, j]+[slice(None)]*(snv.ndim-5)+[1]+[slice(None)]*2])
        col2 = np.stack(col2).reshape(col1.shape)
        return col1 + col2

    def indices_to_assignment(self, I, card):
        """
        :param - I: a list of indices
        :param list card: a list of the cardinalities of the variables in the
        assignment
        """
        I = np.array(I, copy=False)
        card = np.array(card, copy=False)
        C = card.flatten()
        A = np.mod(np.floor(np.tile(I.flatten().T, (len(card), 1)).T / np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))), (len(I), 1))), np.tile(C[::-1], (len(I), 1)))

        return A[:, ::-1]

    def get_tensors(self, sess):
        tensors = [ var for var in dir(self) if (var.strip('_') in PARAMETERS+VARIABLES) ]

        data = {}
        for var in tensors:
            if (type(getattr(self, var)) == tf.Tensor) or (type(getattr(self, var)) == tf.Variable):
                data[var.strip('_')] = np.array(sess.run(getattr(self, var)))
            elif (type(getattr(self, var)) == np.ndarray):
                data[var.strip('_')] = getattr(self, var)
            elif (type(getattr(self, var)) == int):
                data[var] = getattr(self, var)

        for k, v in self._clu_var.items():
            data['k{}'.format(k)] = np.array(sess.run(v))

        return data

    @define_scope
    def learning_rate(self):
        if self.decay_learning_rate == 'constant':
            self._learning_rate = tf.constant(self.starter_learning_rate, dtype=tf.float32, shape=())
        elif self.decay_learning_rate == 'exponential':
            self._learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 1000, 0.95, staircase=True)

        return self._learning_rate

    @define_scope
    def minimize(self):
        if self.optimizer == 'ADAM':
            self._minimize = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.L, self.global_step)
        if self.optimizer == 'gradient_descent':
            self._minimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-self.L, self.global_step)

        return self._minimize

    @define_scope
    def S1(self):
        """
        Initializes the signature tensor
        """
        self.S0 = tf.Variable(tf.truncated_normal([2, 2, self.p-1, self.rank], dtype=self.dtype, seed=self.seed), name='S0') # basic parameters [+/+, +/-] x [Pyr, Pur] x 96-1 x s
        self.S0s = tf.nn.softmax(tf.concat([self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2), dim=2, name='S1') # pad 0
        self._S1 = tf.reshape(
            tf.stack([
                self.S0s[0, 0, :, :], self.S0s[1, 0, :, :], 0.5 * tf.reduce_sum(self.S0s[:, 0, :, :], axis=0),
                self.S0s[1, 1, :, :], self.S0s[0, 1, :, :], 0.5 * tf.reduce_sum(self.S0s[:, 1, :, :], axis=0),
                0.5 * (self.S0s[0, 0, :, :] + self.S0s[1, 1, :, :]), 0.5 * (self.S0s[1, 0, :, :] + self.S0s[0, 1, :, :]), 0.25 * (tf.reduce_sum(self.S0s, axis=(0, 1)))]),
            [3, 3, 1, self.p, self.rank])

        if self.verbose:
            print('S1:', self._S1.shape)

        return self._S1

    @define_scope
    def T(self):

        self.T0 = tf.Variable(tf.truncated_normal([self.q-1, self.rank], dtype=self.dtype, seed=self.seed), name='T0')
        T1 = tf.nn.softmax(tf.concat([self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0), dim=0, name='T')
        self._T = T1 * (1-tf.reshape(self.M, (1, self.rank)))
        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def E(self):
        """
        Exposures.
        """

        self.E0 = tf.Variable(tf.truncated_normal([self.rank, self.samples], dtype=self.dtype, seed=self.seed), name='E0')
        self._E = tf.exp(self.E0, name='E')
        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def A(self):
        """
        Initialize amplitudes.
        """
        self.a0 = tf.Variable(tf.truncated_normal([2, self.rank], dtype=tf.float32, seed=self.seed), name='a0')
        a1 = tf.exp(tf.reshape(tf.concat([self.a0, self.a0, tf.zeros([2, self.rank])], axis=0), (3, 2, self.rank)))
        a2 = a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :] # outer product
        self._A = tf.reshape(a2, [3, 3, 1, 1, self.rank])

        if self.verbose:
            print('A:', self._A.shape)

        return self._A

    @define_scope
    def B(self):
        """
        initializes coefficients capturing pyrimidine/purine biases
        in transcription and replication.
        """
        self.b0 = tf.Variable(tf.truncated_normal([2, self.rank], dtype=tf.float32, seed=self.seed), name='b0') # Pyrimidine - Purine bias x 2 (TS, RS) X s
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
    def K(self):
        """
        Initializes the variables.
        """
        self._clu_var = {}
        self._cbiases = {}

        for i in range(2, 2+self.c):
            k = i-2
            v = tf.Variable(tf.truncated_normal([self.card[k]-1, self.rank], dtype=self.dtype, seed=self.seed), name='k{}'.format(k))

            self._clu_var[i-2] = v
            dim = (1, 1, *[1 if k != i else self.card[k] for i in range(self.c)], 1, 1, self.rank)
            self._cbiases[i-2] = tf.concat([tf.zeros([1, self.rank], dtype=self.dtype), v], axis=0)

            if self.verbose:
                print('k{}:'.format(k), self._cbiases[i-2].shape)

        final_tensor = []
        for r in range(self.idex.shape[0]):
            current_term = []
            for c in range(self.idex.shape[1]):
                current_term.append(self._cbiases[c][self.idex[r, c].astype(int), :])

            final_tensor.append(tf.reduce_sum(tf.stack(current_term), axis=0))

        self._K = tf.exp(tf.reshape(tf.stack(final_tensor), (1, 1, -1, 1, self.rank)))

        if self.verbose:
            print('K:'.format(i), self._K.shape)

        return self._K

    @define_scope
    def M(self):
        self.m0 = tf.Variable(tf.truncated_normal([1, self.rank], dtype=self.dtype, seed=self.seed), name='m0')
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
    def C1(self):
        self._C1 = tf.constant(self.snv.reshape(3, 3, -1, self.p, self.samples), dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)
        self._C2 = tf.constant(self.other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)
        return self._C2

    @define_scope
    def Chat1(self):
        self._Chat1 = tf.reshape(tf.matmul(tf.reshape(self.S, (-1, self.rank)), self.E), (3, 3, -1, 96, self.samples), name='Chat1')

        if self.N is not None:
            self._Chat1 *= (self.N.astype('float32') + 1e-6)
            if self.verbose:
                print('Multiplied N:', self.N.shape)

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
    def L1ij(self):
        if self.objective == 'nbconst':
            if self.verbose:
                print('Using negative binomial likelihood')
            self._L1ij = self.tau * tf.log(self.tau) - tf.lgamma(self.tau) + tf.lgamma(self.C1+self.tau) + self.C1 * tf.log(self.Chat1) - tf.log(self.Chat1+self.tau) * (self.tau + self.C1) - tf.lgamma(self.C1+1)
        if self.objective == 'poisson':
            if self.verbose:
                print('Using poisson likelihood')
            self._L1ij = self.C1 * tf.log(self.Chat1) - self.Chat1 - tf.lgamma(self.C1+1)

        return self._L1ij

    @define_scope
    def L2ij(self):
        if self.objective == 'nbconst':
            self._L2ij = self.tau * tf.log(self.tau) - tf.lgamma(self.tau) + tf.lgamma(self.C2+self.tau) + self.C2 * tf.log(self.Chat2) - tf.log(self.Chat2+self.tau) * (self.tau + self.C2) - tf.lgamma(self.C2+1)
        if self.objective == 'poisson':
            self._L2ij = self.C2 * tf.log(self.Chat2) - self.Chat2 - tf.lgamma(self.C2+1)
        return self._L2ij

    @define_scope
    def L1(self):
        self._L1 = tf.reduce_sum(self.L1ij)
        return self._L1

    @define_scope
    def L2(self):
        self._L2 = tf.reduce_sum(self.L2ij * self.C2_nans)
        return self._L2

    @define_scope
    def L(self):
        self._L = self.L1 + self.L2
        return self._L

class TensorSignatureRefit(TensorSignature):
    def __init__(self, snv, other, N, clu, **kwargs):
        self.clu = clu
        self.init = kwargs.get('init', self.clu.init)
        self.rank = clu.rank
        self.samples = snv.shape[-1]
        self.collapse = kwargs.get('collapse', True)

        self.verbose = kwargs.get('verbose', True)
        self.size = self.clu['tau'][..., self.clu.init]

        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv)
        else:
            self.snv = snv

        self.other = other
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
        self.tau = tf.constant(self.size, dtype=tf.float32)
        self.epochs = kwargs.get(EPOCHS, 5000)
        self.objective = self.clu[OBJECTIVE]

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
    def A(self):
        self.a0 = tf.constant(self.clu['a0'][..., self.init], name='a0')
        a1 = tf.exp(tf.reshape(tf.concat([self.a0, self.a0, tf.zeros([2, self.rank])], axis=0), (3, 2, self.rank)))
        a2 = a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :] # outer product
        self._A = tf.reshape(a2, [3, 3, 1, 1, self.rank])
        if self.verbose:
            print('A:', self._A.shape)

        return self._A

    @define_scope
    def B(self):
        self.b0 = tf.constant(self.clu['b0'][..., self.init], name='b0') # Pyrimidine - Purine bias x 2 (TS, RS) X s
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
        self.S0 = tf.constant(self.clu['S0'][..., self.init], name='S0') # basic parameters [+/+, +/-] x [Pyr, Pur] x 96-1 x s
        S1 = tf.nn.softmax(tf.concat([self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2), dim=2, name='S1') # pad 0
        self._S1 = tf.reshape(
            tf.stack([
                S1[0, 0, :, :], S1[1, 0, :, :], 0.5 * tf.reduce_sum(S1[:, 0, :, :], axis=0),
                S1[1, 1, :, :], S1[0, 1, :, :], 0.5 * tf.reduce_sum(S1[:, 1, :, :], axis=0),
                0.5 * (S1[0, 0, :, :] + S1[1, 1, :, :]), 0.5 * (S1[1, 0, :, :] + S1[0, 1, :, :]), 0.25 * (tf.reduce_sum(S1, axis=(0, 1)))]),
            [3, 3, 1, self.p, self.rank])

        return self._S1

    @define_scope
    def E(self):
        self.E0 = tf.Variable(tf.truncated_normal([self.clu.rank, self.snv.shape[-1]], dtype=self.dtype), name='E0')
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
            v = tf.constant(self.clu[clu][..., self.init], name='k{}'.format(k))

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
        self.m0 = tf.constant(self.clu['m0'][..., self.clu.init], name='m0')
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
        self.T0 = tf.constant(self.clu['T0'][..., self.clu.init], name='T0')
        T1 = tf.nn.softmax(tf.concat([self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0), dim=0, name='T')
        self._T = T1 * (1-tf.reshape(self.M, (1, self.rank)))
        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def Chat1(self):
        self._Chat1 = tf.reshape(tf.matmul(tf.reshape(self.S, (-1, self.rank)), self.E), [3, 3, -1, self.p, self.samples])

        if self.N is not None:
            self._Chat1 *= (self.N.astype('float32') + 1e-6)
            if self.verbose:
                print('Multiplied N:', self.N.shape)

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
        self._C1 = tf.constant(self.snv.reshape(3, 3, -1, self.p, self.samples), dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        sub_set = np.ones_like(self.other)
        sub_set[np.where(np.isnan(self.other))] = 0
        self.other[np.where(np.isnan(self.other))] = 0
        self.C2_nans = tf.constant(sub_set, dtype=self.dtype)
        self._C2 = tf.constant(self.other, dtype=self.dtype)
        if self.verbose:
            print('C2:', self._C2.shape)

        return self._C2

class TensorSignatures2D(TensorSignature):
    def __init__(self, matrix, **kwargs):
        self.p, self.samples = matrix.shape
        self.verbose = kwargs.get(VERBOSE, True)
        self.rank = kwargs.get(RANK, 16)
        self.dtype = kwargs.get("dtype", tf.float32)
        self.seed = kwargs.get(SEED, None)
        self.objective = kwargs.get(OBJECTIVE, 'nbconst')

        self.snv = matrix
        # setup model
        self.size = kwargs.get(DISPERSION, 25)
        self.tau = tf.constant(kwargs.get(DISPERSION, 25), dtype=self.dtype)

        self.S
        self.E

        # matrix
        self.Chat1
        self.C1

        self.L1
        self.L

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = kwargs.get(STARTER_LEARNING_RATE, 0.1)
        self.decay_learning_rate = kwargs.get(DECAY_LEARNING_RATE, 'exponential')
        self.learning_rate

        self.optimizer = kwargs.get(OPTIMIZER, 'ADAM')
        self.minimize

    @define_scope
    def Chat1(self):
        self._Chat1 = tf.matmul(self.S, self.E)

        if self.verbose:
            print('Chat1:', self._Chat1.shape)

        return self._Chat1

    @define_scope
    def C1(self):
        self._C1 = tf.constant(self.snv, dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def S(self):
        """
        Initializes the signature tensor
        """
        #sig = tf.Variable(tf.truncated_normal([dim-1, self.r], dtype=self.dtype))
        #sig = tf.nn.softmax(tf.concat([sig, tf.zeros([1, self.r], dtype=self.dtype)], axis=0), dim=0)

        self.S0 = tf.Variable(tf.truncated_normal([self.p-1, self.rank], dtype=self.dtype, seed=self.seed), name='S0')
        self._S = tf.nn.softmax(tf.concat([self.S0, tf.zeros([1, self.rank])], axis=0), dim=0, name='S1')
        if self.verbose:
            print('S1:', self._S.shape)

        return self._S

    @define_scope
    def L(self):
        self._L = self.L1
        return self._L

    def get_tensors(self, sess):
        tensors = ['S', 'E', 'tau']

        data = {}
        for var in tensors:
            if (type(getattr(self, var)) == tf.Tensor) or (type(getattr(self, var)) == tf.Variable):
                data[var.strip('_')] = np.array(sess.run(getattr(self, var)))
            elif (type(getattr(self, var)) == np.ndarray):
                data[var.strip('_')] = getattr(self, var)
            elif (type(getattr(self, var)) == int):
                data[var] = getattr(self, var)

        return data
