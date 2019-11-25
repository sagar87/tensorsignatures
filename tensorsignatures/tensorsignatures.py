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
from tensorsignatures.util import Initialization
from tqdm import trange
import functools


def doublewrap(function):
    # A decorator decorator, allowing to use the decorator to be used without
    # parentheses if no arguments are provided. All arguments must be optional.
    # https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    # A decorator for functions that define TensorFlow operations. The wrapped
    # function will only be executed once. Subsequent calls to it will directly
    # return the result so that operations are added to the graph only once.
    # The operations added by the function live within a tf.variable_scope().
    # If this decorator is used with arguments, they will be forwarded to the
    # variable scope. The scope name defaults to the name of the wrapped
    # function.
    # https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2

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
    r"""Extracts tensor signatures from a SNV count tensor and a matrix
    containing other mutation types.

    Args:
        snv (array-like, shape :math:`(3, 3, (t_1+1), \dots, (t_l+1), p, n)`):
            Input SNV tensor; first and second dimension represent
            transcription and replication, while the last two dimensions
            contain p mutation types and n samples. Other dimensions may
            represent arbtrary genomic states.
        other (array-like, shape :math:`(q, n)`): Mutation count matrix with q
            mutation types and n samples.
        rank (:obj:`int`, :math:`2 \leq s < n`): Rank :math:`s` of the
            decomposition.
        N (array_like, shape :math:`(3, 3, (t_1+1), \dots, (t_l+1), p, 1)`):
            Optional normalization tensor containing trinucleotide frequencies
            for each genomic state.
        size (:obj:`int`, :math:`1 \leq \tau \leq + \inf`): Size parameter
            :math:`\tau` for negative binomial distribution.
        objective (:obj:`str`, :obj:`{'nbconst', 'poisson'}`): Likelihood
            distribution to model mutation counts. Currently, the negative
            binomial or poisson are supported.
        collapse (:obj:`bool`): Deprecated convinience function.
        starter_learning_rate (:obj:`float`): Starting Learning rate.
        decay_learning_rate (:obj:`str`, :obj:`{'exponential', 'constant'}`):
            Learning rate decay.
        optimizer (:obj:`str`, :obj:`{'ADAM', 'gradient_descent'}`): Allows
            to set the optimizer.
        epochs (:obj:`int`): Number of training epochs.
        log_step (:obj:`int`): Log freuqency.
        display_step (:obj:`int`): Update intervals of progress bar during.
        dtype (:obj:`dtype`): Allows to set tensorflow number type.
        verbose (:obj:`bool`): Verbose mode.
        id (:obj:`str`): Job id.
        init (:obj:`int`): Initialization.
        seed (:obj:`int`): Random seed.
    Returns:
        A tensorsignatures model.

    Examples:

    >>> from tensorsignatures.tensorsignatures import TensorSignature
    >>> model = TensorSignature(snv, other, rank = 5)
    >>> model.fit()

    """

    def __init__(self,
                 snv,
                 other,
                 rank,
                 N=None,
                 size=50,
                 objective='nbconst',
                 collapse=False,
                 starter_learning_rate=0.1,
                 decay_learning_rate='exponential',
                 optimizer='ADAM',
                 epochs=10000,
                 log_step=100,
                 display_step=100,
                 id='TSJOB',
                 init=0,
                 seed=None,
                 dtype=tf.float32,
                 verbose=True):
        # store hyperparameters
        self.rank = rank
        self.size = size
        self.objective = objective
        self.collapse = collapse
        self.starter_learning_rate = starter_learning_rate
        self.decay_learning_rate = decay_learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.log_step = log_step
        self.display_step = display_step
        self.id = id
        self.init = init
        self.seed = seed
        self.dtype = dtype
        self.verbose = verbose

        # hyperparams
        self.samples = snv.shape[-1]
        self.observations = np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other))

        # dimensions
        self.p = snv.shape[-2]
        self.q = other.shape[0]

        # intialize C1 and C2
        self.tau = tf.constant(self.size, dtype=self.dtype)

        # keep data
        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv)
        else:
            self.snv = snv

        self.other = other
        if N is not None:
            if self.collapse:
                self.N = TensorSignature.collapse_data(
                    N).reshape(3, 3, -1, 96, 1)
            else:
                self.N = N.reshape(3, 3, -1, 96, 1)
        else:
            self.N = None

        # clustering dims
        self.c = len(self.snv.shape) - 4
        self.card = list(self.snv.shape)[2: -2]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(
            np.arange(self.card_prod), self.card)

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

        # learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate
        self.minimize

        # intialize logs
        self.log_epochs = np.zeros(self.epochs // self.log_step)
        self.log_learning_rate = np.zeros(self.epochs // self.log_step)
        self.log_L = np.zeros(self.epochs // self.log_step)
        self.log_L1 = np.zeros(self.epochs // self.log_step)
        self.log_L2 = np.zeros(self.epochs // self.log_step)

    def indices_to_assignment(self, I, card):
        # Helper function to collapse additional genomic dimension
        card = np.array(card, copy=False)
        C = card.flatten()
        A = np.mod(
            np.floor(
                np.tile(I.flatten().T, (len(card), 1)).T /
                np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))),
                        (len(I), 1))),
            np.tile(C[::-1], (len(I), 1)))

        return A[:, ::-1]

    @define_scope
    def learning_rate(self):
        # Initialize learning rates.
        if self.decay_learning_rate == 'constant':
            self._learning_rate = tf.constant(
                self.starter_learning_rate, dtype=tf.float32, shape=())
        elif self.decay_learning_rate == 'exponential':
            self._learning_rate = tf.train.exponential_decay(
                self.starter_learning_rate, self.global_step, 1000, 0.95,
                staircase=True)

        return self._learning_rate

    @define_scope
    def minimize(self):
        # Initializes the minimizer.
        if self.optimizer == 'ADAM':
            self._minimize = tf.train.AdamOptimizer(
                self.learning_rate).minimize(-self.L, self.global_step)
        if self.optimizer == 'gradient_descent':
            self._minimize = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(-self.L, self.global_step)

        return self._minimize

    @define_scope
    def S1(self):
        # Initializes the SNV signature tensor.
        self.S0 = tf.Variable(
            tf.truncated_normal(
                [2, 2, self.p - 1, self.rank],
                dtype=self.dtype,
                seed=self.seed),
            name='S0')
        # apply softmax
        self.S0s = tf.nn.softmax(
            tf.concat(
                [self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2),
            dim=2, name='S1')
        # stack tensor
        self._S1 = tf.reshape(
            tf.stack([
                self.S0s[0, 0, :, :],
                self.S0s[1, 0, :, :],
                0.5 * tf.reduce_sum(self.S0s[:, 0, :, :], axis=0),
                self.S0s[1, 1, :, :],
                self.S0s[0, 1, :, :],
                0.5 * tf.reduce_sum(self.S0s[:, 1, :, :], axis=0),
                0.5 * (self.S0s[0, 0, :, :] + self.S0s[1, 1, :, :]),
                0.5 * (self.S0s[1, 0, :, :] + self.S0s[0, 1, :, :]),
                0.25 * (tf.reduce_sum(self.S0s, axis=(0, 1)))
            ]), (3, 3, 1, self.p, self.rank))

        if self.verbose:
            print('S1:', self._S1.shape)

        return self._S1

    @define_scope
    def T(self):
        # Initializes the signature matrix for other mutaiton types.
        # initialize T0 with values from a truncated normal
        self.T0 = tf.Variable(
            tf.truncated_normal(
                [self.q - 1, self.rank], dtype=self.dtype, seed=self.seed),
            name='T0')
        # apply softmax
        T1 = tf.nn.softmax(
            tf.concat(
                [self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0),
            dim=0, name='T')
        # factor mixing factor
        self._T = T1 * (1 - tf.reshape(self.M, (1, self.rank)))

        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def E(self):
        # Initializes exposures.
        self.E0 = tf.Variable(
            tf.truncated_normal(
                [self.rank, self.samples], dtype=self.dtype, seed=self.seed),
            name='E0')
        # exponentiate to satisfy non-negativity constraint
        self._E = tf.exp(self.E0, name='E')

        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def A(self):
        # Initializes signature activities transcription/replication.
        # self.a0[0,:] => to a_t
        # self.a0[1,:] => to a_r
        self.a0 = tf.Variable(
            tf.truncated_normal(
                [2, self.rank], dtype=tf.float32, seed=self.seed),
            name='a0')
        a1 = tf.exp(
            tf.reshape(
                tf.concat(
                    [self.a0, self.a0, tf.zeros([2, self.rank])], axis=0),
                (3, 2, self.rank)))

        # outer product
        self._A = tf.reshape(
            a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :],
            (3, 3, 1, 1, self.rank))

        if self.verbose:
            print('A:', self._A.shape)

        return self._A

    @define_scope
    def B(self):
        # Intializes transcription/replication biases.
        # self.b0[0,:] => b_t (coding / template)
        # self.b0[1,:] => b_r (lagging / leading)
        self.b0 = tf.Variable(
            tf.truncated_normal(
                [2, self.rank], dtype=tf.float32, seed=self.seed),
            name='b0')
        # stack variables
        self._B = tf.exp(tf.reshape(
            tf.stack([
                self.b0[0, :] + self.b0[1, :],
                self.b0[0, :] - self.b0[1, :],
                self.b0[0, :],
                self.b0[1, :] - self.b0[0, :],
                -self.b0[1, :] - self.b0[0, :],
                -self.b0[0, :],
                self.b0[1, :],
                -self.b0[1, :], tf.zeros(self.b0[0, :].shape)]),
            (3, 3, 1, 1, self.rank)))

        if self.verbose:
            print('B:', self._B.shape)

        return self._B

    @define_scope
    def K(self):
        # Initializes variables for generic tensorfactors
        self._clu_var = {}
        self._cbiases = {}

        for i in range(2, 2 + self.c):
            k = i - 2
            v = tf.Variable(
                tf.truncated_normal(
                    [self.card[k] - 1, self.rank],
                    dtype=self.dtype,
                    seed=self.seed),
                name='k{}'.format(k))

            self._clu_var[i - 2] = v
            shapes = [1 if k != i else self.card[k] for i in range(self.c)]
            dim = (1, 1, *shapes, 1, 1, self.rank)
            self._cbiases[i - 2] = tf.concat(
                [tf.zeros([1, self.rank], dtype=self.dtype), v], axis=0)

            if self.verbose:
                print('k{}:'.format(k), self._cbiases[i - 2].shape)

        final_tensor = []
        for r in range(self.idex.shape[0]):
            current_term = []
            for c in range(self.idex.shape[1]):
                current_term.append(
                    self._cbiases[c][self.idex[r, c].astype(int), :])

            final_tensor.append(tf.reduce_sum(tf.stack(current_term), axis=0))

        self._K = tf.exp(
            tf.reshape(tf.stack(final_tensor), (1, 1, -1, 1, self.rank)))

        if self.verbose:
            print('K:'.format(i), self._K.shape)

        return self._K

    @define_scope
    def M(self):
        # Initializes mixing factor variables.

        self.m0 = tf.Variable(
            tf.truncated_normal(
                [1, self.rank], dtype=self.dtype, seed=self.seed),
            name='m0')
        self.m1 = tf.sigmoid(self.m0, name='m1')
        self._M = tf.reshape(self.m1, (1, 1, 1, 1, self.rank))
        if self.verbose:
            print('m:', self._M.shape)

        return self._M

    @define_scope
    def S(self):
        # Initialize the final SNV tensor.
        self._S = self.S1 * self.A * self.B * self.K * self.M
        if self.verbose:
            print('S:', self._S.shape)
        return self._S

    @define_scope
    def C1(self):
        # Stores the count tensor.
        self._C1 = tf.constant(
            self.snv.reshape(3, 3, -1, self.p, self.samples), dtype=self.dtype)

        if self.verbose:
            print('C1:', self._C1.shape)

        return self._C1

    @define_scope
    def C2(self):
        # Stores the other mutation types tensor.
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
        # Compute predicted counts of the count tensor.
        self._Chat1 = tf.reshape(
            tf.matmul(tf.reshape(self.S, (-1, self.rank)), self.E),
            (3, 3, -1, 96, self.samples), name='Chat1')

        if self.N is not None:
            self._Chat1 *= (self.N.astype('float32') + 1e-6)
            if self.verbose:
                print('Multiplied N:', self.N.shape)

        if self.verbose:
            print('Chat1:', self._Chat1.shape)

        return self._Chat1

    @define_scope
    def Chat2(self):
        # Computes predicate counts for the ohter mutation count matrix.
        self._Chat2 = tf.matmul(self.T, self.E)
        if self.verbose:
            print('Chat2:', self._Chat2.shape)
        return self._Chat2

    @define_scope
    def L1ij(self):
        # Computes the log likelihood for each enty in SNV count tensor.
        if self.objective == 'nbconst':
            if self.verbose:
                print('Using negative binomial likelihood')
            self._L1ij = self.tau \
                * tf.log(self.tau) \
                - tf.lgamma(self.tau) \
                + tf.lgamma(self.C1 + self.tau) \
                + self.C1 * tf.log(self.Chat1) \
                - tf.log(self.Chat1 + self.tau) \
                * (self.tau + self.C1) \
                - tf.lgamma(self.C1 + 1)
        if self.objective == 'poisson':
            if self.verbose:
                print('Using poisson likelihood')
            self._L1ij = self.C1 \
                * tf.log(self.Chat1) \
                - self.Chat1 \
                - tf.lgamma(self.C1 + 1)

        return self._L1ij

    @define_scope
    def L2ij(self):
        # Computes the log likelhood for each entry in the matrix of other.
        # mutation types
        if self.objective == 'nbconst':
            self._L2ij = self.tau \
                * tf.log(self.tau) \
                - tf.lgamma(self.tau) \
                + tf.lgamma(self.C2 + self.tau) \
                + self.C2 * tf.log(self.Chat2) \
                - tf.log(self.Chat2 + self.tau) \
                * (self.tau + self.C2) \
                - tf.lgamma(self.C2 + 1)
        if self.objective == 'poisson':
            self._L2ij = self.C2 \
                * tf.log(self.Chat2) \
                - self.Chat2 \
                - tf.lgamma(self.C2 + 1)
        return self._L2ij

    @define_scope
    def L1(self):
        # Sums the log likelihood of each entry in L1ij.
        self._L1 = tf.reduce_sum(self.L1ij)
        return self._L1

    @define_scope
    def L2(self):
        # Sums the log likelihood of each entry in L2ij.
        self._L2 = tf.reduce_sum(self.L2ij * self.C2_nans)
        return self._L2

    @define_scope
    def L(self):
        # Sum of log likelihoods L1 and L2.
        self._L = self.L1 + self.L2
        return self._L

    def fit(self, sess=None):
        """Fits the model.

        Args:
            sess (:obj:`tensorflow.Session`): Tensorflow session, if None
                TensorSignatures will open new tensorflow session and close it
                after fitting the model.
        Returns:
            The tensoflow session.
        """
        # fits the model
        if sess is None:
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

        t = trange(self.epochs, desc='Progress', leave=True)
        previous_likelihood = 0
        for i in t:
            _ = sess.run(self.minimize)
            log_step = i // self.log_step

            if (i % self.log_step == 0):
                self.log_epochs[log_step] = i
                self.log_learning_rate[log_step] = sess.run(self.learning_rate)
                self.log_L[log_step] = sess.run(self.L)
                self.log_L1[log_step] = sess.run(self.L1)
                self.log_L2[log_step] = sess.run(self.L2)

            if (i % self.display_step == 0) and self.verbose:
                current_likelihood = sess.run(self.L)
                log_string = LOG_STRING.format(
                    lh=current_likelihood,
                    snv=sess.run(self.L1),
                    other=sess.run(self.L2),
                    lr=sess.run(self.learning_rate),
                    delta=current_likelihood - previous_likelihood)
                t.set_description(log_string)
                t.refresh()
                previous_likelihood = current_likelihood

        # save the loglikelihood value of the last iteration
        self.log_epochs[-1] = i
        self.log_learning_rate[-1] = sess.run(self.learning_rate)
        self.log_L[-1] = sess.run(self.L)
        self.log_L1[-1] = sess.run(self.L1)
        self.log_L2[-1] = sess.run(self.L2)

        self.result = Initialization(S0=sess.run(self.S0),
            a0=sess.run(self.a0), b0=sess.run(self.b0),
            ki=sess.run(self._clu_var), m0=sess.run(self.m0),
            T0=sess.run(self.T0), E0=sess.run(self.E0), rank=self.rank,
            size=self.size, objective=self.objective,
            starter_learning_rate=self.starter_learning_rate,
            decay_learning_rate=self.decay_learning_rate,
            optimizer=self.optimizer, epochs=self.epochs,
            log_step=self.log_step, display_step=self.display_step,
            observations=self.observations, id=self.id, init=self.init,
            seed=self.seed, log_epochs=self.log_epochs,
            log_learning_rate=self.log_learning_rate, log_L=self.log_L,
            log_L1=self.log_L1, log_L2=self.log_L2,
            sample_indices=np.arange(self.samples))

        if sess is None:
            sess.close()

        return self.result

    def get_tensors(self, sess):
        """Extracts signatures, exposures and tensor factors.

        Args:
            sesss (:obj:`tf.Session`): Tensorflow session in which the model
                was trained.
        Returns:
            A :obj:`dict` containing signatures, exposures and tensorfactors.
        """
        VARS = PARAMETERS + VARIABLES
        tensors = [var for var in dir(self) if (var.strip('_') in VARS)]

        data = {}
        for var in tensors:
            if (type(getattr(self, var)) == tf.Tensor or
                    type(getattr(self, var)) == tf.Variable):
                data[var.strip('_')] = np.array(sess.run(getattr(self, var)))
            elif (type(getattr(self, var)) == np.ndarray):
                data[var.strip('_')] = getattr(self, var)
            elif (type(getattr(self, var)) == int):
                data[var] = getattr(self, var)

        for k, v in self._clu_var.items():
            data['k{}'.format(k)] = np.array(sess.run(v))

        return data

    @staticmethod
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


class TensorSignatureRefit(TensorSignature):
    r"""Fits a set of signatures (ts.Initialization)to a dataset.

    Args:
        snv (array-like, shape :math:`(3, 3, (t_1+1), \dots, (t_l+1), p, n)`):
            Input SNV tensor; first and second dimension represent
            transcription and replication, while the last two dimensions
            contain p mutation types and n samples. Other dimensions may
            represent arbtrary genomic states.
        other (array-like, shape :math:`(q, n)`): Mutation count matrix with q
            mutation types and n samples.
        rank (:obj:`int`, :math:`2 \leq s < n`): Rank :math:`s` of the
            decomposition.
        N (array_like, shape :math:`(3, 3, (t_1+1), \dots, (t_l+1), p, 1)`):
            Optional normalization tensor containing trinucleotide frequencies
            for each genomic state.
        size (:obj:`int`, :math:`1 \leq \tau \leq + \inf`): Size parameter
            :math:`\tau` for negative binomial distribution.
        objective (:obj:`str`, :obj:`{'nbconst', 'poisson'}`): Likelihood
            distribution to model mutation counts. Currently, the negative
            binomial or poisson are supported.
        collapse (:obj:`bool`): Deprecated convinience function.
        starter_learning_rate (:obj:`float`): Starting Learning rate.
        decay_learning_rate (:obj:`str`, :obj:`{'exponential', 'constant'}`):
            Learning rate decay.
        optimizer (:obj:`str`, :obj:`{'ADAM', 'gradient_descent'}`): Allows
            to set the optimizer.
        epochs (:obj:`int`): Number of training epochs.
        log_step (:obj:`int`): Log freuqency.
        display_step (:obj:`int`): Update intervals of progress bar during.
        dtype (:obj:`dtype`): Allows to set tensorflow number type.
        verbose (:obj:`bool`): Verbose mode.
        id (:obj:`str`): Job id.
        init (:obj:`int`): Initialization.
        seed (:obj:`int`): Random seed.
    Returns:
        A tensorsignatures model.

    Examples:

    >>> from tensorsignatures.tensorsignatures import TensorSignature
    >>> model = TensorSignature(snv, other, rank = 5)
    >>> model.fit()

    """
    def __init__(self, snv, other, reference, N=None, **kwargs):
        self.ref = reference

        self.rank = self.ref.rank
        self.size = self.ref.size
        self.objective = self.ref.objective
        self.collapse = kwargs.get('collapse', False)
        self.starter_learning_rate = kwargs.get(
            STARTER_LEARNING_RATE, self.ref.starter_learning_rate)
        self.decay_learning_rate = kwargs.get(
            DECAY_LEARNING_RATE, self.ref.decay_learning_rate)
        self.optimizer = kwargs.get(OPTIMIZER, self.ref.optimizer)
        self.epochs = kwargs.get(EPOCHS, 5000)
        self.log_step = kwargs.get(LOG_STEP, self.ref.log_step)
        self.display_step = kwargs.get(DISPLAY_STEP, self.ref.log_step)
        self.id = kwargs.get(ID, self.ref.id)
        self.init = kwargs.get(INIT, 0)
        self.seed = kwargs.get(SEED, None)
        self.dtype = kwargs.get('dtype', tf.float32)
        self.verbose = kwargs.get('verbose', False)

        # hyper
        self.samples = snv.shape[-1]
        self.observations = self.ref.observations

        # dimensions
        self.p = snv.shape[-2]
        self.q = other.shape[0]

        self.tau = tf.constant(self.ref.size, dtype=self.dtype)

        if self.collapse:
            self.snv = TensorSignature.collapse_data(snv)
        else:
            self.snv = snv

        self.other = other
        if N is not None:
            if self.collapse:
                self.N = TensorSignature.collapse_data(
                    N).reshape(3, 3, -1, 96, 1)
            else:
                self.N = N.reshape(3, 3, -1, 96, 1)
        else:
            self.N = None

        self.card = [k + 1 for k in self.ref._kdim]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(
            np.arange(self.card_prod), self.card)

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

        # learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate
        self.minimize

        # intialize logs
        self.log_epochs = np.zeros(self.epochs // self.ref.log_step)
        self.log_learning_rate = np.zeros(self.epochs // self.ref.log_step)
        self.log_L = np.zeros(self.epochs // self.ref.log_step)
        self.log_L1 = np.zeros(self.epochs // self.ref.log_step)
        self.log_L2 = np.zeros(self.epochs // self.ref.log_step)

    @define_scope
    def A(self):
        # sets signature activities for transcription and replication
        self.a0 = tf.constant(self.ref._a0[..., 0], name='a0')
        a1 = tf.exp(
            tf.reshape(
                tf.concat(
                    [self.a0, self.a0, tf.zeros([2, self.rank])], axis=0),
                (3, 2, self.rank)))

        self._A = tf.reshape(
            a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :],
            (3, 3, 1, 1, self.rank))

        if self.verbose:
            print('A:', self._A.shape)

        return self._A

    @define_scope
    def B(self):
        # sets transcriptional and replicational biases
        self.b0 = tf.constant(
            self.ref._b0[..., 0], name='b0')
        self._B = tf.exp(tf.reshape(
            tf.stack([
                self.b0[0, :] + self.b0[1, :],
                self.b0[0, :] - self.b0[1, :],
                self.b0[0, :],
                self.b0[1, :] - self.b0[0, :],
                -self.b0[1, :] - self.b0[0, :],
                -self.b0[0, :],
                self.b0[1, :], -self.b0[1, :],
                tf.zeros(self.b0[0, :].shape)]),
            (3, 3, 1, 1, self.rank)))

        if self.verbose:
            print('B:', self._B.shape)

        return self._B

    @define_scope
    def S1(self):
        self.S0 = tf.constant(self.ref._S0[..., 0], name='S0')
        S1 = tf.nn.softmax(
            tf.concat([self.S0, tf.zeros([2, 2, 1, self.rank])], axis=2),
            dim=2, name='S1')

        # stack the tensor
        self._S1 = tf.reshape(
            tf.stack([
                S1[0, 0, :, :],
                S1[1, 0, :, :],
                0.5 * tf.reduce_sum(S1[:, 0, :, :], axis=0),
                S1[1, 1, :, :],
                S1[0, 1, :, :],
                0.5 * tf.reduce_sum(S1[:, 1, :, :], axis=0),
                0.5 * (S1[0, 0, :, :] + S1[1, 1, :, :]),
                0.5 * (S1[1, 0, :, :] + S1[0, 1, :, :]),
                0.25 * (tf.reduce_sum(S1, axis=(0, 1)))]),
            (3, 3, 1, self.p, self.rank))

        return self._S1

    @define_scope
    def E(self):
        # exposures are intialized randomly
        self.E0 = tf.Variable(
            tf.truncated_normal(
                [self.ref.rank, self.snv.shape[-1]], dtype=self.dtype),
            name='E0')
        self._E = tf.exp(self.E0, name='E')

        if self.verbose:
            print('E:', self._E.shape)

        return self._E

    @define_scope
    def K(self):
        self._clu_var = {}
        self._cbiases = {}

        for k, v in self.ref._ki.items():
            v = tf.constant(v[..., 0], name='k{}'.format(k))

            self._clu_var[k] = v
            self._cbiases[k] = tf.concat(
                [tf.zeros([1, self.rank], dtype=self.dtype), v], axis=0)

            if self.verbose:
                print('k{}:'.format(k), self._cbiases[k].shape)

        final_tensor = []
        for r in range(self.idex.shape[0]):
            current_term = []
            for c in range(self.idex.shape[1]):
                current_term.append(
                    self._cbiases[c][self.idex[r, c].astype(int), :])

            final_tensor.append(tf.reduce_sum(tf.stack(current_term), axis=0))

        self._K = tf.exp(
            tf.reshape(tf.stack(final_tensor), (1, 1, -1, 1, self.rank)))
        if self.verbose:
            print('K:', self._K.shape)
        return self._K

    @define_scope
    def M(self):
        self.m0 = tf.constant(
            self.ref._m0[..., 0], name='m0')
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
        self.T0 = tf.constant(
            self.ref._T0[..., 0], name='T0')

        T1 = tf.nn.softmax(
            tf.concat(
                [self.T0, tf.zeros([1, self.rank], dtype=self.dtype)], axis=0),
            dim=0, name='T')

        self._T = T1 * (1 - tf.reshape(self.M, (1, self.rank)))

        if self.verbose:
            print('T:', self._T.shape)

        return self._T

    @define_scope
    def Chat1(self):
        self._Chat1 = tf.reshape(
            tf.matmul(tf.reshape(self.S, (-1, self.rank)), self.E),
            (3, 3, -1, self.p, self.samples))

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
        self._C1 = tf.constant(
            self.snv.reshape(3, 3, -1, self.p, self.samples), dtype=self.dtype)

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
