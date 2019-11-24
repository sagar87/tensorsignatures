#!/usr/bin/env python
from pkg_resources import resource_filename
import os
import click
PATH = os.path.dirname(os.path.abspath(__file__))

# Data files

SIMULATION = resource_filename('tensorsignatures', 'data/sim_sig.txt')
OTHER = resource_filename('tensorsignatures', 'data/other.txt')
NORM = resource_filename('tensorsignatures', 'data/norm.h5')
PCAWG = resource_filename('tensorsignatures', 'data/pcawg.pkl')
# Parameters for tesnsorsignatures

RANK = 'rank'
SIZE = 'size'
OBJECTIVE = 'objective'
OBSERVATIONS = 'observations'

STARTER_LEARNING_RATE = 'starter_learning_rate'
DECAY_LEARNING_RATE = 'decay_learning_rate'
OPTIMIZER = 'optimizer'
EPOCHS = 'epochs'

LOG_STEP = 'log_step'
DISPLAY_STEP = 'display_step'

ID = 'id'
INIT = 'init'
SEED = 'seed'
VERBOSE = 'verbose'

PARAMS = [
    RANK,
    SIZE,
    OBJECTIVE,
    STARTER_LEARNING_RATE,
    DECAY_LEARNING_RATE,
    OPTIMIZER,
    EPOCHS,
    LOG_STEP,
    DISPLAY_STEP,
    ID,
    INIT,
    SEED,
    OBSERVATIONS
]

# Parameters of tensor signatures

S0 = '_S0'
a0 = '_a0'
b0 = '_b0'
ki = '_ki'
m0 = '_m0'
E0 = '_E0'
T0 = '_T0'


VARS = [
    S0,
    a0,
    b0,
    ki,
    m0,
    T0,
    E0
]

# Logs of tensorsignatures

LOG_EPOCHS = 'log_epochs'
LOG_LEARNING_RATE = 'log_learning_rate'
LOG_L = 'log_L'
LOG_L1 = 'log_L1'
LOG_L2 = 'log_L2'
SAMPLE_INDICES = 'sample_indices'

LOGS = [
    LOG_EPOCHS,
    LOG_LEARNING_RATE,
    LOG_L,
    LOG_L1,
    LOG_L2,
    SAMPLE_INDICES
]

LOG_STRING = 'Likelihood {lh:.2f} delta {delta:.2f} snv {snv:.2f} other {other:.2f} lr {lr:.4f}'
DUMP = PARAMS + VARS + LOGS

PARAMETERS = 'params'

# For CLI.py

INPUT = 'input'
OUTPUT = 'output'
NORMALIZE = 'norm'
COLLAPSE = 'collapse'

OBJECTIVE_CHOICE = click.Choice(['nbconst', 'poisson'])
OPTIMIZER_CHOICE = click.Choice(['ADAM', 'gradient_descent'])
DECAY_LEARNING_RATE_CHOICE = click.Choice(['exponential', 'constant'])


# CONSTANTS FOR util
LOG_LIKELIHOOD = 'log_likelihood'
AIC = 'AIC'
AIC_C = 'AIC_c'
BIC = 'BIC'

RECOGNITION = 'recognition'
SAMPLES = 'samples'
MUTATIONS = 'mutations'
DIMENSIONS = 'dimensions'

DOF = 'dof'
# intorduced for contrib scripts
ROW = 'row'
COL = 'col'

# contstants for bootstrap
SAMPLE_FRACTION = 2 / 3
DISTORTION = 0.1


ORI = ['+', '-', '*']
NUC = ['A', 'C', 'G', 'T']
PYR = ['C'] * 3 + ['T'] * 3
SUB = ['[C->A]', '[C->G]', '[C->T]', '[T->A]', '[T->C]', '[T->G]']

CHROMSTATES = {
    0: "NA",
    1: "active TSS",
    2: "flanking active TSS",
    3: "transcr. at gene 5'and 3'",
    4: "strong transcription",
    5: "weak transcription",
    6: "genic enhancers",
    7: "enhancers",
    8: "ZNF genes + repeats",
    9: "heterochromatin",
    10: "bivalent/poised TSS",
    11: "flanking bivalent TSS/Enh",
    12: "bivalent enhancer",
    13: "repressed polycomb",
    14: "weak repressed polycomb",
    15: "quiescent/low"
}

COLORPAIRS = [
    ['#a6cee3', '#2196F3'],
    ['dimgray', '#212121'],
    ['#fb9a99', '#f44336'],
    ['lightgrey', '#BDBDBD'],
    ['#b2df8a', '#8BC34A'],
    ['papayawhip', '#FFAB91']
]

LIGHT_PALETTE = [COLORPAIRS[0][0]] * 16 \
    + [COLORPAIRS[1][0]] * 16 \
    + [COLORPAIRS[2][0]] * 16 \
    + [COLORPAIRS[3][0]] * 16 \
    + [COLORPAIRS[4][0]] * 16 \
    + [COLORPAIRS[5][0]] * 16
DARK_PALETTE = [COLORPAIRS[0][1]] * 16 \
    + [COLORPAIRS[1][1]] * 16 \
    + [COLORPAIRS[2][1]] * 16 \
    + [COLORPAIRS[3][1]] * 16 \
    + [COLORPAIRS[4][1]] * 16 \
    + [COLORPAIRS[5][1]] * 16

# needed for plotting
SNV_MUT_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

SNV_MUT = ['       ' + n if m % 2 == 0 else n + '      ' for m, n in enumerate([i + k for j in list(
    map(lambda x: x.split('-')[0] + x.split('-')[1], SUB)) for i in NUC for k in NUC])]
SNV_MUT_ALT = [n for m, n in enumerate(
    [i + k for j in list(map(lambda x: x.split('-')[0] + x.split('-')[1], SUB)) for i in NUC for k in NUC])]
SNV_MUT_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

COLORS = ["#2196F3", "#212121", "#f44336", "#BDBDBD", "#8BC34A", "#FFAB91"]

# Min and max values for simulations
BMIN, BMAX = -0.6931471805599453, 0.6931471805599453
AMIN, AMAX = -0.6931471805599453, 0.6931471805599453
KMIN, KMAX = -2, 2
