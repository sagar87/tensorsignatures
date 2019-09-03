#!/usr/bin/env python
from pkg_resources import resource_filename
import os
import click
PATH = os.path.dirname(os.path.abspath(__file__))

# Path to PCAWG data
PCAWG_SNV = resource_filename('tensorsignatures', 'data/pcawgSNV.h5')
PCAWG_CHROM_SNV = resource_filename('tensorsignatures', 'data/pcawgChromSNV.h5')
PCAWG_CHROM_CLUST = resource_filename('tensorsignatures', 'data/pcawgChromClust4.h5')
PCAWG_TGCA = resource_filename('tensorsignatures', 'data/new_data.h5')
PCAWG_NUC = resource_filename('tensorsignatures', 'data/pcawgNuc.h5')
PCAWG_MINMAJ = resource_filename('tensorsignatures', 'data/pcawgMinMaj.h5')
PCAWG_MINMAJLIN = resource_filename('tensorsignatures', 'data/pcawgMinMajLin.h5') #minMajLinTNC.csv
PCAWG_8DIM = resource_filename('tensorsignatures', 'data/pcawg8Dim.h5')

PCAWG_TNC = resource_filename('tensorsignatures', 'data/tncTR.csv')
PCAWG_CHRM_TNC = resource_filename('tensorsignatures', 'data/chrmTNC3.csv')
PCAWG_CHRM_TNC2 = resource_filename('tensorsignatures', 'data/cutoff.4.csv')
PCAWG_NUC_TNC = resource_filename('tensorsignatures', 'data/nucTNC.csv')
PCAWG_MINMAJ_TNC = resource_filename('tensorsignatures', 'data/minMajTNC.csv')
PCAWG_MINMAJLIN_TNC = resource_filename('tensorsignatures', 'data/minMajLinTNC.csv')
PCAWG_8DIM_TNC = resource_filename('tensorsignatures', 'data/8dimTNC.csv')

CHROM_CLUST_TNC = resource_filename('tensorsignatures', 'data/cutoff.4.unnormed.192.csv')
CHROM_CLUST_IDX = resource_filename('tensorsignatures', 'data/indeces.csv')


PCAWG_TNC2 = resource_filename('tensorsignatures', 'data/tnc_counts.csv')
FEATURE_TABLE = resource_filename('tensorsignatures', 'data/features.csv')
SIMULATION = resource_filename('tensorsignatures', 'data/sim_sig.txt')
OTHER = resource_filename('tensorsignatures', 'data/other.txt')
ORGANS = resource_filename('tensorsignatures', 'data/samples_sorted.csv')

OTHER_MUTATIONS = resource_filename('tensorsignatures', 'data/othermut.csv')
MORITZ_OTHER = resource_filename('tensorsignatures', 'data/allTable.txt')
MORITZ_TNC = resource_filename('tensorsignatures', 'data/tncTransRep.txt')

PCAWG_SIGNATURES = resource_filename('tensorsignatures', 'data/pcawg_signatures.txt')
PCAWG_EXPOSURES = resource_filename('tensorsignatures', 'data/pcawg_exposures.txt')
PCAWG_SAMPLES = resource_filename('tensorsignatures', 'data/pcawg_samples.txt')
PCAWG_TRINUC = resource_filename('tensorsignatures', 'data/pcawg_trinucleotides.txt')
PCAWG_LABELS = resource_filename('tensorsignatures', 'data/pcawg_labels.txt')
PCAWG_TUMOUR = resource_filename('tensorsignatures', 'data/pcawg_tumour.txt')
PCAWG_COLORS = resource_filename('tensorsignatures', 'data/pcawg_colors.txt')

# PCAWG_DATA = {'PCAWG_COUNTS': PCAWG_COUNTS,
#               'PCAWG_SIGNATURES': PCAWG_SIGNATURES,
#               'PCAWG_EXPOSURES': PCAWG_EXPOSURES,
#               'PCAWG_SAMPLES': PCAWG_SAMPLES,
#               'PCAWG_TRINUC': PCAWG_TRINUC,
#               'PCAWG_LABELS': PCAWG_LABELS}

SEED = 'seed'
RANK = 'rank'
SIZE = 'size'
VERBOSE = 'verbose'
OBSERVATIONS = 'observations'
ID = 'id'
INIT = 'init'

# parameters saved during training
OBJECTIVE = 'objective'
OBJECTIVE_CHOICE = click.Choice(['nbconst', 'poisson'])
OPTIMIZER = 'optimizer'
OPTIMIZER_CHOICE = click.Choice(['ADAM', 'gradient_descent'])
DISPERSION = 'dispersion'
LEARNING_RATE = 'learning_rate'


PREFIX = 'prefix'
SUFFIX = 'suffix'

LOG_STEP = 'log_step'
LOG_EPOCHS = 'log_epochs'
LOG_LEARNING_RATE = 'log_learning_rate'
LOG_L = 'log_L'
LOG_L1 = 'log_L1'
LOG_L2 = 'log_L2'
LOG_STRING = 'Likelihood {lh:.2f} delta {delta:.2f} snv {snv:.2f} other {other:.2f} lr {lr:.4f}'

INPUT = 'input'
OUTPUT = 'output'
LOGS = 'logs'

EPOCHS = 'epochs'
STARTER_LEARNING_RATE = 'starter_learning_rate'
DECAY_LEARNING_RATE = 'decay_learning_rate'
DECAY_LEARNING_RATE_CHOICE = click.Choice(['exponential', 'constant'])

JOB_NAME = 'job_name'
ITERATION = 'iteration'
DISPLAY_STEP = 'display_step'

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

LOGS = [
    LOG_EPOCHS,
    LOG_LEARNING_RATE,
    LOG_L,
    LOG_L1,
    LOG_L2
]


S0 = 'S0'
a0 = 'a0'
b0 = 'b0'
k0 = '_k0'
m0 = 'm0'
E0 = 'E0'
T0 = 'T0'


VARS = [
    S0,
    a0,
    b0,
    k0,
    m0,
    T0,
    E0
]

DUMP = PARAMS + VARS + LOGS

DUMPO = [
    'seed',
    'objective',
    'epochs',
    'log_step',
    'display_step',
    'starter_learning_rate',
    'decay_learning_rate',
    'optimizer',
    'observations',
    'rank',
    'dispersion',
    'id',
    'init',
    'S0',
    'a0',
    'b0',
    'k0',
    'm0',
    'T0',
    'E0',
    'log_epochs',
    'log_learning_rate',
    'log_L',
    'log_L1',
    'log_L2'
]


# CONSTANTS FOR util
LOG_LIKELIHOOD = 'log_likelihood'
AIC = 'AIC'
AIC_C = 'AIC_c'
BIC = 'BIC'
REPRODUCIBILITY = 'reproducibility'
RECOGNITION = 'recognition'
SAMPLES = 'samples'
MUTATIONS = 'mutations'
DIMENSIONS = 'dimensions'
SILHOUETTE = 'silhouette'
DATAPOINTS = 'data_points'
DOF = 'dof'
# intorduced for contrib scripts
ROW = 'row'
COL = 'col'

NORMALIZE = 'norm'
COLLAPSE = 'collapse'

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

PARAMETERS = ['S', 'T', 'E', 'a0', 'b0', 'm1', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'L1', 'L2', 'L', 'tau', 'sub']
VARIABLES = ['S0', 'S0s', 'T0', 'E0', 'm0']

SNV_MUT_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
#OTHER_MUT_TYPES = pd.read_csv('/homes/harald/opt/tensigweb/othermut.csv', index_col=0)
SNV_MUT = ['       ' + n if m % 2 == 0 else n + '      ' for m, n in enumerate([i + k for j in list(
    map(lambda x: x.split('-')[0] + x.split('-')[1], SUB)) for i in NUC for k in NUC])]
SNV_MUT_ALT = [n for m, n in enumerate(
    [i + k for j in list(map(lambda x: x.split('-')[0] + x.split('-')[1], SUB)) for i in NUC for k in NUC])]
SNV_MUT_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
#OTHER_MUT_TYPES = pd.read_csv(OTHER_MUTATIONS, index_col=0)
COLORS = ["#2196F3", "#212121", "#f44336", "#BDBDBD", "#8BC34A", "#FFAB91"]


# Min and max values for simulations
BMIN, BMAX = -0.6931471805599453, 0.6931471805599453
AMIN, AMAX = -0.6931471805599453, 0.6931471805599453
KMIN, KMAX = -2, 2




DESCRIPTION = '''\
Run tensorflow NMF
------------------

Models
    squared_error : Run NMF using Squared Objective ||C-SE||^2
    divergence : Run NMF using Divergence Objective D(C||SE)
    negative_binomial : Run NMF using Negative Binomial Objective
    fixed_negative_binomial : Run NMF using Negative Binomial Objective
'''
