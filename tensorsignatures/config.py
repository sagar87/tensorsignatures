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

COUNTS = 'counts'
SIGNATURE = 'signature'
EXPOSURE = 'exposure'
SEED='seed'

INPUT_ROWS = 'rows'
INPUT_COLS = 'cols'
RANK = 'rank'

# parameters saved during training
LOSS = 'loss'
OBJECTIVE = 'objective'
CONSTRAINT = 'constraint'
CONSTRAINT_CHOICE = [ 'l1', 'l2', 'sigmoid', 'relu', -1]
LEARNING_RATE = 'learning_rate'
DISPERSION = 'dispersion'
LAMBDA = 'lambda'

PREFIX = 'prefix'
SUFFIX = 'suffix'


STARTER_KEEP_PROB = 'starter_keep_prob'
DROPOUT = 'dropout'
KEEP_PROB = 'keep_probability'

FULL_TRAIN = 'full_train'  # full train proportion

LOG_CONSTRAINT = 'log_constraint'
LOG_LEARNING_RATE = 'log_learning_rate'
LOG_OBJECTIVE = 'log_objective'
LOG_KEEP_PROB = 'log_keep_prob'
LOG_SIZE = 'log_size'
LOG_L1 = 'log_L1'
LOG_L2 = 'log_L2'
LOG_L = 'log_L'
LOG_LAMBDA_T = 'log_lambda_t' # lambda for transcription bias
LOG_LAMBDA_R = 'log_lambda_r' # lambda for replication bias
LOG_LAMBDA_C = 'log_lambda_c' # lambda for 
LOG_LAMBDA_A = 'log_lambda_a' 


VERBOSE = 'verbose'
INPUT = 'input'
OUTPUT = 'output'
LOGS = 'logs'
OBJECTIVE = 'objective'
OBJECTIVE_CHOICE = click.Choice(['nbconst', 'poisson'])
OPTIMIZER = 'optimizer'
OPTIMIZER_CHOICE = click.Choice(['ADAM', 'gradient_descent'])
FILENAME = 'file_name'

EPOCHS = 'epochs'
STARTER_LEARNING_RATE = 'starter_learning_rate'
DECAY_LEARNING_RATE = 'decay_learning_rate'
DECAY_LEARNING_RATE_CHOICE = click.Choice(['exponential', 'constant'])

PARARMS = 'params'
JOB_NAME = 'job_name'
ITERATION = 'iteration'
DISPLAY_STEP = 'display_step'


# CONSTANTS FOR util
LOG_LIKELIHOOD = 'log_likelihood'
AIC = 'AIC'
AIC_C = 'AIC_c'
BIC = 'BIC'
REPRODUCIBILITY = 'reproducibility'
RECOGNITION = 'recognition'
SAMPLES = 'samples'
SILHOUETTE = 'silhouette'
DATAPOINTS = 'data_points'
DOF = 'dof'
# intorduced for contrib scripts
ROW = 'row'
COL = 'col'


LAMBDA_T = 'lambda_t' # lambda for transcription bias
LAMBDA_R = 'lambda_r' # lambda for replication bias
LAMBDA_C = 'lambda_c' # lambda for 
LAMBDA_A = 'lambda_a' 

NORMALIZE = 'norm'
COLLAPSE = 'collapse'


ORI = ['+', '-', '*']
NUC = ['A', 'C', 'G', 'T']
PYR = ['C'] *3 +['T']*3
SUB = ['[C->A]', '[C->G]', '[C->T]', '[T->A]', '[T->C]', '[T->G]']

CHROMSTATES = {
    0:"NA",
    1:"active TSS",
    2:"flanking active TSS",
    3:"transcr. at gene 5'and 3'",
    4:"strong transcription",
    5:"weak transcription",
    6:"genic enhancers",
    7:"enhancers",
    8:"ZNF genes + repeats",
    9:"heterochromatin",
    10:"bivalent/poised TSS",
    11:"flanking bivalent TSS/Enh",
    12:"bivalent enhancer",
    13:"repressed polycomb",
    14:"weak repressed polycomb",
    15:"quiescent/low"
}

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





DESCRIPTION = '''\
Run tensorflow NMF
------------------

Models
    squared_error : Run NMF using Squared Objective ||C-SE||^2
    divergence : Run NMF using Divergence Objective D(C||SE)
    negative_binomial : Run NMF using Negative Binomial Objective
    fixed_negative_binomial : Run NMF using Negative Binomial Objective
'''
