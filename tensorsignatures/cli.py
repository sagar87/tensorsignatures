# -*- coding: utf-8 -*-

"""Console script for tensorsignatures."""
import sys
import click
import glob 
import os

from multiprocessing import Pool
from multiprocessing import cpu_count

from tensorsignatures.util import load_dict
from tensorsignatures.util import progress
from tensorsignatures.writer import link_datasets
from tensorsignatures.writer import save_h5f
from tensorsignatures.config import *

class Config(object):
    def __init__(self):
        self.verbose = False

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--verbose', is_flag=True)
@pass_config
def main(config, verbose):
    """This is TensorSignatures."""
    config.verbose = verbose
    return 0


@main.command()
@click.option('--seed', default = 0,
    help='Sets the seed for reproduceability.'
    )
def data():
    print('Create some sample data to run with tensor signatures.')

@main.command()
@click.argument(INPUT, type=str)
@click.argument(PREFIX, type=str)
@click.argument(RANK, type=int)


@click.option('--' + OBJECTIVE, '-o',
    metavar = '<str>',
    type = OBJECTIVE_CHOICE,
    default = 'nbconst',
    help='What likelihood model shall be used to model count data')
@click.option('--' + DISPERSION, '-k', 
    metavar='<float>', 
    type=int,
    default=50,
    help='dispersion factor (default = 50)')
@click.option('--' + ITERATION, '-i',  
    metavar='<int>', 
    type=int,
    default=0, 
    help='Iteration to (default = 0)')
@click.option('--' + NORMALIZE, '-n', 
    is_flag=True,
    help='multiply Chat1 with supplied normalisation constant N')
@click.option('--' + COLLAPSE, '-c',
    is_flag=True,
    help='collapse pyrimindine/purine dimension (SNV.shape[-2])')
@click.option('--' + EPOCHS, '-ep', 
    metavar='<int>', 
    type=int,
    default=10000, 
    help='number of epochs / training steps')
@click.option('--' + OPTIMIZER, '-op', 
    type=OPTIMIZER_CHOICE,
    default='ADAM',
    help='choose optimizer (default ADAM)')
@click.option('--' + DECAY_LEARNING_RATE, '-ld', 
    type=DECAY_LEARNING_RATE_CHOICE,
    default='exponential',
    help='learning rate decay (default exponential)')
@click.option('--' + STARTER_LEARNING_RATE, '-lr', 
    metavar='<float>', 
    type=float, 
    default=0.1, 
    help='starter learning rate (default = 0.1)')
@click.option('--' + DISPLAY_STEP, '-ds', 
    metavar='<int>', 
    type=int,
    default=100,
    help='progress updates / log step (default = 100)')
@click.option('--' + SUFFIX, '-su',
    metavar='<str>', 
    type=str,
    help='File suffix (default J_R_I)', 
    default='J_R_I')
@click.option('--' + SEED, 'se',
    metavar='<int>', 
    type=int, 
    default=None,
    help='initialize TensorSignatures variables with a seed')


@pass_config
def train(input, prefix, rank, dispersion, objective, iteration, 
    norm, collapse, epochs, optimizer, starter_learning_rate,
    decay_learning_rate, display_step, suffix, seed):
    """Deciphers tensorsignatures on a dataset.
    """
    click.echo('Ready to learn')
    

@main.command()
def boot():
    print('Run bootstrapping with an single itertion of tensor signature output.')


@main.command()
@click.argument('input', 
    metavar='GLOB', 
    type=str)
@click.argument('output', 
    metavar='FILE', 
    type=str)

@click.option('--cores', '-c', 
    type=int, 
    default=1, 
    help='Number of cores (default=1).')
@click.option('--block_size', '-b', 
    type=int, 
    default=-1, 
    help='To prevent loading too many files to memory, this parameter \
    can be adjusted such that block_size files are written to the hdf \
    file before further pkl are loaded to memory (default = -1 meaning \
    that all files are loaded to memory before writing them to disk).')
@click.option('--remove', 
    is_flag=True, 
    help='Removes all Tensorsignatures pkl files after they have been \
    written to the hdf file.')
@click.option('--link', 
    is_flag=True, 
    help='Links several hdf files, which is sometimes useful for large \
    experiments.')

@pass_config
def write(config, input, output, cores, block_size, remove, link):
    """Creates a hdf file out of tensor signatures pkls. Accepts a 
    glob argument (eg. "*.pkl"). Example: $tensorsignature write 
    "*.pkl" results.h5 
    """ 

    files = glob.glob(input)
    total_files = len(files)

    if config.verbose:
        click.echo('Found {} files.'.format(total_files))

    if cores > 1:
        cores = max([cpu_count(), cores])
        if config.verbose:
            click.echo('Using {} cores.'.format(cores))
        pool = Pool(cores)

    if block_size == -1:

        click.echo('Processing {} files.'.format(total_files))
        if cores > 1:
            data = pool.map(load_dict, files)
        else:
            data = [ load_dict(f) for f in files ] 

        mode = 'a' if os.path.exists(output) else 'w'
        save_h5f(output, mode, data, config.verbose)

    else:
        block_size = block_size
        current_block = 1

        
        for block_start in range(0, len(files), block_size):
            block_end = min(len(files), block_start + block_size)

            click.echo('Processing Block {cb} ({bs}-{be})/{all}.'.format(
                cb=current_block,
                bs=block_start,
                be=block_end,
                all=total_files))

            block = files[block_start:block_end]

            if cores > 1:
                block_data = pool.map(load_dict, block)
            else:
                block_data = [ load_dict(b) for b in block ]
                
            click.echo("Writing Block {}.".format(current_block))
            mode = 'a' if os.path.exists(output) else 'w'
            save_h5f(output, mode, block_data, config.verbose)
            current_block += 1    


    if remove:
        if config.verbose:
            click.echo('Cleaning up ...')
        
        for fname in files:
            os.remove(fname)
        


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
