# -*- coding: utf-8 -*-

"""Console script for tensorsignatures."""
import sys
import click
import glob
import os
import h5py as h5
from tqdm import trange

from multiprocessing import Pool
from multiprocessing import cpu_count

from tensorsignatures.util import load_dump
from tensorsignatures.util import prepare_data
from tensorsignatures.writer import save_hdf

from tensorsignatures.tensorsignatures import TensorSignature
from tensorsignatures.tensorsignatures import TensorSignatureRefit
from tensorsignatures.bootstrap import TensorSignatureBootstrap
from tensorsignatures.data import TensorSignatureData
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
@click.option('--seed',
              default=0,
              help='Sets the seed for reproduceability.')
@click.argument(SEED, type=int)
@click.argument(RANK, type=int)
@click.argument(OUTPUT, type=str)
@click.option('--' + INIT, '-i',
              metavar='<int>',
              type=int,
              default=0,
              help='realization of the data (default = 0)')
@click.option('--' + SAMPLES, '-s',
              metavar='<int>',
              type=int,
              default=100,
              help='number of genomes/samples (default = 100)')
@click.option('--' + MUTATIONS, '-m',
              metavar='<int>',
              type=int,
              default=1000,
              help='mutations per genome (default = 1000)')
@click.option('--' + DIMENSIONS, '-d',
              metavar='<int>',
              multiple=True,
              type=int,
              help='number of additional genomic dimensions',
              default=[2])
def data(seed, rank, output, init, samples, mutations, dimensions):

    synthetic = TensorSignatureData(
        seed=seed,
        rank=rank,
        samples=samples,
        mutations=mutations,
        dimensions=dimensions)

    synthetic.save_init(output, init=init)
    return 0


@main.command()
@click.argument(INPUT, type=str)
@click.argument(OUTPUT, type=str)
@click.argument(RANK, type=int)
@click.option('--' + OBJECTIVE, '-o',
              metavar='<str>',
              type=OBJECTIVE_CHOICE,
              default='nbconst',
              help='What likelihood model shall be used to model count data')
@click.option('--' + SIZE, '-k',
              metavar='<float>',
              type=int,
              default=50,
              help='dispersion factor (default = 50)')
@click.option('--' + INIT, '-i',
              metavar='<int>',
              type=int,
              default=0,
              help='Iteration to (default = 0)')
@click.option('--' + ID, '-j',
              metavar='<str>',
              type=str,
              default='tsTrain',
              help='job id (default = 0)')
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
@click.option('--' + LOG_STEP, '-ls',
              metavar='<int>',
              type=int,
              default=100,
              help='epoch inteval to make logging steps (default = 100)')
@click.option('--' + SEED, '-se',
              metavar='<int>',
              type=int,
              default=None,
              help='initialize TensorSignatures variables with a seed')
@pass_config
def train(config, input, output, rank, objective, size, init, id,
          norm, collapse, epochs, optimizer, decay_learning_rate,
          starter_learning_rate, display_step, log_step, seed):
    """Deciphers tensorsignatures on a dataset.
    """
    snv = h5.File(input, 'r')['SNV'][()]
    other = h5.File(input, 'r')['OTHER'][()]
    N = None
    if norm:
        N = h5.File(input, 'r')['N'][()]

    model = TensorSignature(
        snv=snv, other=other, rank=rank, N=N, size=size, objective=objective,
        collapse=collapse, starter_learning_rate=starter_learning_rate,
        optimizer=optimizer, epochs=epochs, log_step=log_step,
        display_step=display_step, id=id, init=init, seed=seed,
        verbose=config.verbose)

    results = model.fit()
    results.dump(output)


@main.command()
@click.argument(INPUT, type=str)
@click.argument(OUTPUT, type=str)
@click.option('--' + 'reference', '-r',
              metavar='<str>',
              default='PCAWG',
              help='TensorSignature initialization (*.pkl, default = PCAWG)')
@click.option('--' + INIT, '-i',
              metavar='<int>',
              type=int,
              default=0,
              help='Iteration to (default = 0)')
@click.option('--' + ID, '-j',
              metavar='<str>',
              type=str,
              default='tsTrain',
              help='job id (default = 0)')
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
@click.option('--' + STARTER_LEARNING_RATE, '-lr',
              metavar='<float>',
              type=float,
              default=0.1,
              help='starter learning rate (default = 0.1)')
@click.option('--' + DECAY_LEARNING_RATE, '-ld',
              type=DECAY_LEARNING_RATE_CHOICE,
              default='exponential',
              help='learning rate decay (default exponential)')
@click.option('--' + DISPLAY_STEP, '-ds',
              metavar='<int>',
              type=int,
              default=100,
              help='progress updates / log step (default = 100)')
@click.option('--' + LOG_STEP, '-ls',
              metavar='<int>',
              type=int,
              default=100,
              help='epoch inteval to make logging steps (default = 100)')
@click.option('--' + SEED, '-se',
              metavar='<int>',
              type=int,
              default=None,
              help='initialize TensorSignatures variables with a seed')
@pass_config
def refit(config, input, output, reference, init, id, norm, collapse, epochs,
          optimizer, starter_learning_rate, decay_learning_rate, display_step,
          log_step, seed):
    """Refits a set of signatures to set a new dataset.
    """

    with h5.File(input, 'r') as fh:
        snv = fh['SNV'][()]
        other = fh['OTHER'][()]
        N = None
        if norm:
            N = fh['N'][()]

    if reference == 'PCAWG':
        ref = load_dump(PCAWG)
    else:
        ref = load_dump(reference)

    model = TensorSignatureRefit(
        snv=snv, other=other, reference=ref, N=N, collapse=collapse,
        starter_learning_rate=starter_learning_rate,
        optimizer=optimizer, epochs=epochs, log_step=log_step,
        display_step=display_step, id=id, init=init, seed=seed,
        verbose=config.verbose)

    results = model.fit()
    results.dump(output)


@main.command()
@click.argument(INPUT, type=str)
@click.argument('dump', type=str)
@click.argument('max_init', type=int)
@click.option('--' + ID, '-j',
              metavar='<str>',
              type=str,
              default='tsTrain',
              help='job id (default = 0)')
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
@click.option('--' + STARTER_LEARNING_RATE, '-lr',
              metavar='<float>',
              type=float,
              default=0.1,
              help='starter learning rate (default = 0.1)')
@click.option('--' + DECAY_LEARNING_RATE, '-ld',
              type=DECAY_LEARNING_RATE_CHOICE,
              default='exponential',
              help='learning rate decay (default exponential)')
@click.option('--' + DISPLAY_STEP, '-ds',
              metavar='<int>',
              type=int,
              default=100,
              help='progress updates / log step (default = 100)')
@click.option('--' + LOG_STEP, '-ls',
              metavar='<int>',
              type=int,
              default=100,
              help='epoch inteval to make logging steps (default = 100)')
@click.option('--' + SEED, '-se',
              metavar='<int>',
              type=int,
              default=None,
              help='initialize TensorSignatures variables with a seed')
@pass_config
def boot(config, input, dump, max_init, id, norm, collapse, epochs, optimizer,
         starter_learning_rate, decay_learning_rate, display_step,
         log_step, seed):
    snv = h5.File(input, 'r')['SNV'][()]
    other = h5.File(input, 'r')['OTHER'][()]
    N = None
    if norm:
        N = h5.File(input, 'r')['N'][()]

    initialization = load_dump(dump)

    for i in range(max_init):
        model = TensorSignatureBootstrap(
            snv, other, initialization, N, collapse, epochs,
            starter_learning_rate, decay_learning_rate, optimizer, log_step,
            display_step, id, i, seed)

        result = model.fit()
        result.dump(id + '_I=' + str(i) + '.plk')


@main.command()
@click.argument(INPUT,
                type=str)
@click.argument(OUTPUT,
                type=str)
@click.option('--cores', '-c',
              type=int,
              default=1,
              help='Number of cores (default=1).')
@click.option('--block_size', '-b',
              type=int,
              default=-1,
              help='Saves files after block_size files.')
@click.option('--remove',
              is_flag=True,
              help='Removes pkl files after writing them to the hdf file.')
@click.option('--link',
              is_flag=True,
              help='Links several hdf files (useful for large experiments)')
@pass_config
def write(config, input, output, cores, block_size, remove, link):
    """Creates a hdf file out of dumped tensor signatures pkls.

    Args:

        input (:obj:`str`): A GLOB argument (eg. "test_project*.pkl").
        output (:obj:`str`): Output hdf file.

    Example:

        $ tensorsignature write "*.pkl" results.h5
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
            data = pool.map(load_dump, files)
        else:
            data = []
            t = trange(total_files, desc='Progress', leave=True)
            for i in t:
                data.append(load_dump(files[i]))
                t.set_description('Loading: {}'.format(files[i]))
                t.refresh()
            # data = [load_dump(f) for f in files]

        mode = 'a' if os.path.exists(output) else 'w'
        save_hdf(data, output, mode, config.verbose)

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
                block_data = pool.map(load_dump, block)
            else:
                block_data = [load_dump(b) for b in block]

            click.echo("Writing Block {}.".format(current_block))
            mode = 'a' if os.path.exists(output) else 'w'
            save_hdf(block_data, output, mode, config.verbose)
            current_block += 1

    if remove:
        if config.verbose:
            click.echo('Cleaning up ...')

        for fname in files:
            os.remove(fname)


@main.command()
@click.argument(INPUT, type=str)
@click.argument(OUTPUT, type=str)
@pass_config
def prep(config, input, output):
    """Adds a normalization constant and formats tensors appropriately."""
    prepare_data(input, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
