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
@click.argument(
    'input',
    )
@click.option('--mode', default = 'nbconst',
    help='What likelihood model shall be used to model count data'
    )
def train(input):
    """Deciphers tensorsignatures on a dataset.

    Args:\n
        input: hdf file containing the SNV count tensor and other mutation matrix.\n
    Returns:\n
        Saves a pkl file containing deciphered signatures and tensor factors.
    """
    print('Sub function to train a model')

@main.command()
def boot():
    print('Run bootstrapping with an single itertion of tensor signature output.')


@main.command()
@click.argument('input', metavar='GLOB', type=str)
@click.argument('output', metavar='FILE', type=str)

@click.option('--cores', '-c', type=int, default=1, 
    help='Number of cores (default=1).')
@click.option('--block_size', '-b', type=int, default=-1, 
    help='To prevent loading too many files to memory, this parameter \
    can be adjusted such that block_size files are written to the hdf \
    file before further pkl are loaded to memory (default = -1 meaning \
    that all files are loaded to memory before writing them to disk).')
@click.option('--remove', is_flag=True, 
    help='Removes all Tensorsignatures pkl files after they have been \
    written to the hdf file.')
@click.option('--link', is_flag=True, 
    help='Links several hdf files, which is sometimes useful for larege\
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
        if args.c > 1:
            data = pool.map(load_dict, files)
        else:
            data = list(map(load_dict, files))

        mode = 'a' if os.path.exists(args.output) else 'w'
        save_h5f(args.output, mode, data, args.verbose)

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

            if args.c > 1:
                block_data = pool.map(load_dict, block)
            else:
                block_data = list(map(load_dict, block))

            console.echo("Writing Block {}.".format(current_block))
            mode = 'a' if os.path.exists(args.output) else 'w'
            save_h5f(args.output, mode, block_data, args.verbose)
            current_block += 1    


    if remove:
        if config.verbose:
            click.echo('Cleaning up ...')
        
        for fname in files:
            os.remove(fname)
        


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
