# -*- coding: utf-8 -*-

"""Console script for test."""
import sys
import click
import glob 



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
@pass_config
def write(config, input, output):
    """Creates a hdf file out of tensor signatures pkls. 
    Accepts a glob argument (eg. "*.pkl"). Example: 
    $tensorsignature write "*.pkl" results.h5 
    """ 

    files = glob.glob(input)
    len_files = len(files)

    if config.verbose:
        click.echo('Found {} files.'.format(len_files))

    print('Summarizes pkl files.')





if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
