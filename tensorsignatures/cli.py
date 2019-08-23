# -*- coding: utf-8 -*-

"""Console script for test."""
import sys
import click

@click.group()
def main(args=None):
    """Console script for test."""
    click.echo("Replace this message by putting your code into "
               "test.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


@main.command()
@click.option('--seed', default = 0,
    help='Sets the seed for reproduceability.'
    )
def data():
    print('Create some sample data to run with tensor signatures.')

@main.command()
@click.argument('input', required=True,
    help='Input hdf file which contains the count tensor and other mutation types.'
    )
@click.option('--mode', default = 'nbconst',
    help='What likelihood model shall be used to model count data'
    )
def train(input):
    print('Sub function to train a model')

@main.command()
def boot():
    print('Run bootstrapping with an single itertion of tensor signature output.')


@main.command()
def write():
    print('Summarizes pkl files.')





if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
