# -*- coding: utf-8 -*-

"""Console script for test."""
import sys
import click

@main.command()
def data():
	print('Create some sample data to run with tensor signatures.')

@main.command()
def train():
	print('Sub function to train a model')

@main.command()
def boot():
	print('Run bootstrapping with an single itertion of tensor signature output.')


@main.command()
def write():
	print('Summarizes pkl files.')


@click.group()
def main(args=None):
    """Console script for test."""
    click.echo("Replace this message by putting your code into "
               "test.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
