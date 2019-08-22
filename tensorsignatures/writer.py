#!/usr/bin/env python

from multiprocessing import Pool
from multiprocessing import cpu_count
import glob
import h5py as h5
import numpy as np
import os
import sys
import pickle
import re
from tensorsignatures.config import *
from tensorsignatures.util import load_dict, progress

def link_datasets(merged, path, verbose=True):
    
    def visitor_func(name, node, merged, path, verbose = True):
        if isinstance(node, h5.Dataset):
            if verbose:
                print('Linking', name)
            try:
                merged[name] = h5.ExternalLink(path, name)
            except RuntimeError:
                print('Warning could link {}'.format(name))
                


            splitted = name.split("/")
            data_path = "/".join(splitted[:-1])

            #print(data_path)

            #if data_path not in merged:
            #    merged.create_group(data_path)

            # merged[data_path].attrs[splitted[-1]] = str(name)

        if isinstance(node, h5.Group):
            if name not in merged:
                if verbose == True:
                    print('Adding attr to', name)
                merged.create_group(name)

            for k, v in node.attrs.items():
                if verbose:
                    print('Adding attr to', name, k, "->", v)
                merged[name].attrs[k] = v
    if verbose:
        print("Linking:", path)
    fh = h5.File(path)
    fh.visititems(lambda name, node: visitor_func(name, node, merged, path, verbose = verbose))
    fh.close()

def add_array(group, name, array, index):
    
    if name not in group:
        dset = group.create_dataset(
            name, (*array.shape, index + 1),
            maxshape=(*array.shape, None), compression="gzip")
        dset[..., index] = array
    else:
        dset = group[name]
        if index >= dset.shape[-1]:
            dset.resize((*dset.shape[:-1], index + 1))
        dset[..., index] = array


def save_h5f(fname, mode, data, verbose=False):
    with h5.File(fname, mode) as h5f:
        for i, (fname, params) in enumerate(data):
            

            progress(i, len(data), fname)
            #TODO: extract group name directly from params (?)

            regex = re.compile('([A-Z]*=\d+\.?\d*)')
            sub_group = '/'.join([sub for sub in regex.findall(fname.split('.pkl')[0]) if not sub.startswith('I')])
            group_name = fname.split('_')[0] + '/' + sub_group

            if group_name not in h5f:
                group = h5f.create_group(group_name)
            else:
                group = h5f[group_name]

            for k, v in params.items():
                if type(v) == np.ndarray:
                    add_array(group, k, v, params[ITERATION])
                elif (type(v) == int) or (type(v) == float) or (type(v) == str) or (type(v) == np.float32) or (type(v) == np.int64):
                    group.attrs[k] = v
                elif (type(v) == bool):
                    if v:
                        group.attrs[k] = 1
                    else:
                        group.attrs[k] = 0
                elif (None is None):
                    group.attrs[k] = 0
                else:
                    
                    print(type(v))
                    raise TypeError('unsupported type')

    return 0


def arg():
    import argparse
    description = """A description"""
    epilog = 'Designed by H.V.'
    # Initiate a ArgumentParser Class
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    # Call add_options to the parser
    parser.add_argument('input', help='Input dir')
    parser.add_argument('output', help='Output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='verbose mode')
    parser.add_argument('-b', type=int,
                        help='block_size',
                        default=-1)
    parser.add_argument('-c', type=int,
                        help='number of cores (default max)',
                        default=cpu_count())
    parser.add_argument('-r', action='store_true',
                        help='remove files after summarizing',
                        default=False)
    parser.add_argument('-l', action='store_true',
                        help='linking files',
                        default=False)


    return parser


def main():
    import sys
    parser = arg()
    args = parser.parse_args(sys.argv[1:])

    if args.l:
        print('Linking files ...')
        fh = h5.File(args.output)
        files = glob.glob(args.input)

        for p in files:
            link_datasets(fh, p)

        fh.close()
        sys.exit(0)

    files = glob.glob(args.input)
    len_files = len(files)
    

    if args.c > 1:
        pool = Pool(args.c)

    if args.b == -1:

        print('Processing {all} files.'.format(all=len_files))
        if args.c > 1:
            data = pool.map(load_dict, files)
        else:
            data = list(map(load_dict, files))

        mode = 'a' if os.path.exists(args.output) else 'w'
        save_h5f(args.output, mode, data, args.verbose)

    else:
        block_size = args.b
        current_block = 1

        for block_start in range(0, len(files), block_size):
            block_end = min(len(files), block_start + block_size)

            print('Processing Block {cb} ({bs}-{be})/{all}.'.format(
                cb=current_block,
                bs=block_start,
                be=block_end,
                all=len_files))

            block = files[block_start:block_end]

            if args.c > 1:
                block_data = pool.map(load_dict, block)
            else:
                block_data = list(map(load_dict, block))

            print("Writing Block {cb}.".format(cb=current_block))
            mode = 'a' if os.path.exists(args.output) else 'w'
            save_h5f(args.output, mode, block_data, args.verbose)
            current_block += 1

    if args.r:
        print('Cleaning up ...')
        for fname in files:
            os.remove(fname)
        error_files = glob.glob(os.path.join(args.input, '*.err'))
        for fname in error_files:
            os.remove(fname)
        report_files = glob.glob(os.path.join(args.input, '*.out'))
        for fname in report_files:
            os.remove(fname)


if __name__ == "__main__":
    main()
