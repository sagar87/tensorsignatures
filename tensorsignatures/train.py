#!/usr/bin/env python

import sys
import os
import warnings; warnings.filterwarnings("ignore")
import argparse
import textwrap
import datetime
import tensorflow as tf
import numpy as np
import h5py as h5
from tensorsignatures.config import *
from tensorsignatures.tensorsignatures import *
from tensorsignatures.bootstrap import *
from tensorsignatures.util import *

def train_reloaded(model, params):
    
    
    logs = {
    LOG_LEARNING_RATE : np.zeros(params[EPOCHS] // params[DISPLAY_STEP]),
    LOG_SIZE : np.zeros(params[EPOCHS] // params[DISPLAY_STEP]),
    LOG_L : np.zeros(params[EPOCHS] // params[DISPLAY_STEP]),
    LOG_L1 : np.zeros(params[EPOCHS] // params[DISPLAY_STEP]),
    LOG_L2 : np.zeros(params[EPOCHS] // params[DISPLAY_STEP]),
    }

    #conf = tf.ConfigProto()
    #conf.gpu_options.allow_growth = True

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #if model.snv is None:
    for i in range(params[EPOCHS]):
        
        _ = sess.run(model.minimize)
        # update logs
        
        if (i%params[DISPLAY_STEP] == 0) and params[VERBOSE]:
            log_i = i // params[DISPLAY_STEP] 
            logs[LOG_LEARNING_RATE][log_i] = sess.run(model.learning_rate)
            logs[LOG_SIZE][log_i] = sess.run(model.tau) 
            logs[LOG_L][log_i] = sess.run(model.L)
            logs[LOG_L1][log_i] = sess.run(model.L1)
            logs[LOG_L2][log_i] = sess.run(model.L2)
        
            log_string = 'Likelihood {lh:.2f} delta {delta:.2f} snv {snv:.2f} other {other:.2f} size {size:.2f} lr {lr:.4f}'.format(
                lh = logs[LOG_L][log_i],
                snv = logs[LOG_L1][log_i],
                other = logs[LOG_L2][log_i],
                lr = logs[LOG_LEARNING_RATE][log_i],
                delta= logs[LOG_L][log_i] - logs[LOG_L][log_i-1],
                size = logs[LOG_SIZE][log_i]
                )
        
            #print(log_string)
            progress(i, params[EPOCHS], log_string)

    print ('Finished Training')
    data = {**params, **model.get_tensors(sess), **logs}

    sess.close()

    return data

def arg():
    # Initiate a ArgumentParser Class
    job_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    parser = argparse.ArgumentParser(
        prog="train.py", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(DESCRIPTION),
        add_help=False)

    base_parser = argparse.ArgumentParser(add_help=False)
    #base_parser.add_argument('-n', '--name', action='store', required=True,
    #    help="The name of the person running the program")
    base_parser.add_argument('-v', '--' + VERBOSE,
        action='store_true',
        help='verbose mode')
    base_parser.add_argument('-j',  '--' + JOB_NAME,
        metavar='STRING',
        type=str,
        help='job name',
        default=job_name)
    base_parser.add_argument('-i', '--' + ITERATION,
        metavar='INT', 
        type=int,
        nargs='+',
        help='iteration (default = [0])', 
        default=[0])
    base_parser.add_argument('-s', '--' + SEED,
        metavar='INT', 
        type=int,
        help='initialize TensorSignatures variables with a seed', 
        default=None)
    base_parser.add_argument('-fn', '--' + FILENAME,
        metavar='STRING', 
        type=str,
        help='enter string to save the job (default J_R_K_I)', 
        default='J_R_K_I')
    base_parser.add_argument('-ds', '--' + DISPLAY_STEP,
        metavar='INT', 
        type=int,
        help='progress updates / log step (default = 100)', 
        default=100)
    base_parser.add_argument('-n', '--' + NORMALIZE,
        action='store_true',
        help='multiply Chat1 with supplied normalisation constant N')
    base_parser.add_argument('-c', '--' + COLLAPSE,
        action='store_false',
        help='collapse pyrimindine/purine dimension (SNV.shape[-2])')

    init = base_parser.add_argument_group('initialization parameters')
    init.add_argument('-op', '--' + OPTIMIZER,
        metavar='STRING', 
        type=str,
        default='ADAM',
        help='choose optimizer (default ADAM)',
        choices=OPTIMIZER_CHOICE)
    init.add_argument('-ep', '--' + EPOCHS,
        metavar='INT', 
        type=int,
        default=10000, 
        help='number of epochs / training steps')
    init.add_argument('-lr', '--' + STARTER_LEARNING_RATE,
        metavar='FLOAT', 
        type=float,
        default=0.1, 
        help='starter learning rate (default = 0.1)')
    init.add_argument('-ld', '--' + DECAY_LEARNING_RATE,
        metavar='STRING', 
        type=str,
        default='exponential',
        help='learning rate decay (default exponential)',
        choices=DECAY_LEARNING_RATE_CHOICE)
    
    subparsers = parser.add_subparsers(dest='mode', help='choose script action')
    
    train = subparsers.add_parser('train', parents=[base_parser])
    train.add_argument(INPUT,
        metavar='STRING',
        type=str,
        help='input data (hdf5 file with datasets SNV, OTHER and (optionally) N)')
    train.add_argument(OUTPUT,
        metavar='DIR',
        type=str,
        help='output dir (writes ckpt and log folder to it)')
    train.add_argument(OBJECTIVE,
        metavar='STR',
        type=str,
        help='choose model',
        choices=OBJECTIVE_CHOICE,
        )
    train.add_argument(RANK,
        metavar='INT',
        type=int,
        help='rank of the decomposition (int)')
    
    params = train.add_argument_group('model parameters')
    params.add_argument('-k', '--' + DISPERSION,
        metavar='FLOAT', 
        type=int,
        help='dispersion factor (default = 50)',
        default=50)
    # params.add_argument('-la', '--' + LAMBDA_A,
    #     metavar='INT',
    #     type=int,
    #     help='sigma of gaussian on signature activities (sigma = sqrt(1/lambda))',
    #     default=0)
    # params.add_argument('-lts', '--' + LAMBDA_T,
    #     metavar='INT',
    #     type=int,
    #     help='sigma of gaussian prior on transcription',
    #     default=0)
    # params.add_argument('-lrt', '--' + LAMBDA_R,
    #     metavar='INT',
    #     type=int,
    #     help='sigma of gaussian prior on transcription',
    #     default=0)
    # params.add_argument('-lc', '--' + LAMBDA_C,
    #     metavar='INT',
    #     type=int,
    #     help='sigma of gaussian prior clustering coefficient',
    #     default=0)

    bootstrap = subparsers.add_parser('bootstrap', parents=[base_parser])
    bootstrap.add_argument(INPUT,
        metavar='STRING',
        type=str,
        help='input data (hdf5 file with datasets SNV, OTHER and (optionally) N)')
    bootstrap.add_argument(SEED,
        metavar='STRING',
        type=str,
        help='seed pkl file')
    bootstrap.add_argument(OUTPUT,
        metavar='DIR',
        type=str,
        help='output dir (writes ckpt and log folder to it)')
    bootstrap.add_argument('-type',
        metavar='string', 
        type=str,
        help='which bootstrap type', 
        default='randomize')
    bootstrap.add_argument('-sub',
        metavar='FLOAT', 
        type=float,
        help='proportion of samples that are drawn to create the bootstrap sample (default = 2/3)', 
        default=2/3)
    bootstrap.add_argument('-frac',
        metavar='FLOAT', 
        type=float,
        help='proportion of values that are disorted (default = 0.1)', 
        default=.1)

    
    refit = subparsers.add_parser('refit', parents=[base_parser])
    refit.add_argument(INPUT,
        metavar='STRING',
        type=str,
        help='input data (hdf5 file with datasets SNV, OTHER and (optionally) N)')
    refit.add_argument(SEED,
        metavar='STRING',
        type=str,
        help='seed pkl file')
    refit.add_argument(OUTPUT,
        metavar='DIR',
        type=str,
        help='output dir (writes ckpt and log folder to it)')


    try:
        return parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))


def main():
    import sys
    args = arg()
    params = vars(args)

    # Get arguments on main program
    if args.mode == 'train':
        with h5.File(params[INPUT]) as fh:
            snv = fh['SNV'][()]
            other = fh['OTHER'][()]
            
            if params['norm']:
                N = fh['N'][()]
            else:
                N = None
        
        # append necessary information to params dic
        params['data_pts'] = np.array(np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other)))
        
        if len(params[ITERATION]) == 1:
            params[ITERATION] = params[ITERATION][0]
            model = TensorSignature(snv=snv, other=other, N=N, **params)
            results = train_reloaded(model, params)
            fname = create_file_name(params)
            save_dict(results, os.path.join(params[OUTPUT], fname+'.pkl'))
        
        elif len(params[ITERATION]) == 2:
            for i in np.arange(params[ITERATION][0], params[ITERATION][1]):
                if params[VERBOSE]:
                    print('Iteration {}/{}'.format(i, params[ITERATION][1]))
                params_copy = dict(params)
                params_copy[ITERATION] = i

                tf.reset_default_graph()

                model = TensorSignature(snv=snv, other=other, N=N, **params)
                results = train_reloaded(model, params_copy)
                fname = create_file_name(params_copy)
                save_dict(results, os.path.join(params_copy[OUTPUT], fname+'.pkl'))

        sys.exit(0)
    elif args.mode == 'refit':
        seed = Singleton(params[SEED])
        
        with h5.File(params[INPUT]) as fh:
            snv = fh['SNV'][()]
            other = fh['OTHER'][()]
            
            if params['norm']:
                N = fh['N'][()]
            else:
                N = None

        params['data_pts'] = np.array(np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other)))
        params[RANK] = seed.rank
        params[DISPERSION] = seed[DISPERSION]

        if len(params[ITERATION]) == 1:
            params[ITERATION] = params[ITERATION][0]
            model = TensorSignatureRefit(snv=snv, other=other, N=N, clu=seed, **params)
            results = train_reloaded(model, params)
            fname = create_file_name(params)
            save_dict(results, os.path.join(params[OUTPUT], fname+'.pkl'))        
        
        elif len(params[ITERATION]) == 2:
            for i in np.arange(params[ITERATION][0], params[ITERATION][1]):
                if params[VERBOSE]:
                    print('Iteration {}/{}'.format(i, params[ITERATION][1]))
                params_copy = dict(params)
                params_copy[ITERATION] = i

                tf.reset_default_graph()

                model = TensorSignatureRefit(snv=snv, other=other, N=N, clu=seed, **params)
                results = train_reloaded(model, params_copy)
                fname = create_file_name(params_copy)
                save_dict(results, os.path.join(params_copy[OUTPUT], fname+'.pkl'))

        print(seed)
        print('refitting seed to new model')
        sys.exit(0)
    elif args.mode == 'bootstrap':
        seed = Singleton(params[SEED])
        
        with h5.File(params[INPUT]) as fh:
            snv = fh['SNV'][()]
            other = fh['OTHER'][()]
            
            if params['norm']:
                N = fh['N'][()]
            else:
                N = None

        params['data_pts'] = np.array(np.sum(~np.isnan(snv)) + np.sum(~np.isnan(other)))        
        params[RANK] = seed.rank
        params[DISPERSION] = seed[DISPERSION]        


        if len(params[ITERATION]) == 1:
            params[ITERATION] = params[ITERATION][0]

            if args.type == 'randomize':
                model = TensorSignatureRandomize(snv=snv, other=other, N=N, clu=seed, **params)
            elif args.type == 'bootT':
                model = TensorSignatureBootT(snv=snv, other=other, N=N, clu=seed, **params)

            results = train_reloaded(model, params)
            fname = create_file_name(params)
            save_dict(results, os.path.join(params[OUTPUT], fname+'.pkl'))         
        
        elif len(params[ITERATION]) == 2:
            for i in np.arange(params[ITERATION][0], params[ITERATION][1]):
                if params[VERBOSE]:
                    print('Iteration {}/{}'.format(i, params[ITERATION][1]))
                params_copy = dict(params)
                params_copy[ITERATION] = i

                tf.reset_default_graph()
                if args.type == 'randomize':
                    model = TensorSignatureRandomize(snv=snv, other=other, N=N, clu=seed, **params)
                elif args.type == 'bootT':
                    model = TensorSignatureBootT(snv=snv, other=other, N=N, clu=seed, **params)
                
                results = train_reloaded(model, params_copy)
                fname = create_file_name(params_copy)
                save_dict(results, os.path.join(params_copy[OUTPUT], fname+'.pkl'))
        

        print('bootstrapping seed')
        sys.exit(0)


if __name__ == "__main__":
    main()