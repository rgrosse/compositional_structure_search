import argparse
import numpy as np
nax = np.newaxis
import os
import sys

import config
import experiments
import observations


NUM_ROWS = 200
NUM_COLS = 200
NUM_COMPONENTS = 10

DEFAULT_SEARCH_DEPTH = 3

def generate_ar(nrows, ncols, a):
    X = np.zeros((nrows, ncols))
    X[0,:] = np.random.normal(size=ncols)
    for i in range(1, nrows):
        X[i,:] = a * X[i-1,:] + np.random.normal(0., np.sqrt(1-a**2), size=ncols)
    return X

def generate_data(data_str, nrows, ncols, ncomp, return_components=False):
    IBP_ALPHA = 2.
    pi_crp = np.ones(ncomp) / ncomp
    pi_ibp = np.ones(ncomp) * IBP_ALPHA / ncomp

    if data_str[-1] == 'T':
        data_str = data_str[:-1]
        transpose = True
        nrows, ncols = ncols, nrows
    else:
        transpose = False
    
    if data_str == 'pmf':
        U = np.random.normal(0., 1., size=(nrows, ncomp))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)
        
    elif data_str == 'mog':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'ibp':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(nrows, ncomp))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'sparse':
        Z = np.random.normal(0., 1., size=(nrows, ncomp))
        U = np.random.normal(0., np.exp(Z))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)
        

    elif data_str == 'gsm':
        U_inner = np.random.normal(0., 1., size=(nrows, 1))
        V_inner = np.random.normal(0., 1., size=(1, ncomp))
        Z = np.random.normal(U_inner * V_inner, 1.)
        #Z = 2. * Z / np.sqrt(np.mean(Z**2))

        U = np.random.normal(0., np.exp(Z))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'irm':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.multinomial(1, pi_crp, size=ncols).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'bmf':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(nrows, ncomp))
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.binomial(1, pi_ibp[nax,:], size=(ncols, ncomp)).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'mgb':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.binomial(1, pi_ibp[nax,:], size=(ncols, ncomp)).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'chain':
        data = generate_ar(nrows, ncols, 0.9)
        components = (data)

    elif data_str == 'kf':
        U = generate_ar(nrows, ncomp, 0.9)
        V = np.random.normal(size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'bctf':
        temp1, (U1, V1) = generate_data('mog', nrows, ncols, ncomp, True)
        F1 = np.random.normal(temp1, 1.)
        temp2, (U2, V2) = generate_data('mog', nrows, ncols, ncomp, True)
        F2 = np.random.normal(temp2, 1.)
        data = np.dot(F1, F2.T)
        components = (U1, V1, F1, U2, V2, F2)
        

    data /= np.std(data)

    if transpose:
        data = data.T

    if return_components:
        return data, components
    else:
        return data


NOISE_STR_VALUES = ['0.1', '1.0', '3.0', '10.0']
ALL_MODELS = ['pmf', 'mog', 'ibp', 'chain', 'irm', 'bmf', 'kf', 'bctf', 'sparse', 'gsm']


def experiment_name(noise_str, model):
    return 'synthetic_%s_%s' % (noise_str, model)

def all_experiment_names():
    return [experiment_name(noise_str, model)
            for noise_str in NOISE_STR_VALUES
            for model in ALL_MODELS
            ]

def initial_samples_jobs(level):
    return reduce(list.__add__, [experiments.initial_samples_jobs(name, level)
                                 for name in all_experiment_names()])

def initial_samples_key(level):
    return 'synthetic_init_%d' % level

def evaluation_jobs(level):
    return reduce(list.__add__, [experiments.evaluation_jobs(name, level)
                                 for name in all_experiment_names()])

def evaluation_key(level):
    return 'synthetic_eval_%d' % level

def final_model_jobs(level):
    return reduce(list.__add__, [experiments.final_model_jobs(name, level)
                                 for name in all_experiment_names()])

def final_model_key():
    return 'synthetic_final'


def init_experiment(debug, search_depth=3):
    for noise_str in NOISE_STR_VALUES:
        for model in ALL_MODELS:
            name = experiment_name(noise_str, model)
            if debug:
                params = experiments.DebugParams(search_depth=search_depth)
            else:
                params = experiments.SmallParams(search_depth=search_depth)
            data, components = generate_data(model, NUM_ROWS, NUM_COLS, NUM_COMPONENTS, True)
            clean_data_matrix = observations.DataMatrix.from_real_values(data)
            noise_var = float(noise_str)
            noisy_data = np.random.normal(data, np.sqrt(noise_var))
            data_matrix = observations.DataMatrix.from_real_values(noisy_data)
            experiments.init_experiment(name, data_matrix, params, components,
                                        clean_data_matrix=clean_data_matrix)
        
def init_level(level):
    for name in all_experiment_names():
        experiments.init_level(name, level)

def collect_scores_for_level(level):
    for name in all_experiment_names():
        experiments.collect_scores_for_level(name, level)

def run_everything(name, search_depth, args):
    init_level(name, 1)
    experiments.run_jobs(evaluation_jobs(name, 1), args, evaluation_key(name, 1))
    for level in range(2, search_depth + 1):
        init_level(name, level)
        experiments.run_jobs(initial_samples_jobs(name, level), args, initial_samples_key(name, level))
        experiments.run_jobs(evaluation_jobs(name, level), args, evaluation_key(name, level))
        collect_scores_for_level(name, level)
    experiments.run_jobs(final_model_jobs(name, level), args, final_model_key(name))




if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('command')

    if command == 'generate':
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--search_depth', type=int, default=DEFAULT_SEARCH_DEPTH)
        args = parser.parse_args()
        init_experiment(args.debug, args.search_depth)

    elif command == 'init':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        init_level(args.name, args.level)
        if args.level > 1:
            experiments.run_jobs(initial_samples_jobs(args.name, args.level), args,
                                 initial_samples_key(args.name, args.level))

    elif command == 'eval':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        experiments.run_jobs(evaluation_jobs(args.name, args.level), args,
                             evaluation_key(args.name, args.level))
        collect_scores_for_level(args.name, args.level)

    elif command == 'final':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        experiments.run_jobs(final_model_jobs(args.name, args.level), args,
                             final_model_key(args.name))

    elif command == 'everything':
        parser.add_argument('name', type=str)
        parser.add_argument('--search_depth', type=int, default=DEFAULT_SEARCH_DEPTH)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        run_everything(args.name, args.search_depth, args)

    else:
        raise RuntimeError('Unknown command: %s' % command)
