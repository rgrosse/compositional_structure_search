import argparse
import collections
import numpy as np
nax = np.newaxis
import os
import StringIO
import sys

import config
import experiments
import observations
import presentation
from utils import misc, storage


NUM_ROWS = 200
NUM_COLS = 200
NUM_COMPONENTS = 10

DEFAULT_SEARCH_DEPTH = 3
DEFAULT_PREFIX = 'synthetic'

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


def experiment_name(prefix, noise_str, model):
    return '%s_%s_%s' % (prefix, noise_str, model)

def all_experiment_names(prefix):
    return [experiment_name(prefix, noise_str, model)
            for noise_str in NOISE_STR_VALUES
            for model in ALL_MODELS
            ]

def load_params(prefix):
    expt_name = all_experiment_names(prefix)[0]
    return storage.load(experiments.params_file(expt_name))

def initial_samples_jobs(prefix, level):
    return reduce(list.__add__, [experiments.initial_samples_jobs(name, level)
                                 for name in all_experiment_names(prefix)])

def initial_samples_key(prefix, level):
    return '%s_init_%d' % (prefix, level)

def evaluation_jobs(prefix, level):
    return reduce(list.__add__, [experiments.evaluation_jobs(name, level)
                                 for name in all_experiment_names(prefix)])

def evaluation_key(prefix, level):
    return '%s_eval_%d' % (prefix, level)

def final_model_jobs(prefix):
    return reduce(list.__add__, [experiments.final_model_jobs(name)
                                 for name in all_experiment_names(prefix)])

def final_model_key(prefix):
    return '%s_final' % prefix

def report_dir(prefix):
    return os.path.join(config.REPORT_PATH, prefix)

def report_file(prefix):
    return os.path.join(report_dir(prefix), 'results.txt')


def init_experiment(prefix, debug, search_depth=3):
    experiments.check_required_directories()
        
    for noise_str in NOISE_STR_VALUES:
        for model in ALL_MODELS:
            name = experiment_name(prefix, noise_str, model)
            if debug:
                params = experiments.QuickParams(search_depth=search_depth)
            else:
                params = experiments.SmallParams(search_depth=search_depth)
            data, components = generate_data(model, NUM_ROWS, NUM_COLS, NUM_COMPONENTS, True)
            clean_data_matrix = observations.DataMatrix.from_real_values(data)
            noise_var = float(noise_str)
            noisy_data = np.random.normal(data, np.sqrt(noise_var))
            data_matrix = observations.DataMatrix.from_real_values(noisy_data)
            experiments.init_experiment(name, data_matrix, params, components,
                                        clean_data_matrix=clean_data_matrix)
    
        
def init_level(prefix, level):
    for name in all_experiment_names(prefix):
        experiments.init_level(name, level)

def collect_scores_for_level(prefix, level):
    for name in all_experiment_names(prefix):
        experiments.collect_scores_for_level(name, level)

def run_everything(prefix, args):
    params = load_params(prefix)
    init_level(prefix, 1)
    experiments.run_jobs(evaluation_jobs(prefix, 1), args, evaluation_key(prefix, 1))
    collect_scores_for_level(prefix, 1)
    for level in range(2, params.search_depth + 1):
        init_level(prefix, level)
        experiments.run_jobs(initial_samples_jobs(prefix, level), args, initial_samples_key(prefix, level))
        experiments.run_jobs(evaluation_jobs(prefix, level), args, evaluation_key(prefix, level))
        collect_scores_for_level(prefix, level)
    experiments.run_jobs(final_model_jobs(prefix), args, final_model_key(prefix))


def print_failures(prefix, outfile=sys.stdout):
    params = load_params(prefix)
    failures = []
    for level in range(1, params.search_depth + 1):
        ok_counts = collections.defaultdict(int)
        fail_counts = collections.defaultdict(int)
        for expt_name in all_experiment_names(prefix):
            for _, structure in storage.load(experiments.structures_file(expt_name, level)):
                for split_id in range(params.num_splits):
                    for sample_id in range(params.num_samples):
                        ok = False
                        fname = experiments.scores_file(expt_name, level, structure, split_id, sample_id)
                        if os.path.exists(fname):
                            row_loglik, col_loglik = storage.load(fname)
                            if np.all(np.isfinite(row_loglik)) and np.all(np.isfinite(col_loglik)):
                                ok = True

                        if ok:
                            ok_counts[structure] += 1
                        else:
                            fail_counts[structure] += 1

        for structure in fail_counts:
            if ok_counts[structure] > 0:
                failures.append(presentation.Failure(structure, level, False))
            else:
                failures.append(presentation.Failure(structure, level, True))

    presentation.print_failed_structures(failures, outfile)

def print_learned_structures(prefix, outfile=sys.stdout):
    results = []
    for expt_name in all_experiment_names(prefix):
        structure = experiments.final_structure(expt_name)
        results.append(presentation.FinalResult(expt_name, structure))
    presentation.print_learned_structures(results, outfile)

def summarize_results(prefix, outfile=sys.stdout):
    print_learned_structures(prefix, outfile)
    print_failures(prefix, outfile)

def save_report(name, email=None):
    # write to stdout
    summarize_results(name)

    # write to report file
    if not os.path.exists(report_dir(name)):
        os.mkdir(report_dir(name))
    summarize_results(name, open(report_file(name), 'w'))

    if email is not None and email.find('@') != -1:
        header = 'experiment %s finished' % name
        buff = StringIO.StringIO()
        print >> buff, 'These results are best viewed in a monospace font.'
        print >> buff
        summarize_results(name, buff)
        body = buff.getvalue()
        buff.close()
        misc.send_email(header, body, email)



if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('command')

    if command == 'generate':
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--search_depth', type=int, default=DEFAULT_SEARCH_DEPTH)
        parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
        args = parser.parse_args()
        init_experiment(args.prefix, args.debug, args.search_depth)

    elif command == 'init':
        parser.add_argument('level', type=int)
        parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        init_level(args.prefix, args.level)
        if args.level > 1:
            experiments.run_jobs(initial_samples_jobs(args.prefix, args.level), args,
                                 initial_samples_key(args.prefix, args.level))

    elif command == 'eval':
        parser.add_argument('level', type=int)
        parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        experiments.run_jobs(evaluation_jobs(args.prefix, args.level), args,
                             evaluation_key(args.prefix, args.level))
        collect_scores_for_level(args.prefix, args.level)

    elif command == 'final':
        parser.add_argument('level', type=int)
        parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        experiments.run_jobs(final_model_jobs(args.prefix, args.level), args,
                             final_model_key(args.prefix))

    elif command == 'everything':
        parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
        experiments.add_scheduler_args(parser)
        args = parser.parse_args()
        run_everything(args.prefix, args)

    else:
        raise RuntimeError('Unknown command: %s' % command)
