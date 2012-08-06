import glob
import numpy as np
nax = np.newaxis
import os

import config
import experiments
import grammar
import observations
import recursive

NUM_ROWS = 200
NUM_COLS = 200
NUM_COMPONENTS = 10

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


ALL_MODELS = ['pmf', 'mog', 'ibp', 'chain', 'irm', 'bmf', 'kf', 'bctf', 'sparse', 'gsm']


def write_config_files(condition):
    # create dummy config files that point to default-config.txt
    for model in ALL_MODELS:
        config_file = './config/predictive/synthetic/%s/%s-config.txt' % (condition, model)
        outstr = open(config_file, 'w')
        print >> outstr, '#include default-config.txt'
        outstr.close()

def init_experiment():
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            name = 'synthetic/%s/%s' % (condition, model)
            params = experiments.read_config_file(experiments.config_file(name))
            print condition, model
            data, components = generate_data(model, NUM_ROWS, NUM_COLS, NUM_COMPONENTS, True)
            clean_data_matrix = observations.DataMatrix.from_real_values(data)
            noisy_data = np.random.normal(data, np.sqrt(params['noise-var']))
            data_matrix = observations.DataMatrix.from_real_values(noisy_data)
            experiments.init_experiment(name, data_matrix, components, clean_data_matrix=clean_data_matrix)
        
def init_level(level, override=False):
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            print condition, model
            experiments.init_level('synthetic/%s/%s' % (condition, model), level, override=override)

def collect_scores_for_level(level):
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            print condition, model
            experiments.collect_scores_for_level('synthetic/%s/%s' % (condition, model), level)



def write_jobs_init(level):
    if level == 1:
        raise RuntimeError('No need for initialization for level 1.')
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            name = 'synthetic/%s/%s' % (condition, model)
            winning_models = experiments.list_winning_models(name, level - 1)
            if winning_models[-1] == '---':
                continue
            jobs += experiments.list_init_jobs(name, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

def write_jobs_for_level(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            name = 'synthetic/%s/%s' % (condition, model)
            if level > 1:
                winning_models = experiments.list_winning_models(name, level - 1)
                if winning_models[-1] == '---':
                    continue
            jobs += experiments.list_jobs(name, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

def write_jobs_failed(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            jobs += experiments.list_jobs_failed('synthetic/%s/%s' % (condition, model), level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

def list_jobs_satisfying(level, state):
    status = {}
    fnames = ['/afs/csail/u/r/rgrosse/job_info/synthetic/status.txt']
    fnames += glob.glob('/afs/csail/u/r/rgrosse/job_info/synthetic/status-*.txt')
    assert fnames
    for fname in fnames:
        for line in open(fname):
            parts = line.strip().split()
            desc = parts[0][:-1]
            job = ' '.join(parts[1:])
            status[job] = desc
    return [k for k in status if status[k] == state]

## def list_jobs_dead(level):
##     jobs = set()
##     for condition in ['0.1', '1.0', '3.0', '10.0']:
##         for model in ALL_MODELS:
##             jobs.update(experiments.list_jobs('synthetic/%s/%s' % (condition, model), level))

##     jobs = jobs.difference(list_jobs_satisfying(level, 'finished'))
##     jobs = jobs.difference(list_jobs_satisfying(level, 'failed'))

##     return jobs

def list_jobs_dead(level):
    return list_jobs_satisfying(level, 'queued') + \
           list_jobs_satisfying(level, 'running')

def write_jobs_dead(level):
    jobs = list_jobs_dead(level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))


def temp_amazon_jobs():
    jobs = experiments.list_jobs('synthetic/0.1/pmf', 1)
    experiments.write_jobs(jobs, None)

def temp_amazon_jobs2():
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        for model in ALL_MODELS:
            if model == 'pmf' and condition == '0.1':
                continue
            jobs += experiments.list_jobs('synthetic/%s/%s' % (condition, model), 1)
    experiments.write_jobs(jobs, None)

def temp_redo_jobs():
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_jobs('synthetic/%s/bctf' % condition, 1)

    #outstr = open('/afs/csail/u/r/rgrosse/job_info/synthetic/jobs.txt', 'w')
    outstr = open(os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'), 'w')
    for j in jobs:
        print >> outstr, j
    outstr.close()

def temp_redo_init2():
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        experiments.init_level('synthetic/%s/bctf' % condition, 2, True)

def temp_redo_jobs2():
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_init_jobs('synthetic/%s/bctf' % condition, 2)

    #outstr = open('/afs/csail/u/r/rgrosse/job_info/synthetic/jobs.txt', 'w')
    outstr = open(os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'), 'w')
    for j in jobs:
        print >> outstr, j
    outstr.close()

def temp_redo_jobs3():
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_jobs('synthetic/%s/bctf' % condition, 2)

    #outstr = open('/afs/csail/u/r/rgrosse/job_info/synthetic/jobs.txt', 'w')
    outstr = open(os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'), 'w')
    for j in jobs:
        print >> outstr, j
    outstr.close()


def temp_list_winners(level):
    for model in ALL_MODELS:
        for condition in ['0.1', '1.0', '3.0', '10.0']:
            name = 'synthetic/%s/%s' % (condition, model)
            print name, experiments.list_winning_models(name, level)


def temp_average_running_time(structure):
    for model in ALL_MODELS:
        for condition in ['0.1', '1.0', '3.0', '10.0']:
            name = 'synthetic/%s/%s' % (condition, model)
            print name, experiments.average_running_time(name, 1, grammar.parse(structure))

def temp_redo_bctf_init_experiment():
    model = 'bctf'
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        data, components = generate_data(model, NUM_ROWS, NUM_COLS, NUM_COMPONENTS, True)
        data_matrix = recursive.Decomp(recursive.GaussianNode(data))
        experiments.init_experiment('synthetic/%s/%s' % (condition, model), data_matrix, components)

def temp_redo_bctf_init(level):
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        experiments.init_level('synthetic/%s/bctf' % condition, level, override=True)

def temp_redo_bctf_init_jobs(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_init_jobs('synthetic/%s/bctf' % condition, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

def temp_redo_bctf(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_jobs('synthetic/%s/bctf' % condition, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))


def temp_redo_gsm_init_experiment(override=False):
    model = 'gsm'
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        name = 'synthetic/%s/%s' % (condition, model)
        params = experiments.read_config_file(experiments.config_file(name))
        data, components = generate_data(model, NUM_ROWS, NUM_COLS, NUM_COMPONENTS, True)
        #data_matrix = recursive.Decomp(recursive.GaussianNode(data))
        clean_data_matrix = observations.DataMatrix.from_real_values(data)
        noisy_data = np.random.normal(data, np.sqrt(params['noise-var']))
        data_matrix = observations.DataMatrix.from_real_values(noisy_data)
        experiments.init_experiment('synthetic/%s/%s' % (condition, model), data_matrix, components,
                                    clean_data_matrix=clean_data_matrix, override=override)

def temp_redo_gsm_init(level):
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        experiments.init_level('synthetic/%s/gsm' % condition, level, override=True)
        
def temp_redo_gsm_init_jobs(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_init_jobs('synthetic/%s/gsm' % condition, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

def temp_redo_gsm_jobs(level):
    jobs = []
    for condition in ['0.1', '1.0', '3.0', '10.0']:
        jobs += experiments.list_jobs('synthetic/%s/gsm' % condition, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, 'synthetic/jobs.txt'))

    
