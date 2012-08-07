import cPickle
import hashlib
import numpy as np
nax = np.newaxis
import os
import termcolor
import time
import traceback

import algorithms
import grammar
import observations
import recursive
import scoring

NROWS, NCOLS, NCOMP, K = 50, 51, 5, 15
#NROWS, NCOLS, NCOMP, K = 50, 51, 5, 20
IBP_ALPHA = 2.
SIGNAL_VAR = 1.
NOISE_VAR = 0.01

BASEDIR = '/afs/csail/u/r/rgrosse/cached/factorize/regression_tests'
#BASEDIR = os.path.join(config.CACHE_PATH, 'regression_tests')

TEMP_FILL_IN = False
TEMP_NO_MISSING = False

Ex = Exception


PMF_MODEL = ('+', ('*', 'g', 'g'), 'g')
MOG_MODEL = ('+', ('*', 'm', 'g'), 'g')
MOG_TRANSPOSE_MODEL = ('+', ('*', 'g', 'M'), 'g')
IBP_MODEL = ('+', ('*', 'b', 'g'), 'g')
IBP_TRANSPOSE_MODEL = ('+', ('*', 'g', 'B'), 'g')
IRM_MODEL = ('+', ('*', 'm', ('+', ('*', 'g', 'M'), 'g')), 'g')
IRM_TRANSPOSE_MODEL = ('+', ('*', ('+', ('*', 'm', 'g'), 'g'), 'M'), 'g')
BMF_MODEL = ('+', ('*', 'b', ('+', ('*', 'g', 'B'), 'g')), 'g')
CHAIN_MODEL = ('+', ('*', 'c', 'g'), 'g')
CHAIN_TRANSPOSE_MODEL = ('+', ('*', 'g', 'C'), 'g')
KF_MODEL = ('+', ('*', ('+', ('*', 'c', 'g'), 'g'), 'g'), 'g')
SPARSE_CODING_MODEL = ('+', ('*', 's', 'g'), 'g')



def md5(obj):
    return hashlib.md5(str(obj)).hexdigest()

def data_file(data_str):
    return os.path.join(BASEDIR, 'data/%s.pickle' % data_str)
def obs_file():
    return os.path.join(BASEDIR, 'data/obs.pickle')
def samples_file(expt, data_str, estimator_str, missing):
    if missing:
        return os.path.join(BASEDIR, '%s/samples/%s_%s-missing.pickle' % (expt, data_str, md5(estimator_str)))
    else:
        return os.path.join(BASEDIR, '%s/samples/%s_%s.pickle' % (expt, data_str, md5(estimator_str)))
def results_file(expt, missing):
    if missing:
        return os.path.join(BASEDIR, '%s/results-missing.pickle' % expt)
    else:
        return os.path.join(BASEDIR, '%s/results.pickle' % expt)
def expt_path(expt):
    return os.path.join(BASEDIR, expt)
def samples_path(expt):
    return os.path.join(expt_path(expt), 'samples')


ALL_DATA_STR = ['pmf', 'mog', 'mogT', 'ibp', 'ibpT', 'irm', 'bmf', 'chain', 'chainT', 'kf']

def generate_data(data_str, nrows=NROWS, ncols=NCOLS):
    pi_crp = np.ones(NCOMP) / NCOMP
    pi_ibp = np.ones(NCOMP) * IBP_ALPHA / NCOMP

    if data_str[-1] == 'T':
        data_str = data_str[:-1]
        transpose = True
        nrows, ncols = NCOLS, NROWS
    else:
        transpose = False
        nrows, ncols = NROWS, NCOLS
    train_rows, train_cols = range(nrows), range(ncols)
    
    if data_str == 'pmf':
        U = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(2*nrows, NCOMP))
        V = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(NCOMP, 2*ncols))
        left = recursive.ProductNode([recursive.GaussianNode(U),
                                       recursive.GaussianNode(V)])
        
    elif data_str == 'mog':
        U = np.random.multinomial(1, pi_crp, size=2*nrows)
        V = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(NCOMP, 2*ncols))
        left = recursive.ProductNode([recursive.MultinomialNode(U),
                                       recursive.GaussianNode(V)])

    elif data_str == 'ibp':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(2*nrows, NCOMP))
        V = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(NCOMP, 2*ncols))
        left = recursive.ProductNode([recursive.BernoulliNode(U),
                                       recursive.GaussianNode(V)])

    elif data_str == 'irm':
        U = np.random.multinomial(1, pi_crp, size=2*nrows)
        R = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(NCOMP, NCOMP))
        V = np.random.multinomial(1, pi_crp, size=2*ncols).T
        left = recursive.ProductNode([recursive.MultinomialNode(U),
                                       recursive.GaussianNode(R),
                                       recursive.MultinomialTNode(V)])

    elif data_str == 'bmf':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(2*nrows, NCOMP))
        R = np.random.normal(0., np.sqrt(SIGNAL_VAR), size=(NCOMP, NCOMP))
        V = np.random.binomial(1, pi_ibp[nax,:], size=(2*ncols, NCOMP)).T
        left = recursive.ProductNode([recursive.BernoulliNode(U),
                                       recursive.GaussianNode(R),
                                       recursive.BernoulliTNode(V)])

    elif data_str == 'chain':
        temp = np.random.permutation(2*nrows)
        train_rows = np.array(sorted(temp[:nrows]))
        D = np.random.normal(size=(2*nrows, 2*ncols))
        temp = D.cumsum(0)
        mult = 1. / np.sqrt(np.mean(temp**2))
        D *= mult
        left = recursive.ProductNode([recursive.IntegrationNode(algorithms.chains.integration_matrix(2*nrows)),
                                       recursive.GaussianNode(D)])

    elif data_str == 'kf':
        temp = np.random.permutation(2*nrows)
        train_rows = np.array(sorted(temp[:nrows]))
        D = np.random.normal(size=(2*nrows, NCOMP))
        temp = D.cumsum(0)
        mult = 1. / np.sqrt(np.mean(temp**2))
        D *= mult
        V = np.random.normal(size=(NCOMP, 2*ncols))
        left = recursive.ProductNode([recursive.IntegrationNode(algorithms.chains.integration_matrix(2*nrows)),
                                       recursive.GaussianNode(D),
                                       recursive.GaussianNode(V)])
        
        
    noise = np.random.normal(0., np.sqrt(NOISE_VAR), size=(2*nrows, 2*ncols))
    right = recursive.GaussianNode(noise)
    model = recursive.SumNode([left, right])

    if transpose:
        return model.transpose().value(), train_cols, train_rows
    else:
        return model.value(), train_rows, train_cols


PMF_SEQ = [PMF_MODEL]
MOG_SEQ = [MOG_MODEL]
MOGT_SEQ = [MOG_TRANSPOSE_MODEL]
IBP_SEQ = [IBP_MODEL]
IBPT_SEQ = [IBP_TRANSPOSE_MODEL]
IRM_SEQ = [MOG_MODEL, IRM_MODEL]
IRMT_SEQ = [MOG_TRANSPOSE_MODEL, IRM_TRANSPOSE_MODEL]
BMF_SEQ = [IBP_MODEL, BMF_MODEL]
CHAIN_SEQ = [CHAIN_MODEL]
CHAINT_SEQ = [CHAIN_TRANSPOSE_MODEL]
KF_SEQ = [PMF_MODEL, KF_MODEL]

TO_TEST = {'pmf': [PMF_SEQ],
           'mog': [MOG_SEQ],
           'mogT': [MOGT_SEQ],
           'ibp': [IBP_SEQ],
           'ibpT': [IBPT_SEQ],
           #'irm': [MOG_SEQ, MOGT_SEQ, IBP_SEQ, IRM_SEQ, IRMT_SEQ],
           #'irm': [IRMT_SEQ],
           'irm': [MOG_SEQ, MOGT_SEQ, IBP_SEQ, IRM_SEQ],
           'bmf': [IBP_SEQ, IBPT_SEQ, BMF_SEQ],
           'chain': [CHAIN_SEQ],
           'chainT': [CHAINT_SEQ],
           'kf': [KF_SEQ]}

def save_data():
    #assert False
    for data_str in ALL_DATA_STR:
    #for data_str in ['chain', 'chainT']:
        data = generate_data(data_str)
        cPickle.dump(data, open(data_file(data_str), 'w'), protocol=2)
    obs = np.random.binomial(1, 0.9, size=(2*NROWS, 2*NCOLS))
    cPickle.dump(obs, open(obs_file(), 'w'))

def save_obs():
    obs = np.random.binomial(1, 0.9, size=(2*NROWS, 2*NCOLS)).astype(bool)
    cPickle.dump(obs, open(obs_file(), 'w'))

def get_data_matrix(data_str, missing):
    values, train_rows, train_cols = cPickle.load(open(data_file(data_str)))
    #data = recursive.Decomp(recursive.GaussianNode(data))
    if missing:
        obs = cPickle.load(open(obs_file()))
        if not TEMP_FILL_IN:
            values = np.where(obs, values, 10000.)
        data_matrix = observations.DataMatrix.from_real_values(values, obs)
    else:
        data_matrix = observations.DataMatrix.from_real_values(values)
    return data_matrix, np.array(train_rows), np.array(train_cols)

def run_one(expt, data_str, sequence, missing=False):
    data_matrix, train_rows, train_cols = get_data_matrix(data_str, missing)
    test_rows = np.array([i for i in range(2*NROWS) if i not in train_rows])
    test_cols = np.array([i for i in range(2*NCOLS) if i not in train_cols])

    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]
    print '****** Data: %s ******' % data_str
    print '*** Estimator: %s ***' % grammar.pretty_print(sequence[-1])

    model = X_train
    for structure in sequence:
        model = recursive.fit_model(structure, X_train, model)

    row_loglik, col_loglik = scoring.evaluate_model(X_train, model, X_row_test, X_col_test)
    

    

def run_models(expt, num_samples=1, subset=None, missing=False):
    #assert expt != 'baseline'
    if not os.path.exists(expt_path(expt)):
        os.mkdir(expt_path(expt))
    if not os.path.exists(samples_path(expt)):
        os.mkdir(samples_path(expt))

    for data_str, sequences in TO_TEST.items():
        if subset is not None and data_str not in subset:
            continue
        print '****** Data: %s ******' % data_str

        if TEMP_NO_MISSING:
            data_matrix, train_rows, train_cols = get_data_matrix(data_str, False)
        else:
            data_matrix, train_rows, train_cols = get_data_matrix(data_str, missing)
        X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

        # temporary
        if TEMP_FILL_IN and False:
            X_train.observations.mask[:, :] = True

        for sequence in sequences:
            print '*** Estimator: %s ***' % grammar.pretty_print(sequence[-1])
            samples = []
            for s in range(num_samples):
                #model = X_train
                model = recursive.GaussianNode(X_train.observations.values, 'scalar', 1.)
                try:
                    for structure in sequence:
                        model = recursive.fit_model(structure, X_train, model)
                    samples.append(model)
                except Ex:
                    samples.append('FAIL')
                    traceback.print_exc()
                    time.sleep(5)
            #print 'len(samples)', len(samples)
            #print expt, data_str, grammar.pretty_print(sequence[-1]), missing
            cPickle.dump(samples, open(samples_file(expt, data_str, sequence[-1], missing), 'w'), protocol=2)

def compute_results(expt, subset=None, missing=False):
    #assert expt != 'baseline'
    if os.path.exists(results_file(expt, missing)):
        row_results, col_results = cPickle.load(open(results_file(expt, missing)))
    else:
        row_results, col_results = {}, {}

    for data_str, sequences in TO_TEST.items():
        if subset is not None and data_str not in subset:
            continue
        print '****** Data: %s ******' % data_str

        if TEMP_NO_MISSING:
            data_matrix, train_rows, train_cols = get_data_matrix(data_str, False)
        else:
            data_matrix, train_rows, train_cols = get_data_matrix(data_str, missing)
        test_rows = np.array([i for i in range(2*NROWS) if i not in train_rows])
        test_cols = np.array([i for i in range(2*NCOLS) if i not in train_cols])
        X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
        X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
        X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]
        
        for sequence in sequences:
            print '*** Estimator: %s ***' % grammar.pretty_print(sequence[-1])
            structure = sequence[-1]
            samples = cPickle.load(open(samples_file(expt, data_str, structure, missing)))
            #print '***len(samples)', len(samples)
            #print expt, data_str, grammar.pretty_print(structure), missing

            row_results[data_str, structure] = []
            col_results[data_str, structure] = []
            for sample in samples:
                try:
                    assert sample is not 'FAIL'
                    row_loglik, col_loglik = scoring.evaluate_model(X_train, sample, X_row_test, X_col_test)

                    row_results[data_str, structure].append(np.mean(row_loglik))
                    col_results[data_str, structure].append(np.mean(col_loglik))
                except Ex:
                    row_results[data_str, structure].append('FAIL')
                    col_results[data_str, structure].append('FAIL')
                    traceback.print_exc()
                    time.sleep(5)
                
    cPickle.dump((row_results, col_results), open(results_file(expt, missing), 'w'))
    
                
                
def summarize_results(expt, tol=2., subset=None, missing=False):
    baseline_row_results, baseline_col_results = cPickle.load(open(results_file('baseline', missing)))
    row_results, col_results = cPickle.load(open(results_file(expt, missing)))

    for data_str, sequences in TO_TEST.items():
        if subset is not None and data_str not in subset:
            continue
        print 'Data:', data_str
        for sequence in sequences:
            structure = sequence[-1]
            print 'Structure:', grammar.pretty_print(structure)

            bas = baseline_row_results[data_str, structure]
            bas_fail = ('FAIL' in bas)
            curr = row_results[data_str, structure]
            print '   Baseline row:', min(bas)
            for c in curr:
                if bas_fail or str(c) == 'FAIL' or c < min(bas) - tol:
                    #print '   Row:', c, '<------------------------------------------'
                    print termcolor.colored('   Row: %s' % c, 'red', attrs=['bold'])
                else:
                    print '   Row:', c
            
            bas = baseline_col_results[data_str, structure]
            bas_fail = ('FAIL' in bas)
            curr = col_results[data_str, structure]
            print '   Baseline column:', min(bas)
            for c in curr:
                if bas_fail or str(c) == 'FAIL' or c < min(bas) - tol:
                    #print '   Column:', c, '<------------------------------------------'
                    print termcolor.colored('   Column: %s' % c, 'red', attrs=['bold'])
                else:
                    print '   Column:', c
        print

def run_all(expt='default', subset=None, missing=False, num_trials=None):
    t0 = time.time()
    if expt == 'baseline' and num_trials is None:
        num_trials = 5
    elif expt != 'baseline' and num_trials is None:
        num_trials = 3
    run_models(expt, num_trials, subset=subset, missing=missing)
    compute_results(expt, subset=subset, missing=missing)
    summarize_results(expt, subset=subset, missing=missing)
    print 'Tests took %1.1f minutes' % ((time.time() - t0) / 60.)

