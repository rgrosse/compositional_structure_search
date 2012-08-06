import cPickle
import numpy as np
nax = np.newaxis
import pylab
import scipy.linalg

import data
import experiments
import grammar
import recursive


def visualize_matrix(X, row_assignments, col_assignments, row_order_within, col_order_within,
                     row_center_order, col_center_order):
    m, n = X.shape
    if row_assignments is None:
        row_assignments = np.zeros(m, dtype=int)
    if col_assignments is None:
        col_assignments = np.zeros(n, dtype=int)
    if row_order_within is None:
        row_order_within = np.arange(m)
    if col_order_within is None:
        col_order_within = np.arange(n)
    if row_center_order is None:
        row_center_order = np.arange(row_assignments.max() + 1)
    if col_center_order is None:
        col_center_order = np.arange(col_assignments.max() + 1)
        

    def row_cmp(i1, i2):
        if row_assignments[i1] != row_assignments[i2]:
            return cmp(row_center_order[row_assignments[i1]], row_center_order[row_assignments[i2]])
        else:
            return cmp(row_order_within[i1], row_order_within[i2])

    def col_cmp(i1, i2):
        if col_assignments[i1] != col_assignments[i2]:
            return cmp(col_center_order[col_assignments[i1]], col_center_order[col_assignments[i2]])
        else:
            return cmp(col_order_within[i1], col_order_within[i2])

    
    row_idxs = np.array(sorted(range(m), cmp=row_cmp))
    col_idxs = np.array(sorted(range(n), cmp=col_cmp))

    pylab.matshow(X[row_idxs[:, nax], col_idxs[nax, :]])
    pylab.gray()
    pylab.axis('off')
    
    for i in range(row_idxs.size - 1):
        if row_assignments is not None and \
               row_assignments[row_idxs[i]] != row_assignments[row_idxs[i+1]]:
            pylab.axhline(i + 0.5, color='b', linewidth=2)

    for j in range(col_idxs.size - 1):
        if col_assignments is not None and \
               col_assignments[col_idxs[j]] != col_assignments[col_idxs[j+1]]:
            pylab.axvline(j + 0.5, color='b', linewidth=2)


def get_expt_name(year):
    if year == 2010:
        return 'senate-6-21'
    elif year == 1992:
        return 'senate-1992'
    else:
        raise RuntimeError('Unknown year: %d' % year)

def load_data_matrix(year):
    expt_name = get_expt_name(year)
    data_matrix = data.load_senate_data(year, False)
    splits = cPickle.load(open(experiments.splits_file(expt_name)))
    train_rows, train_cols, test_rows, test_cols = splits[0]
    return data_matrix[train_rows[:, nax], train_cols[nax, :]]
    

def visualize_senate(year, level):
    assert level in [1, 2, 3]
    expt_name = get_expt_name(year)
    data_matrix = load_data_matrix(year)

    if level == 1:
        sample = cPickle.load(open(experiments.samples_file(expt_name, 1, grammar.parse('gM+g'), 0, 0)))
    elif level == 2:
        sample = cPickle.load(open(experiments.samples_file(expt_name, 2, grammar.parse('(mg+g)M+g'), 0, 0)))
    elif level == 3:
        sample = cPickle.load(open(experiments.samples_file(expt_name, 3, grammar.parse('(mg+g)(gM+g)+g'), 0, 0)))

    #col_idxs = np.random.permutation(sample.n)[:150]
    #sample = sample[:, col_idxs]
    #data_matrix = data_matrix[:, col_idxs]
    obs = data_matrix.observations
    X = np.where(obs.mask, obs.values.astype(int), 0.5)

    if level in [1, 2]:
        col_assignments = sample.descendant('lr').value().argmax(0)
    elif level == 3:
        col_assignments = sample.descendant('lrlr').value().argmax(0)

    if level == 1:
        row_assignments = None
    elif level in [2, 3]:
        row_assignments = sample.descendant('llll').value().argmax(1)

    if level == 3:
        signal = sample.descendant('l').value()
        signal -= signal.mean(0)[nax, :]
        signal -= signal.mean(1)[:, nax]
        U, s, Vh = scipy.linalg.svd(signal)

        row_order_within = U[:, 0]
        col_order_within = Vh[0, :]

        row_centers = np.dot(sample.descendant('lllr').value(), sample.descendant('lr').value())
        row_center_proj = np.dot(row_centers, Vh[0, :])
        row_center_order = row_center_proj

        col_centers = np.dot(sample.descendant('ll').value(), sample.descendant('lrll').value())
        col_center_proj = np.dot(U[0, :], col_centers)
        col_center_order = col_center_proj
    else:
        row_order_within = col_order_within = row_center_order = col_center_order = None


    np.random.seed(0)
    col_idxs = np.random.permutation(sample.n)[:150]
    X = X[:, col_idxs]
    if col_assignments is not None:
        col_assignments = col_assignments[col_idxs]
    if col_order_within is not None:
        col_order_within = col_order_within[col_idxs]
        
    visualize_matrix(1 - X, row_assignments, col_assignments, row_order_within, col_order_within,
                     row_center_order, col_center_order)
    


def visualize_senate_pmf(year):
    expt_name = get_expt_name(year)
    data_matrix = load_data_matrix(year)

    sample = cPickle.load(open(experiments.samples_file(expt_name, 1, grammar.parse('gg+g'), 0, 0)))

    col_idxs = np.random.permutation(sample.n)[:150]
    sample = sample[:, col_idxs]
    data_matrix = data_matrix[:, col_idxs]
    obs = data_matrix.observations
    X = np.where(obs.mask, obs.values.astype(int), 0.5)

    signal = sample.descendant('l').value()
    
    signal -= signal.mean(0)[nax, :]
    signal -= signal.mean(1)[:, nax]
    U, s, Vh = scipy.linalg.svd(signal)
    
    row_order_within = U[:, 0]
    col_order_within = Vh[0, :]

    visualize_matrix(1 - X, None, None, row_order_within, col_order_within, None, None)
    


def save_figures():
    visualize_senate(2010, 1)
    pylab.savefig('/tmp/roger/senate/2010-clustering.pdf')
    visualize_senate(2010, 2)
    pylab.savefig('/tmp/roger/senate/2010-irm.pdf')
    visualize_senate(2010, 3)
    pylab.savefig('/tmp/roger/senate/2010-bctf.pdf')
    visualize_senate_pmf(2010)
    pylab.savefig('/tmp/roger/senate/2010-pmf.pdf')

    visualize_senate(1992, 1)
    pylab.savefig('/tmp/roger/senate/1992-clustering.pdf')
    visualize_senate(1992, 2)
    pylab.savefig('/tmp/roger/senate/1992-irm.pdf')
    visualize_senate(1992, 3)
    pylab.savefig('/tmp/roger/senate/1992-bctf.pdf')
    visualize_senate_pmf(1992)
    pylab.savefig('/tmp/roger/senate/1992-pmf.pdf')


def visualize_intel(level):
    expt_name = 'intel-6-21'
    
    data_matrix = data.load_intel_data(False)
    splits = cPickle.load(open(experiments.splits_file(expt_name)))
    train_rows, train_cols, test_rows, test_cols = splits[0]
    data_matrix = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    if level == 1:
        sample = cPickle.load(open(experiments.samples_file(expt_name, 1, grammar.parse('mg+g'), 0, 0)))
    elif level == 2:
        sample = cPickle.load(open(experiments.samples_file(expt_name, 2, grammar.parse('m(gg+g)+g'), 0, 0)))

    X = data_matrix.observations.values.copy()

    row_assignments = sample.descendant('ll').value().argmax(1)

    if level == 2:
        signal = sample.descendant('lr').value()
        signal -= signal.mean(0)[nax, :]
        signal -= signal.mean(1)[:, nax]
        U, s, Vh = scipy.linalg.svd(signal)

        row_center_order = U[:, 0]
        col_order_within = Vh[0, :]
    else:
        #row_center_order = col_order_within = None
        row_center_order = np.random.normal(size=sample.descendant('ll').n)
        col_order_within = np.random.normal(size=data_matrix.n)


    np.random.seed(0)
    row_idxs = np.random.permutation(sample.m)[:250]
    X = X[row_idxs, :]
    row_assignments = row_assignments[row_idxs]

    row_order_within = np.random.normal(size=data_matrix.m)

    visualize_matrix(-X, row_assignments, None, row_order_within, col_order_within, row_center_order, None)
    
def visualize_1992_senate():
    expt_name = 'senate-1992'

    data_matrix = data.load_senate_data(1992, False)
    splits = cPickle.load(open(experiments.splits_file(expt_name)))
    train_rows, train_cols, test_rows, test_cols = splits[0]
    data_matrix = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    sample = cPickle.load(open(experiments.samples_file(expt_name, 2, grammar.parse('g(gM+g)+g'), 0, 0)))

    X = data_matrix.observations.values.copy()

    col_assignments = sample.descendant('lrlr').value().argmax(0)

    signal = sample.descendant('l').value()
    signal -= signal.mean(0)[nax, :]
    signal -= signal.mean(1)[:, nax]
    U, s, Vh = scipy.linalg.svd(signal)
    
    row_order_within = U[:, 0]
    col_order_within = Vh[0, :]

    col_centers = np.dot(sample.descendant('ll').value(), sample.descendant('lrll').value())
    col_center_proj = np.dot(U[0, :], col_centers)
    col_center_order = col_center_proj
    
    np.random.seed(0)
    col_idxs = np.random.permutation(sample.n)[:150]
    X = X[:, col_idxs]
    if col_assignments is not None:
        col_assignments = col_assignments[col_idxs]
    if col_order_within is not None:
        col_order_within = col_order_within[col_idxs]

    visualize_matrix(1 - X, None, col_assignments, row_order_within, col_order_within,
                     None, col_center_order)

    
def print_senate_names():
    year = 2010
    expt_name = get_expt_name(year)
    data_matrix = data.load_senate_data(year, False)
    splits = cPickle.load(open(experiments.splits_file(expt_name)))
    train_rows, train_cols, test_rows, test_cols = splits[0]
    data_matrix = data_matrix[train_rows[:, nax], train_cols[nax, :]]
    vote_labels = data.load_vote_labels(year)
    vote_labels = [vote_labels[i] for i in train_cols]
    data_matrix.col_labels = vote_labels

    sample = cPickle.load(open(experiments.samples_file(expt_name, 2, grammar.parse('(mg+g)M+g'), 0, 0)))

    recursive.print_clusters(data_matrix, sample)


