import numpy as np
nax = np.newaxis
import os
import pylab

import experiments
import grammar
import scoring
from utils import storage


def run():
    NAME = 'movielens-integer'
    STRUCTURE = grammar.parse('bg+g')
    SPLIT_ID = 0
    SAMPLE_ID = 0
    root = storage.load(experiments.samples_file(NAME, 1, STRUCTURE, SPLIT_ID, SAMPLE_ID))
    #row_loglik, col_loglik = experiments.evaluate_decomp(NAME, 1, 'g', SPLIT_ID, SAMPLE_ID, root)
    data_matrix = storage.load(experiments.data_file(NAME))
    splits = storage.load(experiments.splits_file(NAME))
    train_rows, train_cols, test_rows, test_cols = splits[SPLIT_ID]

    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]

    row_loglik, col_loglik = scoring.evaluate_model(X_train, root, X_row_test, X_col_test, num_steps=50)

def compare_scores(name, structure1, structure2):
    for i in range(1, 4):
        fname = experiments.collected_scores_file(name, i, grammar.parse(structure1))
        if os.path.exists(fname):
            pls1 = experiments.compute_scores(name, i, grammar.parse(structure1))
            break

    for i in range(1, 4):
        fname = experiments.collected_scores_file(name, i, grammar.parse(structure2))
        if os.path.exists(fname):
            pls2 = experiments.compute_scores(name, i, grammar.parse(structure2))
            break



    title = 'Rows'
    vis.figure(title)
    pylab.clf()
    mn, mx = min(pls1.row_loglik), max(pls1.row_loglik)
    pylab.plot(pls1.row_loglik, pls2.row_loglik, 'rx',
               [mn, mx], [mn, mx], 'b-')
    pylab.xlabel(structure1)
    pylab.ylabel(structure2)
    pylab.title(title)

    title = 'Rows (difference)'
    vis.figure(title)
    pylab.clf()
    mn, mx = min(pls1.row_loglik), max(pls1.row_loglik)
    pylab.plot(pls1.row_loglik, pls2.row_loglik - pls1.row_loglik, 'rx',
               [mn, mx], [0., 0.], 'b-')
    pylab.xlabel(structure1)
    pylab.ylabel(structure2)
    pylab.title(title)
    
    title = 'Columns'
    vis.figure(title)
    pylab.clf()
    mn, mx = min(pls1.col_loglik), max(pls1.col_loglik)
    pylab.plot(pls1.col_loglik, pls2.col_loglik, 'rx',
               [mn, mx], [mn, mx], 'b-')
    pylab.xlabel(structure1)
    pylab.ylabel(structure2)
    pylab.title(title)
    
    title = 'Columns (difference)'
    vis.figure(title)
    pylab.clf()
    mn, mx = min(pls1.col_loglik), max(pls1.col_loglik)
    pylab.plot(pls1.col_loglik, pls2.col_loglik - pls1.col_loglik, 'rx',
               [mn, mx], [0., 0.], 'b-')
    pylab.xlabel(structure1)
    pylab.ylabel(structure2)
    pylab.title(title)

    
def compute_z_scores(name, structure1, structure2):
    for i in range(4):
        try:
            pls1 = experiments.compute_scores(name, i, grammar.parse(structure1))
            break
        except:
            pass

    for i in range(4):
        try:
            pls2 = experiments.compute_scores(name, i, grammar.parse(structure2))
            break
        except:
            pass

    diff = pls2.row_loglik - pls1.row_loglik
    mean = diff.mean()
    std = diff.std() / np.sqrt(diff.size)
    print 'Rows:', mean / std

    row_sum = diff.sum()
    row_std = diff.std() * np.sqrt(diff.size)

    diff = pls2.col_loglik - pls1.col_loglik
    mean = diff.mean()
    std = diff.std() / np.sqrt(diff.size)
    print 'Columns:', mean / std

    col_sum = diff.sum()
    col_std = diff.std() * np.sqrt(diff.size)

    print 'Total:', (row_sum + col_sum) / np.sqrt(row_std**2 + col_std**2)

    print 'Required row improvement:', 5 * row_std / pls1.row_loglik.size
    print 'Required column improvement:', 5 * col_std / pls1.col_loglik.size


def print_significance(expt_name, level):
    structures = storage.load(experiments.structures_file(expt_name, level))
    if experiments.is_list_of_pairs(structures):
        structures = [s for _, s in structures]
    structures = filter(lambda s: experiments.compute_scores(expt_name, level, s) is not None,
                        structures)    # ignore failures
    #return max(structures, key=lambda s: compute_scores(name, level, s).total())
    structures.sort(key=lambda s: experiments.compute_scores(expt_name, level, s).total(), reverse=True)

    scores0 = experiments.compute_scores(expt_name, level, structures[0])

    print 'Winning structure:', grammar.pretty_print(structures[0])
    for st in structures[1:]:
        scores1 = experiments.compute_scores(expt_name, level, st)

        diff = scores1.row_loglik - scores0.row_loglik
        row_sum = diff.sum()
        row_std = diff.std() * np.sqrt(diff.size)

        diff = scores1.col_loglik - scores0.col_loglik
        col_sum = diff.sum()
        col_std = diff.std() * np.sqrt(diff.size)

        total = (row_sum + col_sum) / np.sqrt(row_std ** 2 + col_std ** 2)
        print '    %20s%8.2f' % (grammar.pretty_print(st), total)



        

    
