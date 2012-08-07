import numpy as np
nax = np.newaxis

import experiments
import grammar
from utils import storage


    
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



        

    
