import matplotlib
if __name__ == '__main__':
    matplotlib.use('agg')

import hashlib
import numpy as np
nax = np.newaxis
import os
import sys
import termcolor
import time

import config
import grammar
import observations
import recursive
import scoring
from utils import storage


####################### parameters #############################################

class DefaultParams:
    num_splits = 5                 # Number of row/column splits for cross-validation
    num_samples = 5                # Number of independent sampling runs for each model
    num_expand = 3                 # Number of models to expand in each round
    num_steps_ais = 2000           # Number of AIS steps for GSM models
    save_samples = False           # Whether to save the posterior samples (can take up lots of disk space)
    gibbs_steps = 200              # Number of Gibbs steps for sampling from the posterior

    def __setattr__(self, k, v):
        """Make sure the field already exists, to catch typos."""
        if not hasattr(self, k):
            raise RuntimeError("No such field '%s'; maybe a typo?" % k)
        self.__dict__[k] = v

class SmallParams(DefaultParams):
    """Reasonable parameter settings for small matrices"""
    pass

class LargeParams(DefaultParams):
    """Reasonable parameter settings for larger matrices"""
    num_splits = 2





######################## experiment files ######################################

def md5(obj):
    return hashlib.md5(str(obj)).hexdigest()

def experiment_dir(name):
    """Main directory used for all structure search results."""
    basedir = os.path.join(config.RESULTS_PATH, 'predictive')
    return os.path.join(basedir, name)

def data_file(name):
    """The original data matrix, stored as an observations.DataMatrix instance."""
    return os.path.join(experiment_dir(name), 'data.pickle')

def splits_file(name):
    """The cross-validation splits, stored as a list of (train_rows, train_cols,
    test_rows, test_cols) tuples."""
    return os.path.join(experiment_dir(name), 'splits.pickle')

def clean_data_file(name):
    """The observation matrix before noise was added, if applicable."""
    return os.path.join(experiment_dir(name), 'clean-data.pickle')

def components_file(name):
    """The true decomposition, as a recursive.Decomp instance, if applicable."""
    return os.path.join(experiment_dir(name), 'components.pickle')

def level_dir(name, level):
    """The directory containing the results of one level of the search."""
    return os.path.join(experiment_dir(name), 'level%d' % level)

def structures_file(name, level):
    """The list of all structures to be evaluated in a given level of the search.
    Stored as a list of (init_structure, successor_structure) pairs."""
    return os.path.join(level_dir(name, level), 'structures.pickle')

def init_samples_file(name, level, structure, split_id, sample_id):
    """The decomposition to be used as the initialization for a given structure, i.e.
    one of the top performing structures from the previous level."""
    return os.path.join(level_dir(name, level), 'init', 'samples-%s-%d-%d.pickle' % (grammar.pretty_print(structure, False, False),
                                                                                     split_id, sample_id))

def init_scores_file(name, level, structure, split_id, sample_id):
    """The row and column log-likelihood scores for the model used as an initialization.
    Stored as a (row_log_likelihood, column_log_likelihood) pair, where each is a vector
    giving the performance on all the test rows/columns."""
    return os.path.join(level_dir(name, level), 'init', 'scores-%s-%d-%d.pickle' % (grammar.pretty_print(structure, False, False),
                                                                                    split_id, sample_id))

def samples_file(name, level, structure, split_id, sample_id):
    """A posterior sample for a given structure."""
    return os.path.join(config.CACHE_PATH, 'predictive', name, 'level%d' % level,
                        grammar.pretty_print(structure, False, False),
                        'samples-%d-%d.pickle' % (split_id, sample_id))

def scores_file(name, level, structure, split_id, sample_id):
    """The predictive log-likelihood scores on held-out data for a given CV split."""
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'scores-%d-%d.pickle' % (split_id, sample_id))

def collected_scores_file(name, level, structure):
    """The predictive log-likelihood scores for a given structure, collected over all CV
    splits and ordered by the indices in the original data matrix."""
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'collected-scores.pickle')

def winning_structure_file(name, level):
    """The highest performing structure at a given level of the search."""
    return os.path.join(level_dir(name, level), 'winning-structure.pickle')

def running_time_file(name, level, structure, split_id, sample_id):
    """The running time for sampling from the posterior and computing predictive likelihood."""
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'time-%d-%d.pickle' % (split_id, sample_id))

def winning_samples_file(name, sample_id):
    """Posterior samples from each model in the sequence chosen by the structure search."""
    return os.path.join(experiment_dir(name), 'winning-samples-%d.pickle' % sample_id)



def get_params(name):
    if name.find('synthetic') != -1:
        params = SmallParams()
    elif name == 'image-patches':
        params = SmallParams()
        params.save_samples = True
    elif name == 'intel':
        params = LargeParams()
        params.save_samples = True
    elif name == 'mocap':
        params = SmallParams()
        params.save_samples = True
    elif name == 'senate':
        params = SmallParams()
        params.save_samples = True
    else:
        raise RuntimeError('Unknown experiment name: %s' % name)

    return params




############################# initialization ###################################


def nfold_cv(nrows, ncols, nsplits):
    """Randomly split the row and column indices into folds, where one of the
    folds is used as test data in each of the splits."""
    rowperm = np.random.permutation(nrows)
    colperm = np.random.permutation(ncols)
    splits = []
    for i in range(nsplits):
        test_rows = np.array(sorted(rowperm[i*nrows//nsplits:(i+1)*nrows//nsplits]))
        train_rows = np.array([j for j in range(nrows) if j not in test_rows])
        test_cols = np.array(sorted(colperm[i*ncols//nsplits:(i+1)*ncols//nsplits]))
        train_cols = np.array([j for j in range(ncols) if j not in test_cols])
        splits.append((train_rows, train_cols, test_rows, test_cols))
    return splits


def init_experiment(name, data_matrix, components=None, override=False, clean_data_matrix=None):
    """Initialize the structure search by saving the matrix, and possibly auxiliary
    information, to files, and generating cross-validation splits."""
    if os.path.exists(experiment_dir(name)) and not override:
        raise RuntimeError('Experiment %s already initialized.' % name)
    if not os.path.exists(experiment_dir(name)):
        os.mkdir(experiment_dir(name))
    
    params = get_params(name)
    splits = nfold_cv(data_matrix.m, data_matrix.n, params.num_splits)
    storage.dump(splits, splits_file(name))

    if clean_data_matrix is not None:
        storage.dump(clean_data_matrix, clean_data_file(name))

    storage.dump(data_matrix, data_file(name))

    if components is not None:
        storage.dump(components, components_file(name))
    

def list_structure_pairs(init_structures):
    """Expand all of a set of structures. Returns a list of (init_structure, successor_structure) pairs.
    If a structure is a successor to multiple structures in the previous level, keep only the
    best-performing one."""
    pairs = []
    next_structures = set()
    for s in init_structures:
        succ = grammar.list_successors(s)
        for s1 in succ:
            if s1 not in next_structures:
                pairs.append((s, s1))
                next_structures.add(s1)
    return pairs


def init_level(name, level, override=False):
    """Initialize a given level of the search by saving all of the structures which need
    to be evaluated."""
    if not os.path.exists(experiment_dir(name)):
        raise RuntimeError('Experiment %s not yet initialized.' % name)
    if os.path.exists(level_dir(name, level)) and not override:
        raise RuntimeError('Level %d of experiment %s already initialized.' % (level, name))
    
    if level == 1:
        init_structures = ['g']
    else:
        init_structures = storage.load(winning_structure_file(name, level - 1))
    structure_pairs = list_structure_pairs(init_structures)
    storage.dump(structure_pairs, structures_file(name, level))






######################## the actual computation ################################



def sample_from_model(name, level, init_structure, structure, split_id, sample_id):
    """Run an MCMC sampler to approximately sample from the posterior."""
    params = get_params(name)
    data_matrix = storage.load(data_file(name))
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]
    
    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    if level == 1:
        init = X_train.sample_latent_values(np.zeros((X_train.m, X_train.n)), 1.)
        prev_model = recursive.GaussianNode(init, 'scalar', 1.)
    else:
        if params.save_samples:
            prev_model = storage.load(samples_file(name, level-1, init_structure, split_id, sample_id))
        else:
            prev_model = storage.load(init_samples_file(name, level, init_structure, split_id, sample_id))
        if isinstance(prev_model, recursive.Decomp):
            prev_model = prev_model.root

    return recursive.fit_model(structure, X_train, prev_model, gibbs_steps=params.gibbs_steps)

def evaluate_decomp(name, level, init_structure, split_id, sample_id, root):
    """Given a posterior sample, evaluate the predictive likelihood on the test rows and columns."""
    params = get_params(name)
    data_matrix = storage.load(data_file(name))
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]

    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]

    if level == 1:
        init_row_loglik = init_col_loglik = None
    else:
        if params.save_samples:
            init_row_loglik, init_col_loglik = storage.load(scores_file(name, level-1, init_structure,
                                                                        split_id, sample_id))
        else:
            init_row_loglik, init_col_loglik = storage.load(init_scores_file(name, level, init_structure,
                                                                             split_id, sample_id))

    row_loglik, col_loglik = scoring.evaluate_model(X_train, root, X_row_test, X_col_test,
                                                    init_row_loglik=init_row_loglik,
                                                    init_col_loglik=init_col_loglik,
                                                    num_steps=params.num_steps_ais)
    return row_loglik, col_loglik

def run_model(name, level, init_structure, structure, split_id, sample_id, save=True, save_sample=False):
    """Sample from the posterior given the training data, and evaluate on heldout rows/columns."""
    params = get_params(name)
    t0 = time.time()
    root = sample_from_model(name, level, init_structure, structure, split_id, sample_id)
    if save and (save_sample or params.save_samples):
        storage.dump(root, samples_file(name, level, structure, split_id, sample_id))
        print 'Saved.'
    row_loglik, col_loglik = evaluate_decomp(name, level, init_structure, split_id, sample_id, root)
    print 'Row:', row_loglik.mean()
    print 'Col:', col_loglik.mean()
    if save:
        storage.dump((row_loglik, col_loglik), scores_file(name, level, structure, split_id, sample_id))
        storage.dump(time.time() - t0, running_time_file(name, level, structure, split_id, sample_id))
    


def compute_init_samples(name, level, structure, split_id, sample_id):
    """For one of the high-performing structures in the previous level, sample from the posterior
    so that it can be used to initialize the current level. This is only needed if
    params.save_samples == False. The log-likelihood scores are saved as well for purposes
    of determining statistical significance of the improvement over the previous level."""
    if level == 1:
        return

    init_structure = init_structure_for(name, level-1, structure)

    root = sample_from_model(name, level-1, init_structure, structure, split_id, sample_id)
    storage.dump(root, init_samples_file(name, level, structure, split_id, sample_id))
    row_loglik, col_loglik = evaluate_decomp(name, level-1, init_structure, split_id, sample_id, root)
    storage.dump((row_loglik, col_loglik), init_scores_file(name, level, structure, split_id, sample_id))

def fit_winning_sequence(name, num_levels, sample_id):
    """After the sequence of models is identified, sample factorizations from each of the models on the full
    data matrix."""
    data_matrix = storage.load(data_file(name))
    sequence = sequence_of_structures(name, num_levels)
    params = get_params(name)
    decomps = recursive.fit_sequence(sequence, data_matrix, params.k)
    storage.dump(decomps, winning_samples_file(name, sample_id))




############################## scoring #########################################

class PredictiveLikelihoodScores:
    """Summary of statistics relating to the predictive likelihood of a model. The ordering of row and column
    IDs is arbitrary."""
    def __init__(self, row_loglik, col_loglik, num_entries):
        self.row_loglik = row_loglik
        self.col_loglik = col_loglik
        self.num_entries = num_entries

    def total(self):
        return np.sum(self.row_loglik) + np.sum(self.col_loglik)
    def combined(self):
        return np.mean(self.row_loglik) + np.mean(self.col_loglik)
    def avg_per_entry(self):
        return self.total() / self.num_entries
    def row_total(self):
        return np.sum(self.row_loglik)
    def row_avg(self):
        return np.mean(self.row_loglik)
    def col_total(self):
        return np.sum(self.col_loglik)
    def col_avg(self):
        return np.mean(self.col_loglik)



def structureless_scores(name):
    """Evaluate the probability of the structureless model G on held-out data."""
    data_matrix = storage.load(data_file(name))
    if isinstance(data_matrix, recursive.Decomp):
        data_matrix = observations.DataMatrix.from_real_values(data_matrix.root.value())
    splits = storage.load(splits_file(name))

    row_loglik = np.array([])
    col_loglik = np.array([])
    num_entries = 0

    for train_rows, train_cols, test_rows, test_cols in splits:
        X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
        X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
        X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]

        curr_row_loglik = scoring.no_structure_row_loglik(X_train, X_row_test)
        row_loglik = np.concatenate([row_loglik, curr_row_loglik])
        
        curr_col_loglik = scoring.no_structure_col_loglik(X_train, X_col_test)
        col_loglik = np.concatenate([col_loglik, curr_col_loglik])

        num_entries += train_cols.size * test_rows.size + train_rows.size * test_cols.size

    return PredictiveLikelihoodScores(row_loglik, col_loglik, num_entries)

def collect_scores_for_level(name, level):
    """Collect the held-out predictive log-likelihood scores for all CV splits and
    order them according to the indices of the original data matrix."""
    structures = storage.load(structures_file(name, level))
    structures = [s for _, s in structures]
    
    for s in structures:
        collect_scores(name, level, s)
    save_winning_structures(name, level)


def collect_scores(name, level, structure):
    """Collect the held-out predictive log-likelihood scores for all CV splits and
    order them according to the indices of the original data matrix."""
    params = get_params(name)
    splits = storage.load(splits_file(name))

    row_loglik_all = []
    col_loglik_all = []
    failed = False

    for split_id, (train_rows, train_cols, test_rows, test_cols) in enumerate(splits):
        row_loglik_curr, col_loglik_curr = [], []
        num_samples = params.num_samples
        for sample_id in range(num_samples):
            try:
                row_loglik_single, col_loglik_single = storage.load(scores_file(name, level, structure, split_id, sample_id))
            except:
                row_loglik_single = np.nan * np.ones(len(test_rows))
                col_loglik_single = np.nan * np.ones(len(test_cols))
                failed = True
            row_loglik_curr.append(row_loglik_single)
            col_loglik_curr.append(col_loglik_single)

        row_loglik_all.append(np.array(row_loglik_curr))
        col_loglik_all.append(np.array(col_loglik_curr))

    if failed:
        print termcolor.colored('    failed: %s' % grammar.pretty_print(structure), 'red')

    storage.dump((row_loglik_all, col_loglik_all), collected_scores_file(name, level, structure))


def compute_scores(name, level, structure):
    """Average together the predictive likelihood scores over all the posterior samples,
    and return a PredictiveLikelihoodScores instance."""
    if level == 0:
        if structure != 'g': raise RuntimeError('Invalid structure for level 0: %s' % structure)
        return structureless_scores(name)

    params = get_params(name)
    num_samples = params.num_samples
    splits = storage.load(splits_file(name))

    row_loglik_all, col_loglik_all = storage.load(collected_scores_file(name, level, structure))

    # treat errors as zeros (assume I've already checked that all samples for valid models are completed)
    row_loglik_all = [np.where(np.isnan(rl), -np.infty, rl) for rl in row_loglik_all]
    col_loglik_all = [np.where(np.isnan(cl), -np.infty, cl) for cl in col_loglik_all]

    row_loglik_vec, col_loglik_vec = np.array([]), np.array([])
    num_entries = 0
    for split_id, (train_rows, train_cols, test_rows, test_cols) in enumerate(splits):
        row_loglik_curr, col_loglik_curr = row_loglik_all[split_id], col_loglik_all[split_id]
        row_loglik_vec = np.concatenate([row_loglik_vec, np.logaddexp.reduce(row_loglik_curr, axis=0) - np.log(num_samples)])
        col_loglik_vec = np.concatenate([col_loglik_vec, np.logaddexp.reduce(col_loglik_curr, axis=0) - np.log(num_samples)])
        num_entries += train_cols.size * test_rows.size + train_rows.size * test_cols.size

    return PredictiveLikelihoodScores(row_loglik_vec, col_loglik_vec, num_entries)


def init_structure_for(name, level, structure):
    """Determine which of the previous level's structures was used to initialize a given structure."""
    if level == 1:
        return 'g'
    structure_pairs = storage.load(structures_file(name, level))
    init_structure = None
    for init_s, s, in structure_pairs:
        if s == structure:
            init_structure = init_s
    assert init_structure is not None
    return init_structure


def winning_structures(name, level):
    """Determine the set of structures to expand."""
    if level == 0:
        return ['g']
    params = get_params(name)
    structures = storage.load(structures_file(name, level))
    structures = [s for _, s in structures]
    structures = filter(lambda s: compute_scores(name, level, s) is not None, structures)    # ignore failures
    structures.sort(key=lambda s: compute_scores(name, level, s).total(), reverse=True)
    return structures[:params.num_expand]

def save_winning_structures(name, level):
    storage.dump(winning_structures(name, level), winning_structure_file(name, level))

def compute_improvement(name, level, structure=None):
    """Compute the improvement in predictive likelihood score from one level to the next."""
    if structure is None:
        structure = storage.load(winning_structure_file(name, level))
        if type(structure) == list:
            structure = structure[0]
    prev_structure = init_structure_for(name, level, structure)
    curr_scores = compute_scores(name, level, structure)
    prev_scores = compute_scores(name, level-1, prev_structure)
    return (curr_scores.row_avg() - prev_scores.row_avg() + curr_scores.col_avg() - prev_scores.col_avg()) / 2.

def sequence_of_structures(name, num_levels):
    """Get the sequence of structures corresponding to the final model chosen, i.e. a list
    of structures where each one was used to initialize the next one."""
    sequence = []
    for level in range(1, num_levels+1):
        if compute_improvement(name, level) < 1.:
            break
        sequence.append(storage.load(winning_structures(name, level)[0]))
    return sequence



############################# GNU Parallel #####################################

def pretty_print(structure):
    return grammar.pretty_print(structure, False, False)

def list_init_jobs(name, level):
    if level == 1:
        raise RuntimeError('No need for initialization in level 1.')

    winning_structures = storage.load(winning_structure_file(name, level-1))
    winning_structures = filter(lambda s: compute_improvement(name, level-1, s) > 1.,
                                winning_structures)

    params = get_params(name)
    return ['init %s %d %s %d %d' % (name, level, pretty_print(s), split_id, sample_id)
            for s in winning_structures
            for split_id in range(params.num_splits)
            for sample_id in range(params.num_samples)]

def list_jobs(name, level):
    # TODO: only those structures for which there was improvement
##     if level > 1:
##         winning_models = list_winning_models(name, level-1)
##         if winning_models[-1] == '---':
##             return []
    
    params = get_params(name)
    structures = storage.load(structures_file(name, level))
    return ['run %s %d %s %s %d %d' %
            (name, level, pretty_print(init_s), pretty_print(s), split_id, sample_id)
            for init_s, s in structures
            for split_id in range(params.num_splits)
            for sample_id in range(params.num_samples)]

def list_jobs_failed(name, level):
    if level > 1:
        winning_models = list_winning_models(name, level-1)
        if winning_models[-1] == '---':
            return []
    
    params = get_params(name)
    #structures = cPickle.load(open(structures_file(name, level)))
    structures = storage.load(structures_file(name, level))

    jobs = []
    for init_s, s in structures:
        for split_id in range(params.num_splits):
            for sample_id in range(params.num_samples):
                if not os.path.exists(scores_file(name, level, s, split_id, sample_id)):
                    jobs.append(('run', name, level, init_s, s, split_id, sample_id))
                    
    return jobs


            
        

    

def list_winner_jobs(name, num_levels):
    params = get_params(name)
    return ['winner %s %d %d' % (name, num_levels, i) for i in range(params.num_samples)]

def write_jobs(jobs, fname):
    outstr = open(os.path.join(config.JOBS_PATH, fname), 'w')
    for j in jobs:
        print >> outstr, j
    outstr.close()



###################### summarizing the results #################################

def format_table(table, sep='  '):
    num_cols = len(table[0])
    if any([len(row) != num_cols for row in table]):
        raise RuntimeError('Number of columns must match.')

    widths = [max([len(row[i]) for row in table])
              for i in range(num_cols)]
    format_string = sep.join(['%' + str(w) + 's' for w in widths])
    return [format_string % tuple(row) for row in table]

def format_table_latex(table):
    return [l + ' \\\\' for l in format_table(table, ' & ')]
    


def format_structure(structure, latex=False):
    if latex:
        return '$' + grammar.pretty_print(structure).upper().replace("'", "^T") + '$'
    else:
        return grammar.pretty_print(structure)

def list_winning_models(name, num_levels, latex=False):
    entries = []
    done = False
    for level in range(1, num_levels+1):
        if done:
            entries.append('---')
            continue
        
        #try:
        if True:
            if compute_improvement(name, level) > 1.:
                #curr_structure = winning_structure(name, level)
                curr_structure = storage.load(winning_structure_file(name, level))
                if type(curr_structure) == list:
                    curr_structure = curr_structure[0]
                entries.append(format_structure(curr_structure, latex))
            else:
                entries.append('---')
                done = True
        #except:
        else:
            entries.append('ERROR')
            
    return entries

def print_scores(name, level):
    #structures = cPickle.load(open(structures_file(name, level)))
    structures = storage.load(structures_file(name, level))
    structures = [s for _, s in structures]
    scores = {}
    for s in structures:
        result = compute_scores(name, level, s)
        if result is None:
            scores[s] = 'FAIL'
        else:
            scores[s] = (result.row_avg(), result.col_avg(), result.total())
    def temp_key(s):
        if scores[s] == 'FAIL':
            return -np.infty
        else:
            return scores[s][2]
    structures = sorted(structures, key=temp_key, reverse=True)
    print '%30s%10s%10s%10s' % ('structure', 'row', 'col', 'total')
    for s in structures:
        if scores[s] == 'FAIL':
            print '%30s%10s' % (grammar.pretty_print(s), 'FAIL')
        else:
            print '%30s%10.2f%10.2f%10.2f' % (grammar.pretty_print(s), scores[s][0], scores[s][1], scores[s][2])
            
        
            

## def print_clusters(name):
##     sample_id = pick_winning_sample(name)
##     #decomps = cPickle.load(open(winning_samples_file(name, sample_id)))
##     #orig_data = cPickle.load(open(data_file(name)))
##     decomps = storage.load(winning_samples_file(name, sample_id))
##     orig_data = storage.load(data_file(name))
##     recursive.print_clusters(orig_data, decomps[-1])

def print_clusters(name, level, init_structure, structure, split_id, sample_id):
    data_matrix = storage.load(data_file(name))
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]
    
    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    sample = storage.load(samples_file(name, level, structure, split_id, sample_id))

    recursive.print_clusters(X_train, sample)

def print_clusters2(name, sequence):
    sequence = map(grammar.parse, sequence)
    data_matrix = storage.load(data_file(name))
    temp = recursive.fit_sequence(sequence, data_matrix)
    root = temp[-1]
    recursive.print_clusters(data_matrix, root)
    return root

def fit_sequence(name, sequence):
    sequence = map(grammar.parse, sequence)
    data_matrix = storage.load(data_file(name))
    return recursive.fit_sequence(sequence, data_matrix)



def average_running_time(name, level, structure):
    params = get_params(name)
    total = 0.
    for i in range(params.num_splits):
        for j in range(params.num_samples):
            rtf = running_time_file(name, level, structure, i, j)
            total += float(storage.load(rtf))
    return total / float(params.num_samples * params.num_splits)



if __name__ == '__main__':
    print ' '.join(sys.argv)
    print sys.argv
    print 'Machine:', os.uname()[1]
    cmd = sys.argv[1]
    if cmd == 'init':
        name, level, structure_, split_id, sample_id = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), \
                                                       int(sys.argv[6])
        structure = grammar.parse(structure_)
        compute_init_samples(name, level, structure, split_id, sample_id)
    elif cmd == 'run':
        name, level, init_structure_, structure_, split_id, sample_id = sys.argv[2], int(sys.argv[3]), sys.argv[4], \
                                                                        sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
        init_structure = grammar.parse(init_structure_)
        structure = grammar.parse(structure_)
        run_model(name, level, init_structure, structure, split_id, sample_id)
    elif cmd == 'winner':
        name, num_levels, sample_id = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        fit_winning_sequence(name, num_levels, sample_id)



