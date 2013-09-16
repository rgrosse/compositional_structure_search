import matplotlib
if __name__ == '__main__':
    matplotlib.use('agg')

import argparse
import hashlib
import numpy as np
nax = np.newaxis
import os
import StringIO
import sys
import termcolor
import time

import config
import grammar
import observations
import presentation
import recursive
import scoring
from utils import misc, storage

import single_process
import parallel
if config.SCHEDULER == 'single_process':
    schedule_mod = single_process
elif config.SCHEDULER == 'parallel':
    schedule_mod = parallel


####################### parameters #############################################

class DefaultParams:
    num_splits = 5                 # Number of row/column splits for cross-validation
    num_samples = 5                # Number of independent sampling runs for each model
    num_expand = 3                 # Number of models to expand in each round
    num_steps_ais = 2000           # Number of AIS steps for GSM models
    save_samples = False           # Whether to save the posterior samples (can take up lots of disk space)
    gibbs_steps = 200              # Number of Gibbs steps for sampling from the posterior
    search_depth = 3               # Number of steps in the search

    def __setattr__(self, k, v):
        """Make sure the field already exists, to catch typos."""
        if not hasattr(DefaultParams, k):
            raise RuntimeError("No such field '%s'; maybe a typo?" % k)
        self.__dict__[k] = v

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in self.__class__.__dict__.items():       # check for typos in subclasses
            if not hasattr(DefaultParams, k):
                raise RuntimeError("No such field '%s'; maybe a typo in %s class definition?" %
                                   (k, self.__class__))

class SmallParams(DefaultParams):
    """Reasonable parameter settings for small matrices"""
    pass

class LargeParams(DefaultParams):
    """Reasonable parameter settings for larger matrices"""
    num_splits = 2

class DebugParams(DefaultParams):
    """Parameter settings for debugging, so you can quickly run jobs and make sure they don't crash"""
    num_splits = 2
    num_samples = 2
    num_expand = 1
    num_steps_ais = 20
    gibbs_steps = 10



######################## experiment files ######################################

def md5(obj):
    return hashlib.md5(str(obj)).hexdigest()

def experiment_dir(name):
    """Main directory used for all structure search results."""
    return os.path.join(config.RESULTS_PATH, name)

def params_file(name):
    return os.path.join(experiment_dir(name), 'params.pk')

def data_file(name):
    """The original data matrix, stored as an observations.DataMatrix instance."""
    return os.path.join(experiment_dir(name), 'data.pk')

def splits_file(name):
    """The cross-validation splits, stored as a list of (train_rows, train_cols,
    test_rows, test_cols) tuples."""
    return os.path.join(experiment_dir(name), 'splits.pk')

def clean_data_file(name):
    """The observation matrix before noise was added, if applicable."""
    return os.path.join(experiment_dir(name), 'clean-data.pk')

def components_file(name):
    """The true decomposition, as a recursive.Decomp instance, if applicable."""
    return os.path.join(experiment_dir(name), 'components.pk')

def level_dir(name, level):
    """The directory containing the results of one level of the search."""
    return os.path.join(experiment_dir(name), 'level%d' % level)

def structures_file(name, level):
    """The list of all structures to be evaluated in a given level of the search.
    Stored as a list of (init_structure, successor_structure) pairs."""
    return os.path.join(level_dir(name, level), 'structures.pk')

def init_samples_file(name, level, structure, split_id, sample_id):
    """The decomposition to be used as the initialization for a given structure, i.e.
    one of the top performing structures from the previous level."""
    return os.path.join(level_dir(name, level), 'init',
                        'samples-%s-%d-%d.pk' % (md5(structure), split_id, sample_id))

def init_scores_file(name, level, structure, split_id, sample_id):
    """The row and column log-likelihood scores for the model used as an initialization.
    Stored as a (row_log_likelihood, column_log_likelihood) pair, where each is a vector
    giving the performance on all the test rows/columns."""
    return os.path.join(level_dir(name, level), 'init',
                        'scores-%s-%d-%d.pk' % (md5(structure), split_id, sample_id))

def samples_file(name, level, structure, split_id, sample_id):
    """A posterior sample for a given structure."""
    return os.path.join(config.CACHE_PATH,  name,
                        'level%d' % level, md5(structure), 'samples-%d-%d.pk' % (split_id, sample_id))

def scores_file(name, level, structure, split_id, sample_id):
    """The predictive log-likelihood scores on held-out data for a given CV split."""
    return os.path.join(level_dir(name, level), md5(structure), 'scores-%d-%d.pk' % (split_id, sample_id))

def collected_scores_file(name, level, structure):
    """The predictive log-likelihood scores for a given structure, collected over all CV
    splits and ordered by the indices in the original data matrix."""
    return os.path.join(level_dir(name, level), md5(structure), 'collected-scores.pk')

def winning_structure_file(name, level):
    """The highest performing structure at a given level of the search."""
    return os.path.join(level_dir(name, level), 'winning-structure.pk')

def running_time_file(name, level, structure, split_id, sample_id):
    """The running time for sampling from the posterior and computing predictive likelihood."""
    return os.path.join(level_dir(name, level), md5(structure),
                        'time-%d-%d.pk' % (split_id, sample_id))

def winning_samples_file(name, sample_id):
    """Posterior samples from each model in the sequence chosen by the structure search."""
    return os.path.join(experiment_dir(name), 'winning-samples-%d.pk' % sample_id)

def report_dir(name):
    return os.path.join(config.REPORT_PATH, name)

def report_file(name):
    return os.path.join(report_dir(name), 'results.txt')



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


def init_experiment(name, data_matrix, params, components=None, clean_data_matrix=None):
    """Initialize the structure search by saving the matrix, and possibly auxiliary
    information, to files, and generating cross-validation splits."""
    if not os.path.exists(experiment_dir(name)):
        os.mkdir(experiment_dir(name))

    if not os.path.exists(report_dir(name)):
        os.mkdir(report_dir(name))

    storage.dump(params, params_file(name))
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
        succ = grammar.list_collapsed_successors(s)
        for s1 in succ:
            if s1 not in next_structures:
                pairs.append((s, s1))
                next_structures.add(s1)
    return pairs


def init_level(name, level):
    """Initialize a given level of the search by saving all of the structures which need
    to be evaluated."""
    if not os.path.exists(experiment_dir(name)):
        raise RuntimeError('Experiment %s not yet initialized.' % name)
    
    if level == 1:
        init_structures = ['g']
    else:
        init_structures = storage.load(winning_structure_file(name, level - 1))
    structure_pairs = list_structure_pairs(init_structures)
    storage.dump(structure_pairs, structures_file(name, level))






######################## the actual computation ################################



def sample_from_model(name, level, init_structure, structure, split_id, sample_id):
    """Run an MCMC sampler to approximately sample from the posterior."""
    params = storage.load(params_file(name))
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
    params = storage.load(params_file(name))
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
    params = storage.load(params_file(name))
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

def fit_winning_sequence(name, sample_id):
    """After the sequence of models is identified, sample factorizations from each of the models on the full
    data matrix."""
    data_matrix = storage.load(data_file(name))
    sequence = sequence_of_structures(name)
    params = storage.load(params_file(name))
    decomps = recursive.fit_sequence(sequence, data_matrix, gibbs_steps=params.gibbs_steps)
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
    def all_finite(self):
        return np.all(np.isfinite(self.row_loglik)) and np.all(np.isfinite(self.col_loglik))



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
    params = storage.load(params_file(name))
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

    params = storage.load(params_file(name))
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
    params = storage.load(params_file(name))
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

def sequence_of_structures(name):
    """Get the sequence of structures corresponding to the final model chosen, i.e. a list
    of structures where each one was used to initialize the next one."""
    sequence = []
    params = storage.load(params_file(name))

    structure = storage.load(winning_structure_file(name, params.search_depth))[0]
    sequence = [structure]

    for level in range(1, params.search_depth)[::-1]:
        structure = init_structure_for(name, level + 1, structure)
        sequence = [structure] + sequence
    
    return sequence



############################# Job scheduling ###################################

def run_jobs(jobs, args, key):
    if config.SCHEDULER == 'parallel':
        machines = parallel.parse_machines(args.machines, args.njobs)
        parallel.run('experiments.py', jobs, machines=machines, key=key)
    elif config.SCHEULER == 'single_process':
        single_process.run('experiments.py', jobs)
    else:
        raise RuntimeError('Unknown scheduler: %s' % config.SCHEDULER)

def pretty_print(structure):
    return grammar.pretty_print(structure, False, False)

def initial_samples_jobs(name, level):
    if level == 1:
        raise RuntimeError('No need for initialization in level 1.')

    winning_structures = storage.load(winning_structure_file(name, level-1))

    params = storage.load(params_file(name))

    jobs = ["init_job %s %d '%s' %d %d" % (name, level, pretty_print(s), split_id, sample_id)
            for s in winning_structures
            for split_id in range(params.num_splits)
            for sample_id in range(params.num_samples)]

    return jobs

def initial_samples_key(name, level):
    return '%s_init_%d' % (name, level)

def evaluation_jobs(name, level):
    params = storage.load(params_file(name))
    structures = storage.load(structures_file(name, level))
    
    return ["eval_job %s %d '%s' '%s' %d %d" %
            (name, level, pretty_print(init_s), pretty_print(s), split_id, sample_id)
            for init_s, s in structures
            for split_id in range(params.num_splits)
            for sample_id in range(params.num_samples)]

def evaluation_key(name, level):
    return '%s_eval_%d' % (name, level)

def final_model_jobs(name):
    params = storage.load(params_file(name))
    
    return ["final_job %s %d" % (name, i) for i in range(params.num_samples)]

def final_model_key(name):
    return '%s_final' % name

def run_everything(name, args, email=None):
    params = storage.load(params_file(name))
    init_level(name, 1)
    run_jobs(evaluation_jobs(name, 1), args, evaluation_key(name, 1))
    collect_scores_for_level(name, 1)
    for level in range(2, params.search_depth + 1):
        init_level(name, level)
        run_jobs(initial_samples_jobs(name, level), args, initial_samples_key(name, level))
        run_jobs(evaluation_jobs(name, level), args, evaluation_key(name, level))
        collect_scores_for_level(name, level)
    run_jobs(final_model_jobs(name), args, final_model_key(name))
    save_report(name, email)


###################### summarizing the results #################################




def format_structure(structure, latex=False):
    if latex:
        return '$' + grammar.pretty_print(structure).upper().replace("'", "^T") + '$'
    else:
        return grammar.pretty_print(structure)


def print_failures(name, outfile=sys.stdout):
    params = storage.load(params_file(name))
    failures = []
    for level in range(1, params.search_depth + 1):
        for _, structure in storage.load(structures_file(name, level)):
            total_ok = 0
            
            for split_id in range(params.num_splits):
                for sample_id in range(params.num_samples):
                    fname = scores_file(name, level, structure, split_id, sample_id)
                    if os.path.exists(fname):
                        row_loglik, col_loglik = storage.load(fname)
                        if np.all(np.isfinite(row_loglik)) and np.all(np.isfinite(col_loglik)):
                            total_ok += 1

            if total_ok == 0:
                failures.append(presentation.Failure(structure, level, True))
            elif total_ok < params.num_splits * params.num_samples:
                failures.append(presentation.Failure(structure, level, False))

    presentation.print_failed_structures(failures, outfile)
                

def compute_z_score(loglik, prev_loglik):
    diff = loglik - prev_loglik
    mean = diff.mean()
    std = diff.std() / np.sqrt(loglik.size)
    return mean / std

def get_model_score(structure, result, prev_result):
    row_impvt = result.row_avg() - prev_result.row_avg()
    col_impvt = result.col_avg() - prev_result.col_avg()
    z_row = compute_z_score(result.row_loglik, prev_result.row_loglik)
    z_col = compute_z_score(result.col_loglik, prev_result.col_loglik)
    return presentation.ModelScore(structure, result.row_avg(), result.col_avg(), result.total(),
                                   row_impvt, col_impvt, z_row, z_col)

def print_scores(name, level, outfile=sys.stdout):
    structures = storage.load(structures_file(name, level))
    structures = [s for _, s in structures]
    model_scores = []
    for s in structures:
        result = compute_scores(name, level, s)
        if not result.all_finite():
            continue
        if result is not None:
            prev_structure = init_structure_for(name, level, s)
            prev_result = compute_scores(name, level-1, prev_structure)
            model_scores.append(get_model_score(s, result, prev_result))
    model_scores.sort(key=lambda ms: ms.total, reverse=True)
    presentation.print_scores(level, model_scores, outfile)
    
def print_model_sequence(name, outfile=sys.stdout):
    params = storage.load(params_file(name))
    prev_structure = 'g'
    model_scores = []
    for level in range(1, params.search_depth + 1):
        curr_structure = storage.load(winning_structure_file(name, level))[0]
        result = compute_scores(name, level, curr_structure)
        prev_result = compute_scores(name, level-1, prev_structure)
        model_scores.append(get_model_score(curr_structure, result, prev_result))
        prev_structure = curr_structure
    presentation.print_model_sequence(model_scores, outfile)

def print_running_times(name, outfile=sys.stdout):
    params = storage.load(params_file(name))
    running_times = []
    for level in range(1, params.search_depth+1):
        structures = storage.load(structures_file(name, level))
        structures = [s[1] for s in structures]
        for structure in structures:
            total = 0.
            num_samples = 0
            for split in range(params.num_splits):
                for sample_id in range(params.num_samples):
                    rtf = running_time_file(name, level, structure, split, sample_id)
                    try:
                        total += float(storage.load(rtf))
                        num_samples += 1
                    except IOError:
                        pass
            if num_samples > 0:
                running_times.append(presentation.RunningTime(level, structure, num_samples, total))

    presentation.print_running_times(running_times, outfile)
                
        
def summarize_results(name, outfile=sys.stdout):
    params = storage.load(params_file(name))
    print_model_sequence(name, outfile)
    print_failures(name, outfile)
    print_running_times(name, outfile)
    for level in range(1, params.search_depth+1):
        print_scores(name, level, outfile)

def save_report(name, email=None):
    # write to stdout
    summarize_results(name)

    # write to report file
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
        


        
############################# command line #####################################


def add_scheduler_args(parser):
    if config.SCHEDULER == 'parallel':
        parser.add_argument('--machines', type=str, default=':')
        parser.add_argument('--njobs', type=int, default=config.DEFAULT_NUM_JOBS)
    elif config.SCHEDULER == 'single_process':
        pass
    else:
        raise RuntimeError('Unknown scheduler: %s' % config.SCHEDULER)


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('command')

    if command == 'init':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        add_scheduler_args(parser)
        args = parser.parse_args()
        init_level(args.name, args.level)
        if args.level > 1:
            run_jobs(initial_samples_jobs(args.name, args.level), args,
                     initial_samples_key(args.name, args.level))

    elif command == 'init_job':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        parser.add_argument('structure', type=str)
        parser.add_argument('split_id', type=int)
        parser.add_argument('sample_id', type=int)
        args = parser.parse_args()
        compute_init_samples(args.name, args.level, grammar.parse(args.structure),
                             args.split_id, args.sample_id)

    elif command == 'eval':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        add_scheduler_args(parser)
        args = parser.parse_args()
        run_jobs(evaluation_jobs(args.name, args.level), args, evaluation_key(args.name, args.level))
        collect_scores_for_level(args.name, args.level)

    elif command == 'eval_job':
        parser.add_argument('name', type=str)
        parser.add_argument('level', type=int)
        parser.add_argument('init_structure', type=str)
        parser.add_argument('structure', type=str)
        parser.add_argument('split_id', type=int)
        parser.add_argument('sample_id', type=int)
        args = parser.parse_args()
        run_model(args.name, args.level, grammar.parse(args.init_structure), grammar.parse(args.structure),
                  args.split_id, args.sample_id)

    elif command == 'final':
        parser.add_argument('name', type=str)
        add_scheduler_args(parser)
        args = parser.parse_args()
        run_jobs(final_model_jobs(args.name), args, final_model_key(args.name))

    elif command == 'final_job':
        parser.add_argument('name', type=str)
        parser.add_argument('sample_id', type=int)
        args = parser.parse_args()
        fit_winning_sequence(args.name, args.sample_id)

    elif command == 'everything':
        parser.add_argument('name', type=str)
        parser.add_argument('--email', type=str, default=None)
        add_scheduler_args(parser)
        args = parser.parse_args()
        run_everything(args.name, args, email=args.email)

    else:
        raise RuntimeError('Unknown command: %s' % command)

        

