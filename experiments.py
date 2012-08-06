import matplotlib
if __name__ == '__main__':
    matplotlib.use('agg')

import gc
import hashlib
import numpy as np
nax = np.newaxis
import os
import sys
import termcolor
import time
import traceback

import config
import grammar
import observations
import recursive
import scoring
from utils import storage

#BASEDIR = '/afs/csail/u/r/rgrosse/results/factorize/predictive'
BASEDIR = os.path.join(config.RESULTS_PATH, 'predictive')
CONFIG_DIR = './config/predictive'

OLD_SAMPLES_DIR = False


######################## experiment files ######################################

def md5(obj):
    return hashlib.md5(str(obj)).hexdigest()

def experiment_dir(name):
    return os.path.join(BASEDIR, name)
def config_file(name):
    return os.path.join(CONFIG_DIR, '%s-config.txt' % name)
def data_file(name):
    return os.path.join(experiment_dir(name), 'data.pickle')
def splits_file(name):
    return os.path.join(experiment_dir(name), 'splits.pickle')
def clean_data_file(name):
    return os.path.join(experiment_dir(name), 'clean-data.pickle')
def components_file(name):
    return os.path.join(experiment_dir(name), 'components.pickle')
def level_dir(name, level):
    return os.path.join(experiment_dir(name), 'level%d' % level)
def structures_file(name, level):
    return os.path.join(level_dir(name, level), 'structures.pickle')
def init_samples_file(name, level, structure, split_id, sample_id):
    return os.path.join(level_dir(name, level), 'init', 'samples-%s-%d-%d.pickle' % (grammar.pretty_print(structure, False, False),
                                                                                     split_id, sample_id))
def init_scores_file(name, level, structure, split_id, sample_id):
    return os.path.join(level_dir(name, level), 'init', 'scores-%s-%d-%d.pickle' % (grammar.pretty_print(structure, False, False),
                                                                                    split_id, sample_id))
def samples_file(name, level, structure, split_id, sample_id):
    if OLD_SAMPLES_DIR:
        return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                            'samples-%d-%d.pickle' % (split_id, sample_id))
    else:
        return os.path.join(config.CACHE_PATH, 'predictive', name, 'level%d' % level,
                            grammar.pretty_print(structure, False, False),
                            'samples-%d-%d.pickle' % (split_id, sample_id))
def scores_file(name, level, structure, split_id, sample_id):
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'scores-%d-%d.pickle' % (split_id, sample_id))
def collected_scores_file(name, level, structure):
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'collected-scores.pickle')
def winning_structure_file(name, level):
    return os.path.join(level_dir(name, level), 'winning-structure.pickle')
def running_time_file(name, level, structure, split_id, sample_id):
    return os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False),
                        'time-%d-%d.pickle' % (split_id, sample_id))
def winning_samples_file(name, sample_id):
    return os.path.join(experiment_dir(name), 'winning-samples-%d.pickle' % sample_id)
def winning_training_scores_file(name, sample_id):
    return os.path.join(experiment_dir(name), 'winning-training-scores-%d.pickle' % sample_id)

def all_directories(name, level, structures):
    dirs = [level_dir(name, level)]
    dirs.append(os.path.join(level_dir(name, level), 'init'))
    for structure in structures:
        dirs.append(os.path.join(level_dir(name, level), grammar.pretty_print(structure, False, False)))
    return dirs


def create_directory(path):
    assert not config.USE_AMAZON_S3
    idx = 0
    while True:
        try:
            idx = path.index('/', idx+1)
        except ValueError:
            break
        if not os.path.exists(path[:idx]):
            os.mkdir(path[:idx])
    if not os.path.exists(path):
        os.mkdir(path)



PARAM_NAMES = {'num-splits': int,
               'k': int,
               'num-samples': int,
               'noise-var': float,
               'num-winners': int,
               'save-samples': bool,
               'num-steps-ais': int,
               }

def read_config_file(fname, require_all=False):
    params_dir, _ = os.path.split(fname)
    instr = open(fname)
    params = {}
    for line_ in instr:
        line = line_.strip()
        if line == '':
            continue
        
        if line[0] == '#':
            parts = line.split()
            if parts[0] == '#include':
                include_file = os.path.join(params_dir, parts[1])
                include_params = read_config_file(include_file, False)
                for k, v in include_params.items():
                    params[k] = v
            else:
                raise RuntimeError('Unknown macro: %s' % parts[0])
        else:
            parts = map(str.strip, line.split(':'))
            param_name, param_val = parts
            if param_name not in PARAM_NAMES:
                raise RuntimeError('Unknown parameter: %s' % param_name)
            tp = PARAM_NAMES[param_name]
            if tp == bool:
                params[param_name] = eval(param_val)
                assert type(params[param_name]) == bool
            else:
                params[param_name] = tp(param_val)

    if require_all:
        for k in PARAM_NAMES:
            if k not in params:
                raise RuntimeError('Parameter %s not defined' % k)

    return params


############################# initialization ###################################

def is_list_of_pairs(structures):
    # temporary (changed structures_file to be a list of init_structure, structure pairs,
    # but need to deal with the old versions where it's still a list of structures)
    return type(structures[0]) == tuple and len(structures[0]) == 2


def nfold_cv(nrows, ncols, nsplits):
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
    if not config.USE_AMAZON_S3:
        if os.path.exists(experiment_dir(name)) and not override:
            raise RuntimeError('Experiment %s already initialized.' % name)
        create_directory(experiment_dir(name))
    
    params = read_config_file(config_file(name))
    splits = nfold_cv(data_matrix.m, data_matrix.n, params['num-splits'])
    #cPickle.dump(splits, open(splits_file(name), 'w'), protocol=2)
    storage.dump(splits, splits_file(name))

    if clean_data_matrix is not None:
        storage.dump(clean_data_matrix, clean_data_file(name))

    storage.dump(data_matrix, data_file(name))

    if components is not None:
        storage.dump(components, components_file(name))
    

def list_structure_pairs(init_structures):
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
    if not config.USE_AMAZON_S3:
        if not os.path.exists(experiment_dir(name)):
            raise RuntimeError('Experiment %s not yet initialized.' % name)
        if os.path.exists(level_dir(name, level)) and not override:
            raise RuntimeError('Level %d of experiment %s already initialized.' % (level, name))
    
    if level == 1:
        init_structures = ['g']
    else:
        #init_structure = winning_structure(name, level - 1)
        init_structures = storage.load(winning_structure_file(name, level - 1))
    #structures = grammar.list_successors(init_structure)
    structure_pairs = list_structure_pairs(init_structures)
    storage.dump(structure_pairs, structures_file(name, level))


def collect_scores_for_level(name, level):
    # TODO
    #if level > 1:
    #    winning_models = list_winning_models(name, level-1)
    #    if winning_models[-1] == '---':
    #        return
    
    structures = storage.load(structures_file(name, level))
    #if type(structures[0]) == tuple:
    if is_list_of_pairs(structures):
        structures = [s for _, s in structures]

    
    for s in structures:
        collect_scores(name, level, s)
    save_winning_structures(name, level)




######################## the actual computation ################################

def load_data(name):
    data_matrix = storage.load(data_file(name))
    assert isinstance(data_matrix, recursive.Decomp) or isinstance(data_matrix, observations.DataMatrix)
    if isinstance(data_matrix, recursive.Decomp):
        data_matrix = observations.DataMatrix.from_decomp(data_matrix)
    return data_matrix


TEMP_GIBBS_STEPS = 200 # temporary
def sample_from_model(name, level, init_structure, structure, split_id, sample_id):
    params = read_config_file(config_file(name))
    data_matrix = load_data(name)
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]
    
    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    if level == 1:
        init = X_train.sample_latent_values(np.zeros((X_train.m, X_train.n)), 1.)
        prev_model = recursive.GaussianNode(init, 'scalar', 1.)
    else:
        if params['save-samples']:
            prev_model = storage.load(samples_file(name, level-1, init_structure, split_id, sample_id))
        else:
            prev_model = storage.load(init_samples_file(name, level, init_structure, split_id, sample_id))
        if isinstance(prev_model, recursive.Decomp):
            prev_model = prev_model.root

    return recursive.fit_model(structure, X_train, prev_model, gibbs_steps=TEMP_GIBBS_STEPS)

def evaluate_decomp(name, level, init_structure, split_id, sample_id, root):
    params = read_config_file(config_file(name))
    data_matrix = storage.load(data_file(name))
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]

    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = data_matrix[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = data_matrix[train_rows[:, nax], test_cols[nax, :]]

    if level == 1:
        init_row_loglik = init_col_loglik = None
    else:
        if params['save-samples']:
            init_row_loglik, init_col_loglik = storage.load(scores_file(name, level-1, init_structure,
                                                                        split_id, sample_id))
        else:
            init_row_loglik, init_col_loglik = storage.load(init_scores_file(name, level, init_structure,
                                                                             split_id, sample_id))

    if 'num-steps-ais' in params:
        num_steps = params['num-steps-ais']
    else:
        num_steps = 2000
    row_loglik, col_loglik = scoring.evaluate_model(X_train, root, X_row_test, X_col_test,
                                                    init_row_loglik=init_row_loglik,
                                                    init_col_loglik=init_col_loglik,
                                                    num_steps=num_steps)
    return row_loglik, col_loglik

def run_model(name, level, init_structure, structure, split_id, sample_id, save=True, save_sample=False):
    params = read_config_file(config_file(name))
    t0 = time.time()
    root = sample_from_model(name, level, init_structure, structure, split_id, sample_id)
    if save and (save_sample or params['save-samples']):
        storage.dump(root, samples_file(name, level, structure, split_id, sample_id))
        print 'Saved.'
    row_loglik, col_loglik = evaluate_decomp(name, level, init_structure, split_id, sample_id, root)
    print 'Row:', row_loglik.mean()
    print 'Col:', col_loglik.mean()
    if save:
        storage.dump((row_loglik, col_loglik), scores_file(name, level, structure, split_id, sample_id))
        storage.dump(time.time() - t0, running_time_file(name, level, structure, split_id, sample_id))
    


def compute_init_samples(name, level, structure, split_id, sample_id):
    if level == 1:
        return

    init_structure = init_structure_for(name, level-1, structure)

    #structure = storage.load(winning_structure_file(name, level-1))
    root = sample_from_model(name, level-1, init_structure, structure, split_id, sample_id)
    storage.dump(root, init_samples_file(name, level, structure, split_id, sample_id))
    row_loglik, col_loglik = evaluate_decomp(name, level-1, init_structure, split_id, sample_id, root)
    storage.dump((row_loglik, col_loglik), init_scores_file(name, level, structure, split_id, sample_id))

def fit_winning_sequence(name, num_levels, sample_id):
    """After the sequence of models is identified, sample factorizations from each of the models on the full
    data matrix. Compute the TRAINING predictive likelihood of each sample (nothing is left to be held out)
    in order to choose one that wasn't stuck in a local optimum."""
    #data_matrix = storage.load(data_file(name))
    data_matrix = load_data(name)
    sequence = sequence_of_structures(name, num_levels)
    params = read_config_file(config_file(name))
    decomps = recursive.fit_sequence(sequence, data_matrix, params['k'])
    #cPickle.dump(decomps, open(winning_samples_file(name, sample_id), 'w'), protocol=2)
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

def collect_scores(name, level, structure):
    params = read_config_file(config_file(name))
    splits = storage.load(splits_file(name))

    row_loglik_all = []
    col_loglik_all = []
    failed = False

    for split_id, (train_rows, train_cols, test_rows, test_cols) in enumerate(splits):
        row_loglik_curr, col_loglik_curr = [], []
        num_samples = params['num-samples']
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
    if level == 0:
        if structure != 'g': raise RuntimeError('Invalid structure for level 0: %s' % structure)
        return structureless_scores(name)

    params = read_config_file(config_file(name))
    num_samples = params['num-samples']
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
    if level == 0:
        return ['g']
    params = read_config_file(config_file(name))
    structures = storage.load(structures_file(name, level))
    #if type(structures[0]) == tuple:
    if is_list_of_pairs(structures):
        structures = [s for _, s in structures]
    structures = filter(lambda s: compute_scores(name, level, s) is not None, structures)    # ignore failures
    #return max(structures, key=lambda s: compute_scores(name, level, s).total())
    structures.sort(key=lambda s: compute_scores(name, level, s).total(), reverse=True)
    return structures[:params['num-winners']]

def save_winning_structures(name, level):
    storage.dump(winning_structures(name, level), winning_structure_file(name, level))

def compute_improvement(name, level, structure=None):
    if structure is None:
        structure = storage.load(winning_structure_file(name, level))
        if type(structure) == list:
            structure = structure[0]
    prev_structure = init_structure_for(name, level, structure)
    curr_scores = compute_scores(name, level, structure)
    prev_scores = compute_scores(name, level-1, prev_structure)
    return (curr_scores.row_avg() - prev_scores.row_avg() + curr_scores.col_avg() - prev_scores.col_avg()) / 2.

def sequence_of_structures(name, num_levels):
    sequence = []
    for level in range(1, num_levels+1):
        if compute_improvement(name, level) < 1.:
            break
        sequence.append(storage.load(winning_structures(name, level)[0]))
    return sequence

def pick_winning_sample(name):
    params = read_config_file(config_file(name))
    scores = np.zeros(params['num-samples'])
    for sample_id in range(params['num-samples']):
        curr_scores = storage.load(winning_training_scores_file(name, sample_id))
        row_loglik, col_loglik = curr_scores[-1]
        scores[sample_id] = np.sum(row_loglik) + np.sum(col_loglik)
    print scores
    return np.argmax(scores)




############################# GNU Parallel #####################################

def pretty_print(structure):
    return grammar.pretty_print(structure, False, False)

def list_init_jobs(name, level):
    if level == 1:
        raise RuntimeError('No need for initialization in level 1.')

    #winning_models = list_winning_models(name, level-1)
    #if winning_models[-1] == '---':
    #    return []

    winning_structures = storage.load(winning_structure_file(name, level-1))
    winning_structures = filter(lambda s: compute_improvement(name, level-1, s) > 1.,
                                winning_structures)

    params = read_config_file(config_file(name))
    if config.USE_AMAZON_S3:
        return [('init', name, level, s, split_id, sample_id)
                for s in winning_structures
                for split_id in range(params['num-splits'])
                for sample_id in range(params['num-samples'])]
    else:
        return ['init %s %d %s %d %d' % (name, level, pretty_print(s), split_id, sample_id)
                for s in winning_structures
                for split_id in range(params['num-splits'])
                for sample_id in range(params['num-samples'])]

def list_jobs(name, level):
    # TODO: only those structures for which there was improvement
##     if level > 1:
##         winning_models = list_winning_models(name, level-1)
##         if winning_models[-1] == '---':
##             return []
    
    params = read_config_file(config_file(name))
    structures = storage.load(structures_file(name, level))
    if config.USE_AMAZON_S3:
        return [('run', name, level, init_s, s, split_id, sample_id)
                for init_s, s in structures
                for split_id in range(params['num-splits'])
                for sample_id in range(params['num-samples'])]
    else:
        return ['run %s %d %s %s %d %d' %
                (name, level, pretty_print(init_s), pretty_print(s), split_id, sample_id)
                for init_s, s in structures
                for split_id in range(params['num-splits'])
                for sample_id in range(params['num-samples'])]

def list_jobs_failed(name, level):
    assert not config.USE_AMAZON_S3
    if level > 1:
        winning_models = list_winning_models(name, level-1)
        if winning_models[-1] == '---':
            return []
    
    params = read_config_file(config_file(name))
    #structures = cPickle.load(open(structures_file(name, level)))
    structures = storage.load(structures_file(name, level))

    jobs = []
    for init_s, s in structures:
        for split_id in range(params['num-splits']):
            for sample_id in range(params['num-samples']):
                if not os.path.exists(scores_file(name, level, s, split_id, sample_id)):
                    jobs.append(('run', name, level, init_s, s, split_id, sample_id))
                    
    return jobs


            
        

    

def list_winner_jobs(name, num_levels):
    params = read_config_file(config_file(name))
    if config.USE_AMAZON_S3:
        return [('winner', name, num_levels, i) for i in range(params['num-samples'])]
    else:
        return ['winner %s %d %d' % (name, num_levels, i) for i in range(params['num-samples'])]

def write_jobs(jobs, fname):
    if config.USE_AMAZON_S3:
        for j in jobs:
            amazon.write_job(j)
    else:
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
    #if type(structures[0]) == tuple:
    if is_list_of_pairs(structures):
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
    data_matrix = load_data(name)
    splits = storage.load(splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]
    
    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    sample = storage.load(samples_file(name, level, structure, split_id, sample_id))

    recursive.print_clusters(X_train, sample)

def print_clusters2(name, sequence):
    sequence = map(grammar.parse, sequence)
    data_matrix = load_data(name)
    temp = recursive.fit_sequence(sequence, data_matrix)
    root = temp[-1]
    recursive.print_clusters(data_matrix, root)
    return root

def fit_sequence(name, sequence):
    sequence = map(grammar.parse, sequence)
    data_matrix = load_data(name)
    return recursive.fit_sequence(sequence, data_matrix)



def average_running_time(name, level, structure):
    params = read_config_file(config_file(name))
    total = 0.
    for i in range(params['num-splits']):
        for j in range(params['num-samples']):
            rtf = running_time_file(name, level, structure, i, j)
            #total += float(open(rtf).read().strip())
            #try:
            #    total += float(storage.load(rtf))
            #except:
            #    assert not config.USE_AMAZON_S3
            #    rtf = rtf[:-6] + 'txt'
            #    total += float(open(rtf).read().strip())
            total += float(storage.load(rtf))
    return total / float(params['num-samples'] * params['num-splits'])


def worker():
    assert config.USE_AMAZON_S3
    while True:
        finished = worker_single()
        if finished:
            break
        gc.collect()


def worker_single():
    assert config.USE_AMAZON_S3
    amazon.setup_sqs()
    job = amazon.read_message()
    print 'job:', job
    if job is None:
        return True   # finished
    try:
    #if True:
        if job[0] == 'init':
            compute_init_samples(*job[1:])
        elif job[0] == 'run':
            run_model(*job[1:])
        elif job[0] == 'winner':
            fit_winning_sequence(*job[1:])
        else:
            raise RuntimeError('Unknown command: %s' % job[0])
        amazon.write_message_finished()
        amazon.delete_message()
    #else:
    except:
        traceback.print_exc()
        amazon.delete_message()
        amazon.write_message_failed(traceback.format_exc())
    return False   # not finished


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
    elif cmd == 'worker':
        worker()
    elif cmd == 'worker-single':
        worker_single()


