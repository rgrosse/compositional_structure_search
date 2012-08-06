import numpy as np
nax = np.newaxis
import os
import scipy.io

import config
import experiments
import mocap
import observations



def load_animals_data(real_valued=False):
    matfile = os.path.join(config.DATA_PATH, 'kemp/irmdata/50animalbindat.mat')
    var = scipy.io.loadmat(matfile)
    #var = scipy.io.loadmat('/afs/csail/u/r/rgrosse/data/kemp/irmdata/50animalbindat.mat')
    data = var['data'].astype(bool)
    ndata, nfea = data.shape
    names = [str(var['names'][0,i][0]) for i in range(ndata)]
    features = [str(var['features'][0,i][0]) for i in range(nfea)]

    #return observations.BinaryDataMatrix(data, row_labels=names, col_labels=features)
    if real_valued:
        values = np.random.normal(2 * data + 1, np.sqrt(0.1))
        values -= values.mean()
        values /= values.std()
        return observations.DataMatrix.from_real_values(values, row_labels=names, col_labels=features)
    else:
        return observations.DataMatrix.from_binary_values(data, row_labels=names, col_labels=features)


def load_intel_objects():
    #temp = scipy.io.loadmat('/afs/csail/u/r/rgrosse/data/intel/wordsMri60.mat')
    #matfile = os.path.join(config.DATA_PATH, 'intel/wordsMri60.mat')
    matfile = '/afs/csail/u/r/rgrosse/data/intel/wordsMri60.mat'
    temp = scipy.io.loadmat(matfile)
    objects1 = [str(temp['words60'][i,0][0]) for i in range(60)]
    temp = scipy.io.loadmat('/afs/csail/u/r/rgrosse/data/intel/Intel218Words940.mat')
    objects2 = [str(temp['Intel218Words940'][i,0][0]) for i in range(940)]
    return objects1 + objects2

def load_intel_questions():
    #temp = scipy.io.loadmat('/afs/csail/u/r/rgrosse/data/intel/Intel218Questions.mat')
    #matfile = os.path.join(config.DATA_PATH, 'intel/Intel218Questions.mat')
    matfile = '/afs/csail/u/r/rgrosse/data/intel/Intel218Questions.mat'
    temp = scipy.io.loadmat(matfile)
    return [str(temp['Intel218Questions'][i,0][0]) for i in range(218)]

def read_array(fname):
    lines = open(fname).readlines()
    return np.array([[float(s) for s in line.strip().split()]
                     for line in lines if line.strip() != ''])

def load_intel_data(real_valued=False, noise_variance=0.25):
    #fname = os.path.join(config.DATA_PATH, 'intel/data.txt')
    fname = '/afs/csail/u/r/rgrosse/data/intel/data.txt'
    data = read_array(fname)
    ndata, nfea = data.shape


    row_labels = load_intel_objects()
    col_labels = load_intel_questions()
    values = (2 * data + 2).astype(int)
    if real_valued:
        noisy_values = np.random.normal(data, np.sqrt(noise_variance), size=data.shape)
        noisy_values = (noisy_values - noisy_values.mean()) / noisy_values.std()
        return observations.DataMatrix.from_real_values(noisy_values, row_labels=row_labels, col_labels=col_labels)
    else:
        return observations.DataMatrix.from_integer_values(values, row_labels=row_labels, col_labels=col_labels)


def load_senate_data(year, real_valued=False, noise_variance=0.25):
    if year == 2008:
        fname = '/afs/csail/u/r/rgrosse/data/congress/sen110kh_2008.ord.txt'
        #fname = os.path.join(config.DATA_PATH, 'congress/sen110kh_2008.ord.txt')
    elif year == 1992:
        fname = '/afs/csail/u/r/rgrosse/data/congress/sen102kh.ord'
    elif year == 2010:
        fname = '/afs/csail/u/r/rgrosse/data/congress/sen111kh.ord'
    #elif year == 2009:
    #    fname = '/afs/csail/u/r/rgrosse/data/congress/sen111kh_1st.ord.ord'
    #    #fname = os.path.join(config.DATA_PATH, 'congress/sen111kh_1st.ord.ord')

    records = []
    for line_ in open(fname):
        line = line_.strip()
        votes = line[36:]
        state = line[12:20].strip().lower()
        name = line[25:36].strip().lower()
        records.append((state, name, votes))

    # throw out Bush and everyone who served partial terms
    records = [r for r in records if r[0] != 'usa' and '0' not in r[2]]

    values_mapping = {'1': True, '6': False, '7': False, '9': False}
    mask_mapping = {'1': True, '6': True, '7': True, '9': False}

    ndata, nvotes = len(records), len(records[0][2])
    values = np.zeros((ndata, nvotes), dtype=bool)
    mask = np.zeros((ndata, nvotes), dtype=bool)
    for i, (state, name, votes) in enumerate(records):
        #mapping = {'1': 1, '6': -1, '7': -1, '9': 0}
        values[i,:] = np.array([values_mapping[v] for v in votes])
        mask[i, :] = np.array([mask_mapping[v] for v in votes])
    names = [r[1] for r in records]

    if year == 2008:
        vote_labels = load_vote_labels()
    else:
        vote_labels = None

    #return observations.BinaryDataMatrix(values, row_labels=names, col_labels=vote_labels, mask=mask)
    if real_valued:
        noisy_values = np.random.normal(values.astype(int), np.sqrt(noise_variance))
        noisy_values = (noisy_values - noisy_values.mean()) / noisy_values.std()
        return observations.DataMatrix.from_real_values(noisy_values, mask, row_labels=names, col_labels=vote_labels)
    else:
        return observations.DataMatrix.from_binary_values(values, mask, row_labels=names, col_labels=vote_labels)


def load_vote_labels(year):
    if year == 2008:
        instr = open('/afs/csail/u/r/rgrosse/data/congress/vote-names.txt')
    elif year == 2010:
        instr = open('/afs/csail/u/r/rgrosse/data/congress/s111desc_2010.csv')
    else:
        raise RuntimeError('No vote info for year %d' % year)
    #instr = open(os.path.join(config.DATA_PATH, 'congress/vote-names.txt'))
    instr.readline()
    all_labels = []
    for line in instr:
        all_labels.append(','.join(line.split(',')[6:-2]))
    return all_labels

    
def load_image_patches(ndata=5000):
    side = 12
    patches = ols_field.sample_patches(ols_field.sparsenet_images(), ndata, side)
    X = np.array([p.ravel() for p in patches])
    X -= X.mean(1)[:,nax]
    X /= np.std(X)
    return X









def load_brain_data():
    data = np.load('/afs/csail/u/r/rameshvs/public/data.npz')['data']

    # subtract off means, make unit variance
    data -= data.mean(0)[nax,:]
    data -= data.mean(1)[:,nax]
    data /= np.sqrt(np.mean(data**2))

    # take every 10th row
    data = data[::10, :]

    return observations.DataMatrix.from_real_values(data)


def init_experiment(name, override=False):
    if name in ['intel', 'intel2', 'intel-disc-6-21']:
        data_matrix = load_intel_data()
    elif name in ['senate', 'senate2', 'senate-disc-6-21']:
        data_matrix = load_senate_data(2008)
    elif name == 'animals':
        data_matrix = load_animals_data()
    elif name == 'animals-real':
        data_matrix = load_animals_data(True)
    elif name == 'senate-real':
        data_matrix = load_senate_data(2008, real_valued=True)
    elif name == 'intel-real':
        data_matrix = load_intel_data(real_valued=True)
    elif name in ['senate-real2', 'senate-real3', 'senate-6-20']:
        data_matrix = load_senate_data(2008, real_valued=True, noise_variance=0.083)
    elif name in ['intel-real2', 'intel-real3', 'intel-6-20']:
        data_matrix = load_intel_data(real_valued=True, noise_variance=0.332)
    elif name == 'senate-noiseless':
        data_matrix = load_senate_data(2008, real_valued=True, noise_variance=1e-5)
    elif name == 'intel-noiseless':
        data_matrix = load_intel_data(real_valued=True, noise_variance=1e-5)
    elif name == 'movielens-integer':
        data_matrix = movielens_data.load_data_matrix()
        data_matrix = data_matrix[:, :1000]
    elif name == 'movielens-real':
        data_matrix = movielens_data.load_data_matrix(True)
        data_matrix = data_matrix[:, :1000]
    elif name == 'mocap':
        X = mocap.load_data()
        X = X[:1000, :]
        names = mocap.load_names()
        data_matrix = observations.DataMatrix.from_real_values(X, col_labels=names)
    elif name in ['mocap-small', 'mocap-6-20']:
        X = mocap.load_data()
        X = X[:200, :]
        names = mocap.load_names()
        data_matrix = observations.DataMatrix.from_real_values(X, col_labels=names)
    elif name == 'senate-2010':
        data_matrix = load_senate_data(2010, real_valued=True, noise_variance=0.1)
    elif name == 'senate-1992':
        data_matrix = load_senate_data(1992, real_valued=True, noise_variance=0.1)

    elif name in ['intel-6-21', 'intel-full']:
        data_matrix = load_intel_data(real_valued=True, noise_variance=0.1)
    elif name in ['mocap-6-21', 'mocap-full']:
        X = mocap.load_data()
        X = X[:200, :]
        X /= X.std()
        X = np.random.normal(X, 0.1)
        names = mocap.load_names()
        data_matrix = observations.DataMatrix.from_real_values(X, col_labels=names)
    elif name in ['senate-6-21', 'senate-full']:
        data_matrix = load_senate_data(2010, real_valued=True, noise_variance=0.1)

    elif name == 'image-patches-6-22':
        X = load_image_patches(2000)
        data_matrix = observations.DataMatrix.from_real_values(X)
    elif name == 'image-patches-noisy-6-22':
        X = load_image_patches(2000)
        X = np.random.normal(X, 0.1)
        data_matrix = observations.DataMatrix.from_real_values(X)
    elif name == 'image-patches-large':
        X = load_image_patches(5000)
        X = np.random.normal(X, 0.1)
        data_matrix = observations.DataMatrix.from_real_values(X)
    elif name in ['image-patches-small', 'image-patches-full']:
        X = load_image_patches(1000)
        X = np.random.normal(X, 0.1)
        data_matrix = observations.DataMatrix.from_real_values(X)
        
    else:
        raise RuntimeError('Unknown dataset: %s' % name)

    experiments.init_experiment(name, data_matrix, override=override)


def write_jobs_init(name, level):
    jobs = experiments.list_init_jobs(name, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, '%s/jobs.txt' % name))

def write_jobs_for_level(name, level):
    jobs = experiments.list_jobs(name, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, '%s/jobs.txt' % name))

def write_jobs_failed(name, level):
    jobs = experiments.list_jobs_failed(name, level)
    experiments.write_jobs(jobs, os.path.join(config.JOBS_PATH, '%s/jobs.txt' % name))


