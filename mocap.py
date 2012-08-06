import numpy as np
nax = np.newaxis
import os
import scipy.io, scipy.linalg

import config





def load_data():
    #FILE = '/afs/csail/u/r/rgrosse/data/mocap/data.mat'
    FILE = os.path.join(config.DATA_PATH, 'mocap/data.mat')
    data = scipy.io.loadmat(FILE)
    X = data['Motion'][0,0]
    X -= X.mean(0)
    print 'X.shape', X.shape
    good_cols = (np.mean(X**2, axis=0) > 0.0001)
    X = X[:, good_cols]
    X /= np.sqrt(np.mean(X**2, axis=0))[nax,:]
    print 'good_cols', good_cols
    return X


NAMES = ['pelvis', 'lfemur', 'ltibia', 'lfoot', 'ltoes',
         'rfemur', 'rtibia', 'rfoot', 'rtoes',
         'thorax', 'lclavicle', 'lhumerus', 'lradius', 'lhand',
         'rclavicle', 'rhumerus', 'rradius', 'rhand']

def load_names():
    names = reduce(list.__add__, [['%s-%d' % (n, i) for i in range(6)]
                                  for n in NAMES])

    #FILE = '/afs/csail/u/r/rgrosse/data/mocap/data.mat'
    FILE = os.path.join(config.DATA_PATH, 'mocap/data.mat')
    data = scipy.io.loadmat(FILE)
    X = data['Motion'][0,0]
    X -= X.mean(0)
    good_cols = (np.mean(X**2, axis=0) > 0.0001)
    
    return [names[i] for i in np.where(good_cols)[0]]
         


