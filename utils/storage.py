import cPickle
import os

def ensure_directory(d, trial=False):
    parts = d.split('/')
    for i in range(2, len(parts)+1):
        fname = '/'.join(parts[:i])
        if not os.path.exists(fname):
            print 'Creating', fname
            if not trial:
                try:
                    os.mkdir(fname)
                except:
                    pass

def load(fname):
    return cPickle.load(open(fname))

def dump(obj, fname):
    d, f = os.path.split(fname)
    ensure_directory(d)
    cPickle.dump(obj, open(fname, 'w'), protocol=2)


