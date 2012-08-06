import cPickle

import config
import os
import time

num_reads = 0
num_writes = 0

def reset_counts():
    global num_reads, num_writes
    num_reads = num_writes = 0

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
    global num_reads
    num_reads += 1
    if config.USE_AMAZON_S3:
        if config.CACHE_AMAZON_S3:
            cache_file = os.path.join(config.S3_CACHE_DIR, fname)
            if not os.path.exists(cache_file):
                obj = amazon.load(fname)
                b, f = os.path.split(cache_file)
                ensure_directory(b)
                cPickle.dump(obj, open(cache_file, 'w'), protocol=2)
                return obj
            else:
                #print 'Loaded %s from cache' % fname
                try:
                    return cPickle.load(open(cache_file))
                except:
                    time.sleep(10)
                    return cPickle.load(open(cache_file))

        else:
            return amazon.load(fname)
    else:
        return cPickle.load(open(fname))

def dump(obj, fname):
    
    global num_writes
    num_writes += 1
    if config.USE_AMAZON_S3:
        if config.DEBUG:
            print 'Would write to', fname
            return
        amazon.dump(obj, fname)
    else:
        d, f = os.path.split(fname)
        ensure_directory(d)
        cPickle.dump(obj, open(fname, 'w'), protocol=2)

def write_jobs(jobs, fname, append=False):
    if config.USE_AMAZON_S3:
        for j in jobs:
            amazon.write_job(j)
        print 'There should be %d jobs.' % len(jobs)
    else:
        if append:
            outstr = open(fname, 'a')
        else:
            outstr = open(fname, 'w')
        for j in jobs:
            format_str = ' '.join(['%s'] * len(j))
            print >> outstr, format_str % j
        outstr.close()

