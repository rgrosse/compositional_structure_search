

import time
import pylab








def check_tri():
    def log_f(x):
        if not -1 < x < 1:
            return -np.infty
        return np.log(1 - np.abs(x))

    samples = []
    for s in range(10000):
        x = np.random.uniform(-1., 1.)
        for it in range(100):
            x = slice_sample(log_f, x, -2., 2.)
        samples.append(x)

    pylab.figure()
    pylab.hist(samples, bins=50)   # should be a triangle

def check_tri_gauss():
    mu = -0.5
    sigma_sq = 0.25

    def log_f(x):
        if not -1 < x < 1:
            return -np.infty
        return np.log(1 - np.abs(x)) + 0.5 * (x - mu)**2 / sigma_sq

    samples = []
    t0 = time.time()
    for s in range(100000):
        x = np.random.uniform(-1., 1.)
        for it in range(100):
            x = slice_sample_gauss(log_f, mu, sigma_sq, x)
        samples.append(x)

        if (s+1) % 500 == 0:
            print s+1, '(%1.1f seconds)' % (time.time() - t0)
            t0 = time.time()
            
    pylab.figure()
    pylab.hist(samples, bins=50)  # should be a triangle


