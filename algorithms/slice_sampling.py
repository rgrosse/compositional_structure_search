import numpy as np
nax = np.newaxis
import time
import pylab

from utils import misc

MAX_ITER = 1000

def slice_sample(log_f, x0, L, U):
    assert L < x0 < U
    log_y = log_f(x0) + np.log(np.random.uniform(0, 1))

    count = 0
    while True:
        x1 = np.random.uniform(L, U)
        if log_f(x1) >= log_y:
            return x1

        if x1 < x0:
            L = x1
        else:
            U = x1

        count += 1
        if count >= MAX_ITER:
            raise RuntimeError('Exceeded maximum iterations for slice sampling')


class GaussObj:
    def __init__(self, log_f, mu, sigma_sq):
        self.log_f = log_f
        self.mu = mu
        self.sigma_sq = sigma_sq

    def __call__(self, x):
        return self.log_f(x) - 0.5 * (x - self.mu)**2 / self.sigma_sq

def slice_sample_gauss(log_f, mu, sigma_sq, x0):
    sigma = np.sqrt(sigma_sq)
    temp = (x0 - mu) / sigma
    if not -4. <= temp <= 4.:
        # If x takes an extreme value, scipy.special.erf may fail, so fall back to ordinary slice sampling.
        # This isn't a valid sample, since it assumes a contiguous interval, which may not be the case.
        # Hopefully this case doesn't arise too often.
        return slice_sample(GaussObj(log_f, mu, sigma_sq), x0, x0 - 4. * sigma, x0 + 4. * sigma)

    L, U = 1e-10, 1. - 1e-10
    p0 = misc.inv_probit((x0 - mu) / sigma)
    log_y = log_f(x0) + np.log(np.random.uniform(0, 1))

    count = 0
    while True:
        p1 = np.random.uniform(L, U)
        x1 = mu + misc.probit(p1) * sigma
        if log_f(x1) >= log_y:
            #if np.random.binomial(1, 0.001):
            #    print 'Took %d iterations' % count
            return x1

        if p1 < p0:
            L = p1
        else:
            U = p1

        count += 1
        if count >= MAX_ITER:
            raise RuntimeError('Exceeded maximum iterations for slice sampling')
        






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


