import numpy as np
nax = np.newaxis
import scipy.special

# temporary
ALPHA_CRP = 5


gammaln = scipy.special.gammaln

def uni_gauss_information_to_expectation(lam, J):
    sigma_sq = 1. / lam
    mu = -sigma_sq * J
    return sigma_sq, mu

def uni_gauss_expectation_to_information(sigma_sq, mu):
    lam = 1. / sigma_sq
    J = -lam * mu
    return lam, J

def gauss_loglik(x, mu, sigma_sq):
    return -0.5 * np.log(2*np.pi) - 0.5 * np.log(sigma_sq) \
           - 0.5 * (x - mu)**2 / sigma_sq

def sample_dirichlet(alpha):
    temp = np.random.gamma(alpha)
    return temp / np.sum(temp)

def dirichlet_loglik(alpha, U):
    norm = gammaln(alpha.sum(-1)) - gammaln(alpha).sum(-1)
    return norm + (U * np.log(alpha-1.)).sum(-1)

def dirichlet_multinomial_loglik(alpha, U):
    c = U.sum(0)
    assert alpha.ndim == 1 and alpha.shape == c.shape
    return gammaln(alpha + c).sum(-1) - gammaln(alpha).sum(-1) + \
           gammaln(alpha.sum()) - gammaln(alpha.sum() + c.sum())


def check_dirichlet_multinomial_loglik():
    U = np.array([[1, 0],
                  [1, 0],
                  [0, 1],
                  [1, 0]])
    alpha = np.array([1., 1.])
    assert np.allclose(dirichlet_multinomial_loglik(alpha, U), np.log(1./2 * 2./3 * 1./4 * 3./5))

def beta_bernoulli_loglik(alpha0, alpha1, U):
    M = U.shape[0]
    c = U.sum(0)
    assert alpha0.ndim == 1 and alpha0.shape == alpha1.shape == c.shape
    temp = gammaln(alpha0 + M - c) - gammaln(alpha0) + \
           gammaln(alpha1 + c) - gammaln(alpha1) + \
           gammaln(alpha0 + alpha1 ) - gammaln(alpha0 + alpha1 + M)
    return temp.sum()

def check_beta_bernoulli_loglik():
    U = np.array([[1, 0],
                  [1, 1],
                  [0, 1],
                  [0, 1]])
    alpha0 = np.array([2., 2.])
    alpha1 = np.array([1., 1.])
    result = beta_bernoulli_loglik(alpha0, alpha1, U)
    assert np.allclose(result, np.log(1./3) + np.log(2./4) + np.log(2./5) + np.log(3./6) +
                       np.log(2./3) + np.log(1./4) + np.log(2./5) + np.log(3./6))



class GammaDistribution:
    def __init__(self, a, b):
        if np.shape(a) != np.shape(b):
            raise RuntimeError('a and b should be the same shape')
        self.a = a
        self.b = b

    def expectation(self):
        return self.a / self.b

    def variance(self):
        return self.a / self.b**2

    def expectation_log(self):
        return scipy.special.basic.digamma(self.a) - np.log(self.b)

    def entropy(self):
        return scipy.special.gammaln(self.a) - (self.a - 1.) * scipy.special.basic.digamma(self.a) - np.log(self.b) + self.a

    def sample(self):
        return np.random.gamma(self.a, 1./self.b)
    
    def loglik(self, tau):
        return self.a * np.log(self.b) - scipy.special.gammaln(self.a) + (self.a - 1.) * np.log(tau) - self.b * tau

    def perturb(self, eps=1e-5):
        a = self.a * np.exp(np.random.normal(0., eps, size=self.a.shape))
        b = self.b * np.exp(np.random.normal(0., eps, size=self.b.shape))
        return GammaDistribution(a, b)

    def copy(self):
        try:
            return GammaDistribution(self.a.copy(), self.b.copy())
        except: # not arrays
            return GammaDistribution(self.a, self.b)

class InverseGammaDistribution:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return 1. / np.random.gamma(self.a, 1. / self.b)

    def loglik(self, tau):
        return GammaDistribution(self.a, self.b).loglik(1. / tau) - 2 * np.log(tau)
    
    
class MultinomialDistribution:
    def __init__(self, log_p):
        # take log_p rather than p as an argument because of underflow
        self.log_p = log_p
        self.p = np.exp(log_p)
        self.p /= self.p.sum(-1)[..., nax]    # should already be normalized, but sometimes numerical error causes problems

    def expectation(self):
        return self.p

    def sample(self):
        #return np.random.multinomial(1, self.p)
        shape = self.p.shape[:-1]
        pr = int(np.prod(shape))
        p = self.p.reshape((pr, self.p.shape[-1]))
        temp = np.array([np.random.multinomial(1, p[i, :])
                         for i in range(pr)])
        return temp.reshape(shape + (self.p.shape[-1],))

    def loglik(self, a):
        a = np.array(a)
        if not np.issubdtype(a.dtype, int):
            raise RuntimeError('a must be an integer array')
        if np.shape(a) != np.shape(self.p)[:a.ndim]:
            raise RuntimeError('sizes do not match')
        
        if a.ndim == self.p.ndim:
            if not (np.all((a == 0) + (a == 1)) and a.sum(-1) == 1):
                raise RuntimeError('a must be 1-of-n')
            return np.sum(a * self.log_p)
        elif a.ndim == self.p.ndim - 1:
            shp = np.shape(self.log_p)[:-1]
            size = np.prod(shp).astype(int)
            log_p_ = self.log_p.reshape((size, np.shape(self.log_p)[-1]))
            a_ = a.ravel()
            result = log_p_[np.arange(size), a_]
            return result.reshape(shp)
        else:
            raise RuntimeError('sizes do not match')

    def __slice__(self, slc):
        return MultinomialDistribution(self.log_p[slc])

    @staticmethod
    def from_odds(odds):
        return MultinomialDistribution(odds - np.logaddexp.reduce(odds, axis=-1)[..., nax])

class BernoulliDistribution:
    def __init__(self, odds):
        self.odds = odds

    def _p(self):
        return 1. / (1 + np.exp(-self.odds))

    def expectation(self):
        return self._p()

    def variance(self):
        p = self._p()
        return p * (1. - p)

    def sample(self):
        return np.random.binomial(1, self._p())

    def loglik(self, a):
        if not np.issubdtype(a.dtype, int):
            raise RuntimeError('a must be an integer array')
        if not np.all((a==0) + (a==1)):
            raise RuntimeError('a must be a binary array')

        log_p = -np.logaddexp(0., -self.odds)
        log_1_minus_p = -np.logaddexp(0., self.odds)
        return a * log_p + (1-a) * log_1_minus_p

    @staticmethod
    def from_odds(odds):
        return BernoulliDistribution(odds)



class GaussianDistribution:
    def __init__(self, mu, sigma_sq):
        self.mu = mu
        self.sigma_sq = sigma_sq

    def loglik(self, x):
        return -0.5 * np.log(2*np.pi) + \
               -0.5 * np.log(self.sigma_sq) + \
               -0.5 * (x - self.mu) ** 2 / self.sigma_sq

    def sample(self):
        return np.random.normal(self.mu, self.sigma_sq)

    def maximize(self):
        return self.mu
