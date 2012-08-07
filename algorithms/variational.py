import numpy as np
nax = np.newaxis
import random
Random = random.Random()
import scipy.linalg, scipy.stats

from utils import misc



def perturb_simplex(q, eps=1e-5):
    eps = 1e-5
    k = q.size
    q = q.copy()
    for tr in range(10):
        large_inds = np.where(q > eps)[0]
        i = Random.choice(large_inds)
        j = np.random.randint(0, k)
        if i == j or q[j] > 1-eps:
            continue
        q[i] -= eps
        q[j] += eps
    return q

def perturb_psd(S, eps=1e-5):
    d, V = scipy.linalg.eigh(S)
    d *= np.exp(np.random.normal(0., eps, size=d.shape))
    return np.dot(np.dot(V, np.diag(d)), V.T)

def perturb_pos(x, eps=1e-5):
    return x * np.exp(np.random.normal(0., eps, size=x.shape))

    

ALPHA = 1.
class MultinomialEstimator:
    def __init__(self, pi, A):
        self.pi = pi
        self.nclass = pi.size
        self.A = A

    def expected_log_prob(self, rep):
        return np.dot(rep.q, np.log(self.pi))

    def fit_representation(self, t, Sigma_N, init=None):
        data_term = np.zeros(self.nclass)
        Lambda_N = np.linalg.inv(Sigma_N)
        for i in range(self.nclass):
            diff = t - self.A[i,:]
            #data_term[i] = -0.5 * np.sum(diff**2 / sigma_sq_N)
            data_term[i] = -0.5 * np.dot(np.dot(diff, Lambda_N), diff)
        log_q = np.log(self.pi) + data_term
        log_q -= np.logaddexp.reduce(log_q)
        q = np.exp(log_q)
        return MultinomialRepresentation(q)

    def init_representation(self):
        return MultinomialRepresentation(self.pi)

    @staticmethod
    def random(k, n):
        pi = np.random.uniform(0., 1., size=k)
        pi /= pi.sum()
        A = np.random.normal(size=(k, n))
        return MultinomialEstimator(pi, A)

    @staticmethod
    def random_u(k):
        u = np.random.uniform(0., 1., size=k)
        return u / u.sum()

class MultinomialRepresentation:
    def __init__(self, q):
        self.q = q
        assert np.allclose(np.sum(self.q), 1.)

    def expected_value(self):
        return self.q

    def covariance(self):
        return np.diag(self.q) - np.outer(self.q, self.q)

    def entropy(self):
        return scipy.stats.distributions.entropy(self.q)

    def sample(self):
        return np.random.multinomial(1, self.q)

    def perturb(self, eps):
        return MultinomialRepresentation(perturb_simplex(self.q, eps))



class BernoulliEstimator:
    def __init__(self, pi, A):
        self.pi = pi
        self.A = A
        self.nclass = self.pi.size

    def expected_log_prob(self, rep):
        return np.dot(rep.q, np.log(self.pi)) + np.dot(1-rep.q, np.log(1-self.pi))

    def fit_representation(self, t, Sigma_N, init=None):
        Lambda_N = np.linalg.inv(Sigma_N)
        J = -np.log(self.pi) + np.log(1. - self.pi) - np.dot(self.A, np.dot(Lambda_N, t))
        Lambda = np.dot(np.dot(self.A, Lambda_N), self.A.T)
        return BernoulliRepresentation(misc.mean_field(J, Lambda, init.q))

    def init_representation(self):
        return BernoulliRepresentation(self.pi)

    @staticmethod
    def random(k, n):
        pi = np.random.uniform(0., 1., size=k)
        A = np.random.normal(size=(k, n))
        return BernoulliEstimator(pi, A)

    @staticmethod
    def random_u(k):
        return np.random.uniform(0., 1., size=k)

class BernoulliRepresentation:
    def __init__(self, q):
        self.q = q
        
    def expected_value(self):
        return self.q

    def covariance(self):
        return np.diag(self.q * (1. - self.q))

    def entropy(self):
        #return misc.bernoulli_entropy(self.q) * np.log(2)
        return np.sum([scipy.stats.distributions.entropy([p, 1.-p]) for p in self.q])

    def sample(self):
        return np.random.binomial(1, self.q)

    def perturb(self, eps):
        q = np.clip(np.random.normal(self.q, eps), 0., 1.)
        return BernoulliRepresentation(q)



class VariationalProblem:
    def __init__(self, estimators, x, Sigma_N):
        self.estimators = estimators
        self.x = x
        self.nterms = len(estimators)
        self.nfea = self.x.size
        self.Sigma_N = Sigma_N
        assert Sigma_N.shape == (x.size, x.size)
        
    def objective_function(self, reps, collapse_z=False):
        assert len(reps) == self.nterms

        fobj = 0.
        m = np.zeros(self.nfea)
        S = np.zeros((self.nfea, self.nfea))
        for estimator, rep in zip(self.estimators, reps):
            # E[log P(u|U)]
            fobj += estimator.expected_log_prob(rep)
            
            # H(q)
            fobj += rep.entropy()
            
            # sufficient statistics
            m += np.dot(estimator.A.T, rep.expected_value())
            S += misc.mult([estimator.A.T, rep.covariance(), estimator.A])

        Lambda_N = np.linalg.inv(self.Sigma_N)

        fobj += -0.5 * self.nfea * np.log(2*np.pi) - 0.5 * misc.logdet(self.Sigma_N)
        diff = self.x - m
        fobj += -0.5 * np.dot(np.dot(diff, Lambda_N), diff)
        fobj += -0.5 * np.sum(S * Lambda_N)

        return fobj

    def update_one(self, reps, i):
        reps = reps[:] # make copy
        m = np.zeros(self.nfea)
        for j, estimator in enumerate(self.estimators):
            if i == j:
                continue
            m += np.dot(estimator.A.T, reps[j].expected_value())

        t = self.x - m
        reps[i] = self.estimators[i].fit_representation(t, self.Sigma_N, reps[i])
        return reps

    def update_all(self, reps):
        for i in range(self.nterms):
            reps = self.update_one(reps, i)
        return reps

    def solve(self):
        if len(self.estimators) <= 1:
            NUM_ITER = 1
        else:
            NUM_ITER = 10
        reps = [estimator.init_representation() for estimator in self.estimators]
        for it in range(NUM_ITER):
            reps = self.update_all(reps)
        return reps

