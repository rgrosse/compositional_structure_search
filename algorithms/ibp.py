import numpy as np
nax = np.newaxis
import scipy.special
import scipy.weave
import time
import traceback

import ibp_split_merge
import low_rank
import observations

from utils import distributions, gaussians, misc
fmi = gaussians.Potential.from_moments_iso


MAX_COLUMNS = 100

np.seterr(divide='ignore', invalid='ignore')

class IBPModel:
    def __init__(self, alpha, feature_var_prior, noise_var_prior):
        self.alpha = alpha
        self.feature_var_prior = feature_var_prior
        self.noise_var_prior = noise_var_prior

class CollapsedIBPState:
    def __init__(self, X, Z, sigma_sq_f, sigma_sq_n):
        self.X = X.copy()
        self.Z = Z.copy()
        self.sigma_sq_f = sigma_sq_f
        self.sigma_sq_n = sigma_sq_n

    def copy(self):
        return CollapsedIBPState(self.X.copy(), self.Z.copy(), self.sigma_sq_f, self.sigma_sq_n)

class FullIBPState:
    def __init__(self, X, Z, A, sigma_sq_f, sigma_sq_n):
        self.X = X.copy()
        self.Z = Z.copy()
        self.A = A.copy()
        self.sigma_sq_f = sigma_sq_f
        self.sigma_sq_n = sigma_sq_n

    def copy(self):
        return FullIBPState(self.X.copy(), self.Z.copy(), self.A.copy(), self.sigma_sq_f, self.sigma_sq_n)


class SimpleSigmaInfo:
    def __init__(self, Sigma, mu, z):
        # z: K
        # Sigma: K x K
        self.z = z.copy()
        self.Sigma = Sigma.copy()
        self.Sigma_z = np.dot(Sigma, z)
        self._mu = mu.copy()
        self.mu_z = np.dot(z, mu)

    def update(self, k, zk):
        if zk != self.z[k]:
            diff = zk - self.z[k]
            self.Sigma_z += diff * self.Sigma[:, k]
            self.mu_z += diff * self._mu[k, :]
            self.z[k] = zk

    def sigma_sq(self):
        return np.dot(self.z, self.Sigma_z)

    def sigma_sq_for(self, k, zk):
        diff = zk - self.z[k]
        return self.sigma_sq() + \
               2 * diff * self.Sigma_z[k] + \
               diff**2 * self.Sigma[k, k]

    def mu(self):
        return self.mu_z

    def mu_for(self, k, zk):
        diff = zk - self.z[k]
        if diff:
            return self.mu_z + diff * self._mu[k, :]
        else:
            return self.mu_z


class SimpleFeaturePosterior:
    def __init__(self, h, Lambda, mu, Sigma):
        self.h = h.copy()             # K x D
        self.Lambda = Lambda.copy()   # K x K
        self.mu = mu.copy()           # K x D
        self.Sigma = Sigma.copy()     # K x K
        self.K, self.D = self.h.shape

    def copy(self):
        return SimpleFeaturePosterior(self.h.copy(), self.Lambda.copy(), self.mu.copy(), self.Sigma.copy())

    def rank_one_update(self, a, r, dh):
        # a: scalar
        # r: D
        # dh: K x D

        self.h += dh

        self.Lambda += a * np.outer(r, r)

        w = np.dot(self.Sigma, r)
        coeff = a / (1 + a * np.dot(r, w))
        self.Sigma -= coeff * np.outer(w, w)

        self.mu = np.dot(self.Sigma, self.h)

    def add_dish(self, sigma_sq_f):
        """Add a new dish chosen by no customers. (The row currently being sampled is not included in the cache.
        When that row is added, it will include the new dish.)"""
        self.h = np.vstack([self.h, np.zeros((1, self.D))])
        self.mu = np.vstack([self.mu, np.zeros((1, self.D))])

        Lambda_new = np.zeros((self.K + 1, self.K + 1))
        Lambda_new[:self.K, :self.K] = self.Lambda
        Lambda_new[-1, -1] = 1. / sigma_sq_f
        self.Lambda = Lambda_new

        Sigma_new = np.zeros((self.K + 1, self.K + 1))
        Sigma_new[:self.K, :self.K] = self.Sigma
        Sigma_new[-1, -1] = sigma_sq_f
        self.Sigma = Sigma_new

        self.K += 1

    def predictive_mu(self, z):
        return np.dot(z, self.mu)

    def predictive_ssq(self, z):
        return np.dot(z, np.dot(self.Sigma, z))

    def sample(self):
        dist = gaussians.Distribution.from_moments_full(self.mu.T, self.Sigma[nax, :, :])
        return dist.sample().T

    def Sigma_info(self, z):
        return SimpleSigmaInfo(self.Sigma, self.mu, z)

    @staticmethod
    def from_information_form(h, Lambda):
        K, D = h.shape
        Sigma = np.linalg.inv(Lambda)
        mu = np.dot(Sigma, h)
        return SimpleFeaturePosterior(h, Lambda, mu, Sigma)

    def check(self):
        assert np.allclose(self.Sigma, np.linalg.inv(self.Lambda))
        assert np.allclose(self.mu, np.dot(self.Sigma, self.h))

    def check_close(self, other):
        assert np.allclose(self.h, other.h)
        assert np.allclose(self.Lambda, other.Lambda)
        assert np.allclose(self.mu, other.mu)
        assert np.allclose(self.Sigma, other.Sigma)
        assert self.K == other.K

class IBPCache:
    def __init__(self, model, X, mask, sigma_sq_f, sigma_sq_n, fpost, Z, counts, rows_included):
        self.model = model
        self.X = X.copy()
        self.mask = mask.copy()
        self.sigma_sq_n = sigma_sq_n
        self.sigma_sq_f = sigma_sq_f
        self.fpost = fpost
        self.Z = Z.copy()
        self.counts = counts.copy()
        self.rows_included = rows_included.copy()
        self.num_included = self.rows_included.sum()
        self.N, self.D = X.shape

    def add(self, i, z, x):
        assert not self.rows_included[i]
        self.rows_included[i] = True
        self.num_included += 1
        self.Z[i, :] = z
        self.counts += z

        #x, obs = self.data.X[i, :], self.data.obs[i, :]
        a = 1. / self.sigma_sq_n
        r = z
        dh = np.outer(z, x) / self.sigma_sq_n
        self.fpost.rank_one_update(a, r, dh)
        
    def remove(self, i):
        assert self.rows_included[i]
        self.rows_included[i] = False
        self.num_included -= 1
        self.counts -= self.Z[i, :]

        #x, obs, z = self.X[i, :], self.data.obs[i, :], self.Z[i, :]
        x, z = self.X[i, :], self.Z[i, :]
        a = -1. / self.sigma_sq_n
        r = z
        dh = -np.outer(z, x) / self.sigma_sq_n
        self.fpost.rank_one_update(a, r, dh)

    def add_dish(self):
        """Add a new dish chosen by no customers. (The row currently being sampled is not included in the cache.
        When that row is added, it will include the new dish.)"""
        self.fpost.add_dish(self.sigma_sq_f)
        self.Z = np.hstack([self.Z, np.zeros((self.N, 1))])
        self.counts = np.concatenate([self.counts, [0]])

    @staticmethod
    def from_state(model, data, state, rows_included):
        K, D = state.Z.shape[1], state.X.shape[1]
        
        h = np.zeros((K, D))
        Lambda = np.zeros((K, K, D))
        Z = state.Z[rows_included, :]
        X = state.X[rows_included, :]

        for j in range(D):
            x = X[:, j]
            h[:, j] = np.dot(Z.T, x) / state.sigma_sq_n

        Lambda = np.eye(K) / state.sigma_sq_f + \
                 np.dot(Z.T, Z) / state.sigma_sq_n
        fpost = SimpleFeaturePosterior.from_information_form(h, Lambda)

        counts = Z.sum(0)

        return IBPCache(model, state.X, data.mask, state.sigma_sq_f, state.sigma_sq_n, fpost, state.Z, counts, rows_included)

    def check_close(self, other):
        assert np.allclose(self.sigma_sq_n, other.sigma_sq_n)
        assert np.all(self.Z[self.rows_included, :] == other.Z[self.rows_included, :])
        assert np.all(self.counts == other.counts)
        assert np.all(self.rows_included == other.rows_included)
        assert np.allclose(self.X, other.X)
        assert np.all(self.mask == other.mask)
        self.fpost.check_close(other.fpost)

    def check(self, data, state):
        new_cache = IBPCache.from_state(self.model, data, state, self.rows_included)
        self.check_close(new_cache)


def first_customer(Z):
    """vector giving the first customer to try each dish"""
    N, K = Z.shape
    return (Z * np.arange(N, 0, -1)[:, nax]).argmax(0)

def ibp_loglik(Z, alpha):
    """Probability of an assignment matrix drawn from an IBP prior, where each customer chooses new dishes
    consecutively from the end, but otherwise the columns are unordered."""
    N, K = Z.shape
    total = K * np.log(alpha)
    new_dish_counts = np.bincount(first_customer(Z))
    total -= scipy.special.gammaln(new_dish_counts + 1).sum()
    total -= alpha * (1. / np.arange(1, N+1)).sum()

    m = Z.sum(0)
    total += scipy.special.gammaln(N - m + 1).sum()
    total += scipy.special.gammaln(m).sum()
    total -= K * scipy.special.gammaln(N+1)

    return total

def ibp_loglik_unordered(Z, alpha):
    """Probability of an assignment matrix drawn from an IBP prior, except that the columns are completely
    unordered. I.e. take the result of ibp_loglik, and divide by the number of matrices which map
    to the same canonical form."""
    N, K = Z.shape
    new_dish_counts = np.bincount(first_customer(Z))
    total = ibp_loglik(Z, alpha)
    total -= scipy.special.gammaln(K + 1)
    total += scipy.special.gammaln(new_dish_counts + 1).sum()
    return total
    

def gauss_loglik_vec(x, mu, ssq):
    dim = x.size
    return -0.5 * dim * np.log(2*np.pi) + \
           -0.5 * np.sum(np.log(ssq)) + \
           -0.5 * np.sum((x-mu)**2 / ssq)

def gauss_loglik_vec_C(x, mu, ssq):
    dim = int(x.size)
    logpi = float(np.log(2*np.pi))
    code = """
    int i;
    double diff;
    double ans = dim * logpi;
    for (i = 0; i < dim; i++){
        ans += log(ssq(i));
        diff = x(i) - mu(i);
        ans += diff * diff / ssq(i);
    }
    return_val = -0.5 * ans;
    """
    for i in range(5):
        try:
            ans = scipy.weave.inline(code, ['x', 'mu', 'ssq', 'logpi', 'dim'], type_converters=scipy.weave.converters.blitz)
            return ans
        except:
            traceback.print_exc()
            time.sleep(5)
    raise RuntimeError('Error in weave')

def gauss_loglik_vec_C2(x, mu, ssq):
    ssq = float(ssq)   # Weave complains if it's type np.float64
    dim = int(x.size)
    logpi = float(np.log(2*np.pi))
    code = """
    int i;
    double diff;
    double ans = dim * logpi;
    double log_ssq = log(ssq);
    for (i = 0; i < dim; i++){
        ans += log_ssq;
        diff = x(i) - mu(i);
        ans += diff * diff / ssq;
    }
    return_val = -0.5 * ans;
    """
    for i in range(5):
        try:
            ans = scipy.weave.inline(code, ['x', 'mu', 'ssq', 'logpi', 'dim'], type_converters=scipy.weave.converters.blitz)
            return ans
        except:
            traceback.print_exc()
            time.sleep(5)
    raise RuntimeError('Error in weave')

def evidence_collapsed_slow(model, data, state):
    N, K, D = state.Z.shape[0], state.Z.shape[1], state.X.shape[1]
    Sigma = state.sigma_sq_f * np.dot(state.Z, state.Z.T) + state.sigma_sq_n * np.eye(N)

    total = 0.
    for j in range(D):
        idxs = np.where(data.mask[:, j])[0]
        assert idxs.size > 0
        x = state.X[idxs, j]
        curr_Sigma = Sigma[idxs[:, nax], idxs[nax, :]]
        total += gaussians.Potential.from_moments_full(np.zeros(idxs.size), curr_Sigma).score(x)

    return total

def feature_loglik(model, data, state):
    N, K, D = state.Z.shape[0], state.Z.shape[1], state.X.shape[1]
    return fmi(np.zeros(K*D), state.sigma_sq_f).score(state.A.ravel())

def evidence_uncollapsed(model, data, state):
    pred = np.dot(state.Z, state.A)
    idxs = np.where(data.mask)
    return fmi(pred[idxs], state.sigma_sq_n).score(state.X[idxs])

def variance_loglik(model, data, state):
    return model.feature_var_prior.loglik(state.sigma_sq_f) + \
           model.noise_var_prior.loglik(state.sigma_sq_n)

def p_tilde_collapsed(model, data, state):
    return ibp_loglik_unordered(state.Z, model.alpha) + \
           evidence_collapsed_slow(model, data, state) + \
           variance_loglik(model, data, state)

def p_tilde_uncollapsed(model, data, state):
    return ibp_loglik_unordered(state.Z, model.alpha) + \
           feature_loglik(model, data, state) + \
           evidence_uncollapsed(model, data, state) + \
           variance_loglik(model, data, state)

def cond_assignment_collapsed(model, data, state, cache, Sigma_info, i, k):
    assert not cache.rows_included[i]

    evidence = np.zeros(2)
    for assignment in [0, 1]:
        mu = Sigma_info.mu_for(k, assignment)
        ssq = Sigma_info.sigma_sq_for(k, assignment) + state.sigma_sq_n
        evidence[assignment] = data[i, :].loglik(mu, ssq)
    data_odds = evidence[1] - evidence[0]

    prior_odds = np.log(cache.counts[k]) - np.log(cache.num_included - cache.counts[k] + 1)

    return distributions.BernoulliDistribution.from_odds(data_odds + prior_odds)

def new_dish_evidence(model, data, state, cache, i):
    assert not cache.rows_included[i]

    z = state.Z[i, :].copy()
    mu = cache.fpost.predictive_mu(z)
    ssq_off = cache.fpost.predictive_ssq(z) + state.sigma_sq_n
    ssq_on = ssq_off + state.sigma_sq_f

    return data[i, :].loglik(mu, ssq_on) - data[i, :].loglik(mu, ssq_off)

def poisson_conditional_prob(k, lam):
    """P(x >= k | x >= k-1) under a Poisson distribution with parameter lam."""
    if k == 0:
        return 1.
    temp = np.arange(k)
    p = np.exp(-lam + temp * np.log(lam) - scipy.special.gammaln(temp+1))
    prob_x_geq_km1 = 1. - p[:k-1].sum()
    prob_x_geq_k = 1. - p.sum()
    return prob_x_geq_k / prob_x_geq_km1

def log_poisson_conditional_odds(k, lam):
    if k == 0:
        return 0.
    ki = np.arange(k-1, k+100)
    log_poiss = -lam + ki * np.log(lam) - scipy.special.gammaln(ki+1)
    return np.logaddexp.reduce(log_poiss[1:]) - log_poiss[0]
    

def sample_new_dishes(model, data, state, cache, i):
    new_dish_num = 1
    lam = model.alpha / (cache.num_included + 1)
    assert isinstance(state, CollapsedIBPState)  # TODO: for uncollapsed sampler, need to sample new features
    N = state.X.shape[0]

    while True:
        if state.Z.shape[1] > MAX_COLUMNS:
            break
        
        prior_odds = log_poisson_conditional_odds(new_dish_num, lam)
        data_odds = new_dish_evidence(model, data, state, cache, i)
        dist = distributions.BernoulliDistribution.from_odds(prior_odds + data_odds)
        result = dist.sample()

        if not result:
            break

        new_dish_num += 1
        state.Z = np.hstack([state.Z, np.zeros((N, 1))])
        state.Z[i, -1] = 1
        cache.add_dish()


def cond_sigma_sq_f(model, data, state):
    a = model.noise_var_prior.a + 0.5 * state.A.size
    b = model.noise_var_prior.b + 0.5 * np.sum(state.A**2)
    return distributions.InverseGammaDistribution(a, b)

def cond_sigma_sq_n(model, data, state):
    diff = state.X - np.dot(state.Z, state.A)
    a = model.feature_var_prior.a + 0.5 * np.sum(data.mask)
    b = model.feature_var_prior.b + 0.5 * np.sum(data.mask * diff**2)
    return distributions.InverseGammaDistribution(a, b)

def squeeze(state):
    # eliminate unused dishes
    counts = state.Z.sum(0)
    nz = np.where(counts > 0)[0]
    state.Z = state.Z[:, nz]
    state.A = state.A[nz, :]

    # put in canonical form
    idxs = first_customer(state.Z).argsort()
    state.Z = state.Z[:, idxs]
    state.A = state.A[idxs, :]

def fill_in_X(model, data, state, cache, i):
    z = state.Z[i, :]
    mu = np.dot(z, cache.fpost.mu)
    sigma_sq = np.dot(z, np.dot(cache.fpost.Sigma, z)) + state.sigma_sq_n
    state.X[i, :] = data[i, :].sample_latent_values(mu, sigma_sq)


def gibbs_sweep(model, data, state, split_merge, fit_hyper=True, fixed_variance=False):
    # sample assignments
    N, D = state.X.shape
    cache = IBPCache.from_state(model, data, state, np.ones(N, dtype=bool))

    for i in range(N):
        K = state.Z.shape[1]
        cache.remove(i)
        Sigma_info = cache.fpost.Sigma_info(state.Z[i, :])
        for k in range(K):
            cond = cond_assignment_collapsed(model, data, state, cache, Sigma_info, i, k)
            state.Z[i, k] = cond.sample()
            Sigma_info.update(k, state.Z[i, k])
        sample_new_dishes(model, data, state, cache, i)
        fill_in_X(model, data, state, cache, i)
        cache.add(i, state.Z[i, :], state.X[i, :])


    if not np.any(state.Z.sum(0) > 0): # things crash if Z is degenerate
        state.Z[0, 0] = 1

    # sample features
    state.A = cache.fpost.sample()

    del cache
    squeeze(state)

    if split_merge:
        for i in range(5):
            ibp_split_merge.split_merge_step(model, data, state)

    if not np.any(state.Z.sum(0) > 0): # things crash if Z is degenerate
        state.Z[0, 0] = 1

    # eliminate unused dishes, put in canonical form
    squeeze(state)

    # sample hyperparameters
    if fit_hyper:
        cond = cond_sigma_sq_f(model, data, state)
        state.sigma_sq_f = cond.sample()
        if not fixed_variance:
            cond = cond_sigma_sq_n(model, data, state)
            state.sigma_sq_n = cond.sample()



def init_X(data):
    svd_K = min(20, data.shape[0] // 4, data.shape[1] // 4)
    svd_K = max(svd_K, 2)   # 1 and 0 cause it to crash
    dummy = observations.DataMatrix(data)
    _, _, _, _, _, X = low_rank.fit_model(dummy, svd_K, 10)
    return X

def zero_init(model, data, fixed_variance=False):
    X = init_X(data)
    Z_init = np.zeros((X.shape[0], 1), dtype=int)
    Z_init[0, 0] = 1
    sigma_sq_f = np.mean(X[data.mask] ** 2)
    if fixed_variance:
        sigma_sq_n = 1.
    else:
        sigma_sq_n = sigma_sq_f / 3.
    return CollapsedIBPState(X, Z_init, sigma_sq_f, sigma_sq_n)

def sequential_init(model, data, fixed_variance=False):
    state = zero_init(model, data, fixed_variance)
    N, D = state.X.shape
    cache = IBPCache.from_state(model, data, state, np.zeros(N, dtype=bool))
    for i in range(N):
        K = state.Z.shape[1]
        Sigma_info = cache.fpost.Sigma_info(state.Z[i, :])
        for k in range(K):
            cond = cond_assignment_collapsed(model, data, state, cache, Sigma_info, i, k)
            state.Z[i, k] = cond.sample()
            Sigma_info.update(k, state.Z[i, k])
        sample_new_dishes(model, data, state, cache, i)
        cache.add(i, state.Z[i, :], state.X[i, :])   # for the moment, use the initial values rather than resampling

    return state

            


NUM_ITER = 200

TIME_LIMIT = 900.  # 15 min

def fit_model(data_matrix, num_iter=NUM_ITER):
    model = IBPModel(1., distributions.InverseGammaDistribution(1., 1.), distributions.InverseGammaDistribution(1., 1.))
    fixed_variance = data_matrix.fixed_variance()
    data = data_matrix.observations
    state = sequential_init(model, data, fixed_variance)

    pbar = misc.pbar(num_iter)

    t0 = time.time()
    for it in range(num_iter):
        gibbs_sweep(model, data, state, True, True, fixed_variance)

        pred = np.dot(state.Z, state.A)
        state.X = data.sample_latent_values(pred, state.sigma_sq_n)
        
        if time.time() - t0 > TIME_LIMIT:
            break

        pbar.update(it)
    pbar.finish()

    return state
