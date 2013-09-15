import numpy as np
nax = np.newaxis
import sklearn.cluster
import scipy.special
import time

import low_rank
from utils import distributions, gaussians, psd_matrices, misc
from_iso = gaussians.Potential.from_moments_iso


np.seterr(divide='ignore', invalid='ignore')

MAX_COMPONENTS = 100

class CRPModel:
    def __init__(self, alpha, ndim, within_var_prior, between_var_prior, isotropic_w, isotropic_b):
        self.alpha = alpha
        self.ndim = ndim
        self.within_var_prior = within_var_prior
        self.between_var_prior = between_var_prior
        self.isotropic_w = isotropic_w
        self.isotropic_b = isotropic_b


class CollapsedCRPState:
    def __init__(self, X, assignments, sigma_sq_w, sigma_sq_b):
        self.X = X.copy()
        self.assignments = assignments
        self.sigma_sq_w = sigma_sq_w
        self.sigma_sq_b = sigma_sq_b

class FullCRPState:
    def __init__(self, X, assignments, centers, sigma_sq_w, sigma_sq_b):
        self.X = X
        self.assignments = assignments
        self.centers = centers
        self.sigma_sq_w = sigma_sq_w
        self.sigma_sq_b = sigma_sq_b

    def copy(self):
        if np.isscalar(self.sigma_sq_w):
            sigma_sq_w = self.sigma_sq_w
        else:
            sigma_sq_w = self.sigma_sq_w.copy()
        if np.isscalar(self.sigma_sq_b):
            sigma_sq_b = self.sigma_sq_b
        else:
            sigma_sq_b = self.sigma_sq_b.copy()
        return FullCRPState(self.X.copy(), self.assignments.copy(), self.centers.copy(), sigma_sq_w, sigma_sq_b)


    


class CollapsedCRPCache:
    def __init__(self, model, X, mask, assignments, counts, obs_counts, sum_X, sum_X_sq):
        self.model = model
        self.X = X.copy()
        self.mask = mask.copy()
        self.assignments = assignments
        self.ncomp = assignments.max() + 1
        self.counts = counts
        self.obs_counts = obs_counts
        self.sum_X = sum_X
        self.sum_X_sq = sum_X_sq

    def copy(self):
        return CollapsedCRPCache(self.model, self.X, self.mask, self.assignments.copy(), self.counts.copy(),
                                 self.obs_counts.copy(), self.sum_X.copy(), self.sum_X_sq.copy())

    def add(self, i, k, x):
        assert self.assignments[i] == -1
        if k == self.ncomp:
            self.counts = np.concatenate([self.counts, [0]])
            self.obs_counts = np.vstack([self.obs_counts, np.zeros(self.model.ndim, dtype=int)])
            self.sum_X = np.vstack([self.sum_X, np.zeros((1, self.model.ndim))])
            self.sum_X_sq = np.vstack([self.sum_X_sq, np.zeros((1, self.model.ndim))])
            self.ncomp += 1
        self.counts[k] += 1
        self.obs_counts[k, :] += self.mask[i, :]
        self.sum_X[k, :] += self.mask[i, :] * x
        self.sum_X_sq[k, :] += self.mask[i, :] * x ** 2
        self.assignments[i] = k
        self.X[i, :] = x

    def remove(self, i):
        assert self.assignments[i] != -1
        k = self.assignments[i]
        self.counts[k] -= 1
        self.obs_counts[k, :] -= self.mask[i, :]
        self.sum_X[k, :] -= self.mask[i, :] * self.X[i, :]
        self.sum_X_sq[k, :] -= self.mask[i, :] * self.X[i, :] ** 2
        self.assignments[i] = -1

    def replace(self, i, k):
        self.remove(i)
        self.add(i, k, self.X[i, :])

    def squeeze(self, state):
        # renumber the clusters to eliminate empty ones
        for i in range(self.ncomp)[::-1]:
            if self.counts[i] == 0:
                assert np.all(state.assignments == self.assignments)
                self.assignments = np.where(self.assignments > i, self.assignments - 1, self.assignments)
                state.assignments = self.assignments.copy()
                self.ncomp -= 1
                self.counts = np.concatenate([self.counts[:i], self.counts[i+1:]])
                self.obs_counts = np.vstack([self.obs_counts[:i, :], self.obs_counts[i+1:, :]])
                self.sum_X = np.vstack([self.sum_X[:i, :], self.sum_X[i+1:, :]])
                self.sum_X_sq = np.vstack([self.sum_X_sq[:i, :], self.sum_X_sq[i+1:, :]])
                

    def check(self, data, state):
        new_cache = CollapsedCRPCache.from_state(self.model, data, state)
        self.check_close(new_cache)
        assert np.all(self.counts > 0)

    def check_close(self, other):
        assert np.all(self.counts == other.counts)
        assert np.all(self.obs_counts == other.obs_counts)
        assert np.allclose(self.sum_X, other.sum_X)
        assert np.allclose(self.sum_X_sq, other.sum_X_sq)
        assert np.all(self.assignments == other.assignments)

    @staticmethod
    def from_state(model, data, state):
        assignments = state.assignments.copy()
        ncomp = assignments.max() + 1
        counts = misc.get_counts(state.assignments, ncomp)
        obs_counts = np.zeros((ncomp, model.ndim), dtype=int)
        sum_X = np.zeros((ncomp, model.ndim))
        sum_X_sq = np.zeros((ncomp, model.ndim))
        for k in range(ncomp):
            obs_counts[k, :] = data.mask[assignments==k, :].sum(0)
            sum_X[k, :] = (data.mask * state.X)[assignments==k, :].sum(0)
            sum_X_sq[k, :] = (data.mask * state.X**2)[assignments==k, :].sum(0)
        return CollapsedCRPCache(model, state.X, data.mask, assignments, counts, obs_counts, sum_X, sum_X_sq)


def crp_loglik(assignments, alpha):
    counts = np.bincount(assignments)
    N = counts.sum()
    K = counts.size
    return scipy.special.gammaln(alpha) + \
           -scipy.special.gammaln(alpha + N) + \
           K * np.log(alpha) + \
           scipy.special.gammaln(counts).sum()


def p_tilde_collapsed(model, data, state):
    cache = CollapsedCRPCache.from_state(model, data, state)
    ncomp = cache.counts.size
    total = 0.
    
    # data evidence, marginalizing out the centers
    ce = center_evidence(model, state, cache)
    for k in range(ncomp):
        if model.isotropic_b:
            prior_term = from_iso(np.zeros(model.ndim), state.sigma_sq_b)
        else:
            prior_term = gaussians.Potential.from_moments_diag(np.zeros(model.ndim), state.sigma_sq_b)
        evidence = ce[k]
        total += (prior_term + evidence).integral()

    # hyperparameters
    total += np.sum(model.within_var_prior.loglik(state.sigma_sq_w))
    total += np.sum(model.between_var_prior.loglik(state.sigma_sq_b))

    # partition
    total += crp_loglik(state.assignments, model.alpha)

    return total

def p_tilde(model, data, state):
    total = 0.

    # data likelihood
    evidence = p_X_given_centers(model, data, state)
    total += evidence.score(state.centers[state.assignments, :]).sum()

    # centers likelihood
    if model.isotropic_b:
        centers_dist = from_iso(np.zeros(model.ndim), state.sigma_sq_b)
    else:
        centers_dist = gaussians.Potential.from_moments_diag(np.zeros(model.ndim), state.sigma_sq_b)
    total += centers_dist[nax].score(state.centers).sum()

    # hyperparameters
    total += np.sum(model.within_var_prior.loglik(state.sigma_sq_w))
    total += np.sum(model.between_var_prior.loglik(state.sigma_sq_b))

    # partition
    total += crp_loglik(state.assignments, model.alpha)

    return total

def p_X_given_centers(model, data, state):
    lam = data.mask / state.sigma_sq_w
    h = lam * state.X
    temp = -0.5 * np.log(2*np.pi) + \
           -0.5 * np.log(state.sigma_sq_w) + \
           -0.5 * lam * state.X**2
    Z = (data.mask * temp).sum(1)
    return gaussians.Potential(h, psd_matrices.DiagonalMatrix(lam), Z)
    
    

def center_evidence(model, state, cache):
    lam = cache.obs_counts / state.sigma_sq_w
    mu = np.where(cache.obs_counts > 0, cache.sum_X / cache.obs_counts, 0.)
    h = mu * lam
    if model.isotropic_w:
        Z = -0.5 * cache.obs_counts.sum(1) * np.log(2*np.pi) + \
            -0.5 * cache.obs_counts.sum(1) * np.log(state.sigma_sq_w) + \
            -0.5 * cache.sum_X_sq.sum(1) / state.sigma_sq_w
    else:
        Z = -0.5 * cache.obs_counts.sum(1) * np.log(2 * np.pi) + \
            -0.5 * (cache.obs_counts * np.log(state.sigma_sq_w)).sum(1) + \
            -0.5 * (cache.sum_X_sq / state.sigma_sq_w).sum(1)
    return gaussians.Potential(h, psd_matrices.DiagonalMatrix(lam), Z)

def center_beliefs(model, state, cache):
    if model.isotropic_b:
        prior_term = from_iso(np.zeros(model.ndim), state.sigma_sq_b)
    else:
        prior_term = gaussians.Potential.from_moments_diag(np.zeros(model.ndim), state.sigma_sq_b)
    return (center_evidence(model, state, cache) + prior_term).renorm()

def new_center_beliefs(model, state):
    return from_iso(np.zeros(model.ndim), state.sigma_sq_b)

def center_predictive(model, state, cache, k):
    N, D = state.X.shape
    if k == cache.ncomp:
        return np.zeros(D), np.ones(D) * (state.sigma_sq_b + state.sigma_sq_w)
    else:
        ssq_w, ssq_b = state.sigma_sq_w, state.sigma_sq_b
        lam = cache.obs_counts[k, :] / ssq_w + 1. / ssq_b
        predictive_mu = (cache.sum_X[k, :] / ssq_w) / lam
        predictive_sigma_sq = 1. / lam + state.sigma_sq_w
        return predictive_mu, predictive_sigma_sq

def cond_assignments_collapsed(model, data, state, cache, i):
    cache.remove(i)
    
    prior_term = np.concatenate([np.log(cache.counts), [np.log(model.alpha)]])

    if MAX_COMPONENTS is not None and cache.ncomp >= MAX_COMPONENTS:
        prior_term[-1] = -np.infty

    data_term = np.zeros(cache.ncomp + 1)
    for k in range(cache.ncomp + 1):
        predictive_mu, predictive_ssq = center_predictive(model, state, cache, k)
        data_term[k] = data[i, :].loglik(predictive_mu, predictive_ssq)
        
    cache.add(i, state.assignments[i], state.X[i, :])

    return distributions.MultinomialDistribution.from_odds(prior_term + data_term)


def gibbs_step_assignments_collapsed(model, data, state, cache, i):
    dist = cond_assignments_collapsed(model, data, state, cache, i)
    new_assignment = dist.sample().argmax()
    state.assignments[i] = new_assignment
    cache.remove(i)
    predictive_mu, predictive_ssq = center_predictive(model, state, cache, new_assignment)
    state.X[i, :] = data[i, :].sample_latent_values(predictive_mu, predictive_ssq)
    cache.add(i, new_assignment, state.X[i, :])
    cache.squeeze(state)


def cond_centers(model, data, state, cache):
    if model.isotropic_b:
        prior_term = from_iso(np.zeros(model.ndim), state.sigma_sq_b)
    else:
        prior_term = gaussians.Potential.from_moments_diag(np.zeros(model.ndim), state.sigma_sq_b)
    center_beliefs = center_evidence(model, state, cache) + prior_term
    return center_beliefs.renorm()

def gibbs_step_centers(model, data, state, cache):
    cond = cond_centers(model, data, state, cache)
    new_centers = cond.to_distribution().sample()
    state.centers = new_centers

def cond_sigma_sq_b(model, data, state):
    counts = np.bincount(state.assignments)
    nz = np.where(counts > 0)[0]
    centers = state.centers[nz, :]

    if model.isotropic_b:
        a = model.between_var_prior.a + 0.5 * nz.size * model.ndim
        b = model.between_var_prior.b + 0.5 * np.sum(centers**2)
    else:
        a = model.between_var_prior.a + 0.5 * nz.size * np.ones(model.ndim)
        b = model.between_var_prior.b + 0.5 * np.sum(centers**2, axis=0)
    return distributions.InverseGammaDistribution(a, b)

def gibbs_step_sigma_sq_b(model, data, state):
    cond = cond_sigma_sq_b(model, data, state)
    state.sigma_sq_b = cond.sample()

def cond_sigma_sq_w(model, data, state):
    diff = state.X - state.centers[state.assignments, :]
    if model.isotropic_w:
        a = model.within_var_prior.a + 0.5 * np.sum(data.mask)
        b = model.within_var_prior.b + 0.5 * np.sum(data.mask * diff**2)
    else:
        a = model.within_var_prior.a + 0.5 * np.sum(data.mask, axis=0)
        b = model.within_var_prior.b + 0.5 * np.sum(data.mask * diff**2, axis=0)
    return distributions.InverseGammaDistribution(a, b)

def gibbs_step_sigma_sq_w(model, data, state):
    cond = cond_sigma_sq_w(model, data, state)
    state.sigma_sq_w = cond.sample()

def gibbs_sweep_collapsed(model, data, state, fixed_variance=False):
    cache = CollapsedCRPCache.from_state(model, data, state)
    num = state.X.shape[0]
    for i in range(num):
        gibbs_step_assignments_collapsed(model, data, state, cache, i)
        
    cache = CollapsedCRPCache.from_state(model, data, state)
    gibbs_step_centers(model, data, state, cache)
    #assert False
    gibbs_step_sigma_sq_b(model, data, state)
    if not fixed_variance:
        gibbs_step_sigma_sq_w(model, data, state)
    
    


NUM_ITER = 200

def init_X(data_matrix):
    X_init = data_matrix.sample_latent_values(np.zeros((data_matrix.m, data_matrix.n)), 1.)
    svd_K = min(20, data_matrix.m // 4, data_matrix.n // 4)
    svd_K = max(svd_K, 2)  # 0 and 1 cause it to crash
    _, _, _, _, _, X_init = low_rank.fit_model(data_matrix, svd_K, 10)
    return X_init


def fit_model(data_matrix, isotropic_w=True, isotropic_b=True, num_iter=NUM_ITER):
    X_init = init_X(data_matrix)

    model = CRPModel(1., X_init.shape[1], distributions.InverseGammaDistribution(0.01, 0.01),
                     distributions.InverseGammaDistribution(0.01, 0.01), isotropic_w, isotropic_b)

    N, D = X_init.shape

    k_init = min(N//4, 40)
    km = sklearn.cluster.KMeans(n_clusters=k_init)
    km.fit(X_init)
    init_assignments = km.labels_

    
    
    sigma_sq_f = sigma_sq_n = X_init.var() / 2.
    if not model.isotropic_b:
        sigma_sq_f = X_init.var(0) / 2.
    state = CollapsedCRPState(X_init, init_assignments, sigma_sq_n, sigma_sq_f)
    state.centers = km.cluster_centers_

    fixed_variance = data_matrix.fixed_variance()

    data = data_matrix.observations

    if fixed_variance:
        if isotropic_w:
            state.sigma_sq_w = 1.
        else:
            state.sigma_sq_w = np.ones(D)

    pbar = misc.pbar(num_iter)

    t0 = time.time()
    for it in range(num_iter):
        pred = state.centers[state.assignments, :]
        state.X = data_matrix.sample_latent_values(pred, state.sigma_sq_w)
        gibbs_sweep_collapsed(model, data, state, fixed_variance)

        if time.time() - t0 > 3600.:   # 1 hour
            break

        pbar.update(it)
    pbar.finish()

    # sample the centers
    cache = CollapsedCRPCache.from_state(model, data, state)
    gibbs_step_centers(model, data, state, cache)

    return state


