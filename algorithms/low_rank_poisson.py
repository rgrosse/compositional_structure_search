import itertools
import numpy as np
nax = np.newaxis
import random
import scipy.integrate
import scipy.linalg
import scipy.special
import time

from utils import distributions, gaussians, misc, psd_matrices

A = 0.1
B = 0.1

VERBOSE = False
SEED_0 = False
K_INIT = 2

class State:
    def __init__(self, U, V, ssq_U, ssq_N):
        self.U = U
        self.V = V
        self.ssq_U = ssq_U
        self.ssq_N = ssq_N

    def copy(self):
        return State(self.U.copy(), self.V.copy(), self.ssq_U.copy(), self.ssq_N)

def sample_variance(values, axis, mask=None):
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    a = 0.01 + 0.5 * mask.sum(axis)
    b = 0.01 + 0.5 * (mask * values ** 2).sum(axis)
    prec = np.random.gamma(a, 1. / b)
    return 1. / prec

def p_u(u):
    N = u.size
    return -(A + 0.5 * N) * np.log(B + 0.5 * np.sum(u ** 2))

def givens_move(U, V, a, b):
    N = U.shape[0]
    theta = np.linspace(-np.pi / 4., np.pi / 4.)
    uaa = np.dot(U[:, a], U[:, a])
    uab = np.dot(U[:, a], U[:, b])
    ubb = np.dot(U[:, b], U[:, b])

    sin, cos = np.sin(theta), np.cos(theta)
    uaa_prime_ssq = uaa * cos ** 2 + 2 * uab * cos * sin + ubb * sin ** 2
    ubb_prime_ssq = uaa * sin ** 2 - 2 * uab * cos * sin + ubb * cos ** 2
    odds = -(A + 0.5 * N) * (np.log(B + 0.5 * uaa_prime_ssq) + np.log(B + 0.5 * ubb_prime_ssq))
    p = np.exp(odds - np.logaddexp.reduce(odds))
    p /= np.sum(p)
    idx = np.random.multinomial(1, p).argmax()
    
    theta = theta[idx]
    sin, cos = np.sin(theta), np.cos(theta)
    U[:, a], U[:, b] = cos * U[:, a] + sin * U[:, b], -sin * U[:, a] + cos * U[:, b]
    V[a, :], V[b, :] = cos * V[a, :] + sin * V[b, :], -sin * V[a, :] + cos * V[b, :]
    
def givens_moves(state):
    U, V = state.U, state.V
    N, K, D = U.shape[0], U.shape[1], V.shape[1]
    pairs = list(itertools.combinations(range(K), 2))
    if not SEED_0:
        random.shuffle(pairs)
    for a, b in pairs:
        givens_move(U, V, a, b)
    state.ssq_U = sample_variance(U, 0)

def scaling_move(U, V, a):
    alpha_pts = np.logspace(-2., 2., 100)
    odds = np.zeros(len(alpha_pts))
    for i, alpha in enumerate(alpha_pts):
        odds[i] = p_u(alpha * U[:, a]) + distributions.gauss_loglik(V[a, :] / alpha, 0., 1.).sum()
    p = np.exp(odds - np.logaddexp.reduce(odds))
    p /= np.sum(p)
    idx = np.random.multinomial(1, p).argmax()
    alpha = alpha_pts[idx]
    
    U[:, a] *= alpha
    V[a, :] /= alpha

def scaling_moves(state):
    U, V = state.U, state.V
    N, K, D = U.shape[0], U.shape[1], V.shape[1]
    for a in range(K):
        scaling_move(U, V, a)
    state.ssq_U = sample_variance(U, 0)


def cond_U(X, obs, V, ssq_U, ssq_N):
    N, K, D = X.shape[0], V.shape[0], X.shape[1]
    if np.all(obs):
        Lambda = np.diag(1. / ssq_U) + np.dot(V, V.T) / ssq_N
        Lambda = Lambda[nax, :, :]
    else:
        Lambda = np.zeros((N, K, K))
        for i in range(N):
            idxs = np.where(obs[i, :])[0]
            V_curr = V[:, idxs]
            Lambda[i, :, :] = np.diag(1. / ssq_U) + np.dot(V_curr, V_curr.T) / ssq_N
    h = np.dot(X * obs, V.T) / ssq_N
    return gaussians.Potential(h, psd_matrices.FullMatrix(Lambda), 0.)

def cond_Vt(X, obs, U, ssq_N):
    K = U.shape[1]
    return cond_U(X.T, obs.T, U.T, np.ones(K), ssq_N)

def sample_U_V(state, X, obs):
    state.U = cond_U(X, obs, state.V, state.ssq_U, state.ssq_N).to_distribution().sample()
    state.V = cond_Vt(X, obs, state.U, state.ssq_N).to_distribution().sample().T
    

class InstabilityError(Exception):
    pass

class ProposalInfo:
    def __init__(self, resid, obs, ssq_N):
        N, D = resid.shape
        self.resid = resid.copy()
        self.obs = obs.copy()
        self.ssq_N = ssq_N
        self.u = np.zeros(N)
        self.assigned = np.zeros(N, dtype=bool)
        self.lam = np.ones(D)   # N(0, 1) prior
        self.h = np.zeros(D)
        self.v = None
        self.ssq_u = None
        self.num_assigned = 0
        self.sum_u_sq = 0.

    def update_u(self, i, ui):
        assert not self.assigned[i]
        self.u[i] = ui
        idxs = np.where(self.obs[i, :])[0]
        self.lam[idxs] += ui ** 2 / self.ssq_N
        self.h[idxs] += ui * self.resid[i, idxs] / self.ssq_N
        self.assigned[i] = True
        self.num_assigned += 1
        self.sum_u_sq += ui ** 2

    def cond_v(self):
        return distributions.GaussianDistribution(self.h / self.lam, 1. / self.lam)

    def cond_ssq_u(self):
        a = A + 0.5 * self.num_assigned
        b = B + 0.5 * self.sum_u_sq
        return distributions.InverseGammaDistribution(a, b)

    def cond_u(self, i):
        idxs = np.where(self.obs[i, :])[0]
        #lam = np.dot(self.v[idxs], self.v[idxs]) / self.ssq_N + 1. / self.ssq_u
        v = self.v[idxs]
        lam = (v**2).sum() / self.ssq_N + 1. / self.ssq_u
        h = (self.resid[i, idxs] * v).sum() / self.ssq_N
        if np.abs(h / lam) < 1e-10:
            raise InstabilityError()
        return distributions.GaussianDistribution(h / lam, 1. / lam)

    def fit_v_and_var(self):
        self.v = self.cond_v().maximize()
        #self.v /= np.sqrt(np.mean(self.v ** 2))
        self.ssq_u = self.sum_u_sq / (self.num_assigned + 1)

class Proposal:
    def __init__(self, u, v, ssq_u):
        self.u = u
        self.v = v
        self.ssq_u = ssq_u


def make_proposal(resid, obs, ssq_N, order=None):
    pi = ProposalInfo(resid, obs, ssq_N)
    N, D = resid.shape
    if order is None:
        order = range(N)

    for i in order:
        if i == order[0]:
            dist = distributions.GaussianDistribution(0., 1.)
        else:
            dist = pi.cond_u(i)
        pi.update_u(i, dist.sample())
        pi.fit_v_and_var()

    v = pi.cond_v().sample()
    ssq_u = pi.cond_ssq_u().sample()

    return Proposal(pi.u.copy(), v, ssq_u)
        
def proposal_probability(resid, obs, ssq_N, proposal, order=None):
    pi = ProposalInfo(resid, obs, ssq_N)
    N, D = resid.shape
    if order is None:
        order = range(N)

    total = 0.
    for i in order:
        if i == order[0]:
            dist = distributions.GaussianDistribution(0., 1.)
        else:
            dist = pi.cond_u(i)

        total += dist.loglik(proposal.u[i])
        pi.update_u(i, proposal.u[i])
        pi.fit_v_and_var()

    total += pi.cond_v().loglik(proposal.v).sum()
    total += pi.cond_ssq_u().loglik(proposal.ssq_u)

    return total


def log_poisson(K, lam):
    return -lam + K * np.log(lam) - scipy.special.gammaln(K+1)

def p_star(state, X, obs):
    K = state.U.shape[1]
    total = log_poisson(K, 1.)

    var_prior = distributions.InverseGammaDistribution(A, B)
    total += var_prior.loglik(state.ssq_U).sum()

    assert np.isfinite(total)

    U_dist = distributions.GaussianDistribution(0., state.ssq_U[nax, :])
    total += U_dist.loglik(state.U).sum()

    assert np.isfinite(total)

    V_dist = distributions.GaussianDistribution(0., 1.)
    total += V_dist.loglik(state.V).sum()

    assert np.isfinite(total)
    
    pred = np.dot(state.U, state.V)
    X_dist = distributions.GaussianDistribution(pred, state.ssq_N)
    total += X_dist.loglik(X)[obs].sum()

    assert np.isfinite(total)

    return total

def add_delete_move(state, X, obs):
    N, K, D = state.U.shape[0], state.U.shape[1], state.V.shape[1]
    order = np.random.permutation(N)
    if np.random.binomial(1, 0.5):   # add move
        pred = np.dot(state.U, state.V)
        resid = X - pred
        try:
            proposal = make_proposal(resid, obs, state.ssq_N, order)
        except InstabilityError:
            return state
        except OverflowError:
            return state
        forward_prob = -np.log(2) + proposal_probability(resid, obs, state.ssq_N, proposal, order)
        backward_prob = -np.log(2) - np.log(K + 1)

        new_U = np.hstack([state.U, proposal.u[:, nax]])
        new_V = np.vstack([state.V, proposal.v[nax, :]])
        new_ssq_U = np.concatenate([state.ssq_U, [proposal.ssq_u]])
        new_state = State(new_U, new_V, new_ssq_U, state.ssq_N)
        p_star_new = p_star(new_state, X, obs)
        p_star_old = p_star(state, X, obs)

        ratio = p_star_new - p_star_old - forward_prob + backward_prob
        assert np.isfinite(ratio)
        if np.random.binomial(1, min(np.exp(ratio), 1)):
            if VERBOSE:
                print 'Add move accepted (ratio=%1.2f)' % ratio
            return new_state
        else:
            if VERBOSE:
                print 'Add move rejected (ratio=%1.2f)' % ratio
            return state
        
    else:   # delete move
        if K <= 2:   # zero or one dimensions causes NumPy awkwardness
            return state
        
        k = np.random.randint(0, K)
        pred = np.dot(state.U, state.V) - np.outer(state.U[:, k], state.V[k, :])
        resid = X - pred
        reverse_proposal = Proposal(state.U[:, k], state.V[k, :], state.ssq_U[k])
        forward_prob = -np.log(2) - np.log(K)
        try:
            backward_prob = -np.log(2) + proposal_probability(resid, obs, state.ssq_N, reverse_proposal, order)
        except InstabilityError:
            return state
        except OverflowError:
            return state

        new_U = np.hstack([state.U[:, :k], state.U[:, k+1:]])
        new_V = np.vstack([state.V[:k, :], state.V[k+1:, :]])
        new_ssq_U = np.concatenate([state.ssq_U[:k], state.ssq_U[k+1:]])
        new_state = State(new_U, new_V, new_ssq_U, state.ssq_N)

        p_star_new = p_star(new_state, X, obs)
        p_star_old = p_star(state, X, obs)

        ratio = p_star_new - p_star_old - forward_prob + backward_prob
        assert np.isfinite(ratio)
        if np.random.binomial(1, min(np.exp(ratio), 1)):
            if VERBOSE:
                print 'Delete move accepted (ratio=%1.2f)' % ratio
            return new_state
        else:
            if VERBOSE:
                print 'Delete move rejected (ratio=%1.2f)' % ratio
            return state
        


NUM_ITER = 200

def init_state(data_matrix, K):
    N, D = data_matrix.m, data_matrix.n
    X = data_matrix.sample_latent_values(np.zeros((N, D)), 1.)
    U = np.random.normal(0., 1. / np.sqrt(K), size=(N, K))
    V = np.random.normal(0., 1., size=(K, D))
    ssq_U = np.mean(U**2, axis=0)

    pred = np.dot(U, V)
    if data_matrix.observations.fixed_variance():
        ssq_N = 1.
    else:
        ssq_N = np.mean((X - pred) ** 2)
    return X, State(U, V, ssq_U, ssq_N)

def fit_model(data_matrix, K=K_INIT, num_iter=NUM_ITER, name=None):
    if SEED_0:
        np.random.seed(0)
    N, D = data_matrix.m, data_matrix.n
    X, state = init_state(data_matrix, K)

    pbar = misc.pbar(num_iter)

    t0 = time.time()
    for it in range(num_iter):
        sample_U_V(state, X, data_matrix.observations.mask)

        old = np.dot(state.U, state.V)
        givens_moves(state)
        assert np.allclose(np.dot(state.U, state.V), old)
        scaling_moves(state)
        assert np.allclose(np.dot(state.U, state.V), old)

        state.ssq_U = sample_variance(state.U, 0)
        pred = np.dot(state.U, state.V)
        if not data_matrix.observations.fixed_variance():
            state.ssq_N = sample_variance(X - pred, None, mask=data_matrix.observations.mask)

        X = data_matrix.sample_latent_values(pred, state.ssq_N)

        for i in range(10):
            state = add_delete_move(state, X, data_matrix.observations.mask)

        if VERBOSE:
            print 'K =', state.U.shape[1]
            print 'ssq_N =', state.ssq_N
            print 'X.var() =', X.var()

        #misc.print_dot(it+1, num_iter)
        pbar.update(it)

        if time.time() - t0 > 3600.:   # 1 hour
            break

    pbar.finish()

    return state, X



