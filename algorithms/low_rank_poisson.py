import cPickle
import itertools
import numpy as np
nax = np.newaxis
import pylab
import random
import scipy.integrate
import scipy.linalg
import scipy.special
import time

import config
import low_rank
import observations
from utils import distributions, gaussians, misc, psd_matrices

A = 0.1
B = 0.1

VERBOSE = False
SEED_0 = False
PRINT = False
PRINT_PREDICTIVE = False
PRINT_ALL = False
DEBUG_DELETE = False
FIXED_NOISE_VARIANCE = None
K_INIT_OVERRIDE = 2
PLOT_POINTS = False

INIT_SVD = False

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
        #if PRINT:
        #    print 'self.v', self.v
        #    print 'self.ssq_u', self.ssq_u

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

        if PRINT and PRINT_ALL:
            print 'i =', i
            print 'dist.mu', dist.mu
            print 'dist.sigma_sq', dist.sigma_sq
            print 'u[i]', proposal.u[i]
            if i != order[0]:
                print 'norm(v)', np.linalg.norm(pi.v)
                print 'sigma_sq_u', pi.ssq_u
            print 'score:', dist.loglik(proposal.u[i])
            print
        
        total += dist.loglik(proposal.u[i])
        pi.update_u(i, proposal.u[i])
        pi.fit_v_and_var()

    if PRINT:
        print 'cond_v.mu', pi.cond_v().mu
        print 'cond_v.sigma_sq', pi.cond_v().sigma_sq
        print 'v', proposal.v
        print

        print 'total U score:', total

    total += pi.cond_v().loglik(proposal.v).sum()
    total += pi.cond_ssq_u().loglik(proposal.ssq_u)

    if PRINT:
        print 'V score:', pi.cond_v().loglik(proposal.v).sum()
        print 'ssq_u score:', pi.cond_ssq_u().loglik(proposal.ssq_u)
        assert pi.cond_ssq_u().loglik(proposal.ssq_u) > -100.
        print
        
    assert np.isscalar(total)

    return total

def print_predictive(resid, obs, ssq_N, proposal, order=None):
    pi = ProposalInfo(resid, obs, ssq_N)
    N, D = resid.shape
    dist_without = distributions.GaussianDistribution(0., ssq_N)
    if order is None:
        order = range(N)

    total_with = 0.
    total_without = 0.
    for i in order:
        idxs = np.where(obs[i, :])[0]

        if PRINT_ALL and i % 20 == 0:
            print 'ssq_u', pi.ssq_u
            print 'ssq_N', ssq_N
            print 'pi.v', pi.v
            print 'proposal.v', proposal.v
            print 'u[i]', proposal.u[i]

        if i == order[0]:
            Sigma = ssq_N * np.eye(idxs.size)
        else:
            Sigma = pi.ssq_u * np.outer(pi.v[idxs], pi.v[idxs]) + ssq_N * np.eye(idxs.size)
        pot = gaussians.Potential.from_moments_full(np.zeros(idxs.size), Sigma)
        score_with = pot.score(resid[i, idxs])
        total_with += score_with
        
        score_without = dist_without.loglik(resid[i, :])[idxs].sum()
        total_without += score_without

        pi.update_u(i, proposal.u[i])
        pi.fit_v_and_var()


        if PRINT_ALL:
            print 'i =', i
            print 'Score with:', score_with
            print 'Score without:', score_without
            print

    print 'Total with:', total_with
    print 'Total without:', total_without


def log_poisson(K, lam):
    return -lam + K * np.log(lam) - scipy.special.gammaln(K+1)

def p_star(state, X, obs):
    K = state.U.shape[1]
    total = log_poisson(K, 1.)

    var_prior = distributions.InverseGammaDistribution(A, B)
    total += var_prior.loglik(state.ssq_U).sum()

    assert np.isfinite(total)
    if PRINT:
        print 'ssq_U prior', var_prior.loglik(state.ssq_U).sum()

    U_dist = distributions.GaussianDistribution(0., state.ssq_U[nax, :])
    total += U_dist.loglik(state.U).sum()

    assert np.isfinite(total)
    if PRINT:
        print 'U', U_dist.loglik(state.U).sum()

    V_dist = distributions.GaussianDistribution(0., 1.)
    total += V_dist.loglik(state.V).sum()

    assert np.isfinite(total)
    if PRINT:
        print 'V', V_dist.loglik(state.V).sum()
    
    pred = np.dot(state.U, state.V)
    X_dist = distributions.GaussianDistribution(pred, state.ssq_N)
    total += X_dist.loglik(X)[obs].sum()

    assert np.isfinite(total)
    if PRINT:
        print 'X', X_dist.loglik(X)[obs].sum()
        print 'MSE:', np.mean((X - pred) ** 2)

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

        if PRINT:
            print 'New:'
        p_star_new = p_star(new_state, X, obs)
        if PRINT:
            print
            print 'Old:'
        p_star_old = p_star(state, X, obs)
        if PRINT:
            print

        if PRINT_PREDICTIVE:
            print_predictive(resid, obs, state.ssq_N, reverse_proposal)

        if PRINT:
            print 'p_star_new', p_star_new
            print 'p_star_old', p_star_old
            print 'forward_prob', forward_prob
            print 'backward_prob', backward_prob

        ratio = p_star_new - p_star_old - forward_prob + backward_prob
        assert np.isfinite(ratio)
        if np.random.binomial(1, min(np.exp(ratio), 1)):
            if VERBOSE:
                print 'Delete move accepted (ratio=%1.2f)' % ratio
            return new_state
        else:
            if VERBOSE:
                print 'Delete move rejected (ratio=%1.2f)' % ratio
            #assert ratio < -500.
            #if ratio > -500.:
                #print_predictive(resid, obs, state.ssq_N, reverse_proposal)
            #    assert False
            return state
        


NUM_ITER = 200

def init_state(data_matrix, K):
    N, D = data_matrix.m, data_matrix.n
    X = data_matrix.sample_latent_values(np.zeros((N, D)), 1.)
    if INIT_SVD:
        U_, s_, V_ = scipy.linalg.svd(X, full_matrices=False)
        U = U_[:, :K] * np.sqrt(s_[:K][nax, :])
        V = V_[:K, :] * np.sqrt(s_[:K][:, nax])
    else:
        U = np.random.normal(0., 1. / np.sqrt(K), size=(N, K))
        V = np.random.normal(0., 1., size=(K, D))
    ssq_U = np.mean(U**2, axis=0)

    pred = np.dot(U, V)
    if data_matrix.observations.fixed_variance():
        ssq_N = 1.
    else:
        ssq_N = np.mean((X - pred) ** 2)
    return X, State(U, V, ssq_U, ssq_N)

def fit_model(data_matrix, K=20, num_iter=NUM_ITER, name=None):
    if K_INIT_OVERRIDE is not None:
        K = K_INIT_OVERRIDE

    if SEED_0:
        np.random.seed(0)
    N, D = data_matrix.m, data_matrix.n
    X, state = init_state(data_matrix, K)

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
        if FIXED_NOISE_VARIANCE is not None:
            state.ssq_N = FIXED_NOISE_VARIANCE

        X = data_matrix.sample_latent_values(pred, state.ssq_N)

        for i in range(10):
            state = add_delete_move(state, X, data_matrix.observations.mask)

        if VERBOSE:
            print 'K =', state.U.shape[1]
            print 'ssq_N =', state.ssq_N
            print 'X.var() =', X.var()

        misc.print_dot(it+1, num_iter)

        if time.time() - t0 > 3600.:   # 1 hour
            break
        if config.USE_AMAZON_S3:
            amazon.check_visibility()

        if PLOT_POINTS:
            U_, s_, Vt_ = scipy.linalg.svd(X)
            vis.figure('X (row)')
            pylab.clf()
            pylab.plot(U_[:, 0], U_[:, 1], 'bx')
            pylab.title('X (row)')
            vis.figure('X (col)')
            pylab.clf()
            pylab.plot(Vt_[0, :], Vt_[1, :], 'rx')
            pylab.title('X (col)')

            U_, s_, Vt_ = scipy.linalg.svd(state.U)
            vis.figure('U')
            pylab.clf()
            pylab.plot(U_[:, 0], U_[:, 1], 'bx')

            U_, s_, Vt_ = scipy.linalg.svd(state.V)
            vis.figure('V')
            pylab.clf()
            pylab.plot(Vt_[0, :], Vt_[1, :], 'rx')

            pylab.draw()

    return state, X







K_INIT_ALL = [2, 5, 10, 20, 50]

def fit_model_parallel(data_matrix, num_iter=NUM_ITER, name=None):
    N, D = data_matrix.m, data_matrix.n
    states = []
    for K in K_INIT_ALL:
        X, st = init_state(data_matrix, K)
        states.append(st)

    # temporary
    ssq_N_log = []

    K_log = []
    for it in range(num_iter):
        for j, state in enumerate(states):
            sample_U_V(state, X, data_matrix.observations.mask)

            givens_moves(state)
            scaling_moves(state)

            print 'ssq_U', sorted(state.ssq_U)


            state.ssq_U = sample_variance(state.U, 0)
            pred = np.dot(state.U, state.V)
            if not data_matrix.observations.fixed_variance():
                state.ssq_N = sample_variance(X - pred, None, mask=data_matrix.observations.mask)

            print 'ssq_N', state.ssq_N

            #state.ssq_N = 0.1

            X = data_matrix.sample_latent_values(pred, state.ssq_N)

            for i in range(10):
                states[j] = add_delete_move(states[j], X, data_matrix.observations.mask)

        misc.print_dot(it+1, num_iter)

        K_log.append([s.U.shape[1] for s in states])


        # temporary
        ssq_N_log.append([s.ssq_N for s in states])

        title = 'Convergence of low-rank model (%s)' % name
        vis.pw.figure(title)
        pylab.clf()
        pylab.plot(range(1, it+2), K_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('Number of dimensions')

        title = 'Convergence of low rank model (%s, log-log)' % name
        vis.pw.figure(title)
        pylab.clf()
        pylab.loglog(range(1, it+2), K_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('Number of dimensions')
        pylab.draw()

        # temporary
        title = 'sigma_sq_N'
        vis.pw.figure(title)
        pylab.clf()
        pylab.semilogy(range(1, it+2), ssq_N_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('sigma_sq_N')
        pylab.draw()
        



def run():
    dm = data.load_senate_data(2008, real_valued=True)
    U, V, ssq_U, ssq_V, ssq_N, X = low_rank.fit_model(dm, 20, num_iter=2, rotation_trick=False)
    N, K, D = U.shape[0], U.shape[1], V.shape[1]

    #norm = (U**2).sum(0)
    #idxs = np.argsort(norm)
    #i, j = idxs[-2], idxs[-1]

    for tr in range(10):
        a = np.random.randint(0, K)
        b = np.random.randint(0, K)
        if a == b:
            continue

        theta = np.linspace(-np.pi, np.pi)
        uaa = np.dot(U[:, a], U[:, a])
        uab = np.dot(U[:, a], U[:, b])
        ubb = np.dot(U[:, b], U[:, b])

        sin, cos = np.sin(theta), np.cos(theta)
        uaa_prime_ssq = uaa * cos ** 2 + 2 * uab * cos * sin + ubb * sin ** 2
        ubb_prime_ssq = uaa * sin ** 2 - 2 * uab * cos * sin + ubb * cos ** 2
        prob = -(A + 0.5 * N) * (np.log(B + 0.5 * uaa_prime_ssq) + np.log(B + 0.5 * ubb_prime_ssq))

        pylab.figure()
        pylab.plot(theta, prob)


def run2():
    dm = data.load_senate_data(2008, real_valued=True)
    U, V, ssq_U, ssq_V, ssq_N, X = low_rank.fit_model(dm, 20, num_iter=2, rotation_trick=False)
    N, K, D = U.shape[0], U.shape[1], V.shape[1]

    vis.pw.figure('before')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))
    #print np.dot(U.T, U)

    state = State(U, V, None, None)
    givens_moves(state)

    vis.pw.figure('after')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))

    givens_moves(state)
    vis.pw.figure('after 2')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))
    


def check_integral():
    u1 = np.random.normal(size=20)
    u2 = np.random.normal(size=20)
    exact1 = p_u(u1)
    exact2 = p_u(u2)

    #lam = np.logspace(-4., 4., 100000)
    npts = 10000
    lam = np.linspace(0., 4., npts)
    p_lam = A * np.log(lam) - B * lam
    p_u1_given_lam = distributions.gauss_loglik(u1[:, nax], 0., 1. / lam[nax, :]).sum(0)
    p_u2_given_lam = distributions.gauss_loglik(u2[:, nax], 0., 1. / lam[nax, :]).sum(0)
    approx1 = np.log(scipy.integrate.trapz(np.exp(p_lam + p_u1_given_lam), lam))
    approx2 = np.log(scipy.integrate.trapz(np.exp(p_lam + p_u2_given_lam), lam))
    
    print 'exact1 - exact2', exact1 - exact2
    print 'approx1 - approx2', approx1 - approx2

def check_integral2():
    exact = np.zeros(100)
    approx = np.zeros(100)
    u0 = np.random.normal(size=20)
    alpha_all = np.linspace(0., 1., 100)
    for i, alpha in enumerate(alpha_all):
        exact[i] = p_u(alpha * u0)
        
        npts = 1000000
        lam = np.linspace(0., 1000., npts)
        p_lam = A * np.log(lam) - B * lam
        p_u_given_lam = distributions.gauss_loglik(alpha * u0[:, nax], 0., 1. / lam[nax, :]).sum(0)
        approx[i] = np.log(scipy.integrate.simps(np.exp(p_lam + p_u_given_lam), lam))

        misc.print_dot(i+1, len(alpha_all))

    exact -= exact[-1]
    approx -= approx[-1]
    pylab.figure()
    pylab.plot(alpha_all, exact, 'b', alpha_all, approx, 'r')

def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        raise RuntimeError('a=%f, b=%f' % (a, b))
    
def check_cond_U_Vt(missing=False):
    N, K, D = 20, 15, 30
    ssq_U = np.random.uniform(0., 1., size=K)
    ssq_N = np.random.uniform(0., 1.)
    U = np.random.normal(0., np.sqrt(ssq_U[nax, :]), size=(N, K))
    V = np.random.normal(0., 1., size=(K, D))
    X = np.random.normal(np.dot(U, V), np.sqrt(ssq_N))
    state = State(U, V, ssq_U, ssq_N)
    if missing:
        mask = np.random.binomial(1, 0.5, size=(N, D)).astype(bool)
    else:
        mask = np.ones((N, D), dtype=bool)

    U_new = np.random.normal(size=(N, K))
    new_state = state.copy()
    new_state.U = U_new
    cond = cond_U(X, mask, V, ssq_U, ssq_N)
    check_close(cond.score(U_new).sum() - cond.score(U).sum(),
                p_star(new_state, X, mask) - p_star(state, X, mask))

    V_new = np.random.normal(size=(K, D))
    new_state = state.copy()
    new_state.V = V_new
    cond = cond_Vt(X, mask, U, ssq_N)
    check_close(cond.score(V_new.T).sum() - cond.score(V.T).sum(),
                p_star(new_state, X, mask) - p_star(state, X, mask))
    
    

def generate_synthetic(noise_var):
    #N, K, D = 1000, 10, 1050
    #N, K, D = 100, 10, 150
    N, K, D = 100, 10, 150
    U = np.random.normal(size=(N, K))
    V = np.random.normal(size=(K, D))
    X = np.dot(U, V)
    X /= np.sqrt(np.mean(X ** 2))
    X = np.random.normal(X, np.sqrt(noise_var))
    return observations.DataMatrix.from_real_values(X)
    
    

def run_real(name, parallel=False, run=1, num_iter=100):
    if name == 'senate':
        dm = data.load_senate_data(2008, real_valued=True)
    elif name == 'intel':
        dm = data.load_intel_data(real_valued=True)
    elif name == 'senate-binary':
        dm = data.load_senate_data(2008)
    elif name == 'intel-integer':
        dm = data.load_intel_data()
    elif name == 'senate-noiseless':
        dm = data.load_senate_data(2008, real_valued=True, noise_variance=1e-5)
    elif name == 'intel-noiseless':
        dm = data.load_intel_data(real_valued=True, noise_variance=1e-5)
    elif name[:9] == 'synthetic':
        noise_var = float(name[10:])
        dm = generate_synthetic(noise_var)
    elif name == 'movielens-integer':
        dm = movielens_data.load_data_matrix()
        dm = dm[:, :1000]
    elif name == 'movielens-real':
        dm = movielens_data.load_data_matrix(True)
        dm = dm[:, :1000]
    elif name == 'patches':
        X = data.load_image_patches(1000)
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'patches2':
        X = data.load_image_patches(1000)
        X = X[:, ::2]
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'temp-intel':
        dm = data.load_intel_data(real_valued=True, noise_variance=0.1)
    elif name == 'temp-patches':
        X = data.load_image_patches(4000)
        cols = np.random.permutation(144)[:115]
        X = X[:, cols]
        X /= X.std()
        X = np.random.normal(X, 0.1)
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'temp-patches2':
        dm = cPickle.load(open(experiments.data_file('image-patches-6-22')))
        splits = cPickle.load(open(experiments.splits_file('image-patches-6-22')))
        train_rows, train_cols, test_rows, test_cols = splits[0]
        train_cols = np.random.permutation(144)[:144*4//5]
        train_rows = np.arange(500)
        dm = dm[train_rows[:, nax], train_cols[nax, :]]
    name += ' run %d' % run
    
    if parallel:
        fit_model_parallel(dm, num_iter=num_iter, name=name)
    else:
        fit_model(dm, num_iter=num_iter, name=name)

def run_all():
    #NAMES = ['synthetic-1', 'synthetic-0.1', 'senate', 'senate-binary', 'intel', 'intel-integer']
    NAMES = ['intel', 'intel-integer']
    for name in NAMES:
        for i in range(1, 4):
            print name, i
            run_real(name, True, i)
        vis.save_all_figures('/tmp/roger/low-rank')


import initialization
import recursive
import grammar
import scoring

def compare_old_and_new(name):
    NUM_SAMPLES = 2
    
    if name == 'senate':
        dm = data.load_senate_data(2008, real_valued=True)
    elif name == 'intel':
        dm = data.load_intel_data(real_valued=True)
    elif name == 'senate-binary':
        dm = data.load_senate_data(2008)
    elif name == 'intel-binary':
        dm = data.load_intel_data()
    elif name[:9] == 'synthetic':
        noise_var = float(name[10:])
        dm = generate_synthetic(noise_var)

    train_rows = np.arange(0, dm.m, 2)
    test_rows = np.arange(1, dm.m, 2)
    train_cols = np.arange(0, dm.n, 2)
    test_cols = np.arange(1, dm.n, 2)

    X_train = dm[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = dm[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = dm[train_rows[:, nax], test_cols[nax, :]]

    old_row_loglik = []
    old_col_loglik = []
    new_row_loglik = []
    new_col_loglik = []
    


    for i in range(NUM_SAMPLES):
        initialization.USE_OLD_LOW_RANK = True
        old_root = recursive.fit_model(grammar.parse('gg+g'), X_train, gibbs_steps=50)
        rl, cl = scoring.evaluate_model(X_train, old_root, X_row_test, X_col_test)
        old_row_loglik.append(rl)
        old_col_loglik.append(cl)

        initialization.USE_OLD_LOW_RANK = False
        new_root = recursive.fit_model(grammar.parse('gg+g'), X_train, gibbs_steps=50)
        rl, cl = scoring.evaluate_model(X_train, new_root, X_row_test, X_col_test)
        new_row_loglik.append(rl)
        new_col_loglik.append(cl)

    old_row_loglik = np.array(old_row_loglik)
    old_col_loglik = np.array(old_col_loglik)
    new_row_loglik = np.array(new_row_loglik)
    new_col_loglik = np.array(new_col_loglik)

    print 'Average row log-likelihood (each sample)'
    print '    Old:', old_row_loglik.mean(1)
    print '    New:', new_row_loglik.mean(1)
    print 'Average column log-likelihood (each sample)'
    print '    Old:', old_col_loglik.mean(1)
    print '    New:', new_col_loglik.mean(1)
    print 'Average row log-likelihood (combined)'
    print '    Old:', np.logaddexp.reduce(old_row_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print '    New:', np.logaddexp.reduce(new_row_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print 'Average column log-likelihood (combined)'
    print '    Old:', np.logaddexp.reduce(old_col_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print '    New:', np.logaddexp.reduce(new_col_loglik, 0).mean() - np.log(NUM_SAMPLES)


    
