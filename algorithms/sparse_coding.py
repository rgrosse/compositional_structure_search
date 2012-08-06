import matplotlib
if __name__ == '__main__':
    matplotlib.use('agg')

import numpy as np
nax = np.newaxis
import pylab
import scipy.optimize

import observations
import slice_sampling
from utils import distributions, gaussians, misc, psd_matrices


debugger = None
NUM_MINIBATCHES = 5
INIT_MU_Z = -2.
INIT_SIGMA_SQ_Z = 1.
INIT_SIGMA_SQ_N = 1.
INIT_OLS_FIELD = True
NEW_HYPERPARAMETER_METHOD = True
OPTIMIZE_BASES = True
NUM_ITER = 200


def log_pz(Z, S, sigma_sq_Z, mu_Z):
    var_term = -0.5 * (Z-mu_Z)**2 / sigma_sq_Z
    log_det_term = -0.5 * Z
    S_term = -0.5 * S**2 * np.exp(-Z)
    return var_term + log_det_term + S_term

class ZObj:
    def __init__(self, S, sigma_sq_Z, mu_Z):
        self.shape = S.shape
        self.S = S
        self.sigma_sq_Z = sigma_sq_Z
        self.mu_Z = mu_Z

    def value(self, Z):
        return -np.sum(log_pz(Z, self.S, self.sigma_sq_Z, self.mu_Z))

    def gradient(self, Z):
        return (Z - self.mu_Z) / self.sigma_sq_Z + 0.5 - 0.5 * self.S**2 * np.exp(-Z)

    def __call__(self, z):
        Z = z.reshape(self.shape)
        return self.value(Z), self.gradient(Z).ravel()
    

def sample_coeffs(X, A, Z, S, sigma_sq_N, by_column=False):
    assert False # unused
    S = S.copy()
    
    # estimate variances
    if by_column:
        mu_Z = np.mean(Z, axis=0)
        sigma_sq_Z = np.mean((Z - mu_Z)**2, axis=0)
    else:
        mu_Z = np.mean(Z)
        sigma_sq_Z = np.mean((Z-mu_Z)**2)

    # sample Z
    for i in range(10):
        scores = log_pz(Z, S, sigma_sq_Z, mu_Z)
        Z_guess = Z + np.random.normal(0., 0.1, size=Z.shape)
        scores_guess = log_pz(Z_guess, S, sigma_sq_Z, mu_Z)
        p = np.clip(np.exp(scores_guess - scores), 0., 1.)
        Z = np.where(np.random.binomial(1, p), Z_guess, Z)

    # sample S
    ndata = S.shape[0]
    for i in range(ndata):
        Lambda_S = np.dot(np.dot(A, np.diag(1. / sigma_sq_N[i,:])), A.T) + np.diag(np.exp(-Z[i,:]))
        temp = np.dot(A, X[i,:] / sigma_sq_N[i,:])
        mu_S = np.linalg.solve(Lambda_S, temp)
        S[i,:] = np.random.multivariate_normal(mu_S, np.linalg.inv(Lambda_S))

    if hasattr(debugger, 'after_sample_coeffs'):
        debugger.after_sample_coeffs(vars())

    return Z, S


class SparseCodingState:
    def __init__(self, S, A, Z, sigma_sq_N, mu_Z, sigma_sq_Z, sigma_sq_A):
        self.S = S
        self.A = A
        self.Z = Z
        self.sigma_sq_N = sigma_sq_N
        self.mu_Z = mu_Z
        self.sigma_sq_Z = sigma_sq_Z
        self.sigma_sq_A = sigma_sq_A

    def copy(self):
        if np.isscalar(self.mu_Z):
            mu_Z = self.mu_Z
        else:
            mu_Z = self.mu_Z.copy()
        return SparseCodingState(self.S.copy(), self.A.copy(), self.Z.copy(), self.sigma_sq_N, mu_Z,
                                 self.sigma_sq_Z, self.sigma_sq_A)

def log_p(X, state):
    N, D = X.shape
    K = state.S.shape[1]
    total = 0.

    # improper priors over hyperparameters
    total -= np.log(state.sigma_sq_Z)
    total -= np.log(state.sigma_sq_N)
    
    # probability of Z
    if np.isscalar(state.mu_Z):
        prior_Z = gaussians.Potential.from_moments_iso(state.mu_Z * np.ones(K), state.sigma_sq_Z)
    else:
        prior_Z = gaussians.Potential.from_moments_iso(state.mu_Z, state.sigma_sq_Z)
    prob_Z = prior_Z.score(state.Z).sum()
    total += prob_Z

    # probability of S given Z
    dist_S = gaussians.Potential.from_moments_diag(np.zeros(K), np.exp(state.Z))
    prob_S_given_Z = dist_S.score(state.S).sum()
    total += prob_S_given_Z

    # probability of A
    dist_A = gaussians.Potential.from_moments_iso(np.zeros(K), state.sigma_sq_A)
    prob_A = dist_A.score(state.A.T).sum()
    total += prob_A

    # evidence
    pred = np.dot(state.S, state.A)
    dist_X = gaussians.Potential.from_moments_iso(pred, state.sigma_sq_N)
    prob_evidence = dist_X.score(X).sum()
    total += prob_evidence

    return total

def cond_mu_Z(state, by_column=False):
    if by_column:
        mu = state.Z.mean(0)
        sigma_sq = state.sigma_sq_Z / state.Z.shape[0] * np.ones(state.Z.shape[1])
    else:
        mu = state.Z.mean()
        sigma_sq = state.sigma_sq_Z / state.Z.size
    return distributions.GaussianDistribution(mu, sigma_sq)

def cond_sigma_sq_Z(state):
    a = 1. + 0.5 * state.Z.size
    b = 1. + 0.5 * np.sum((state.Z - state.mu_Z) ** 2)
    return distributions.InverseGammaDistribution(a, b)

def cond_sigma_sq_N(X, state):
    a = 1. + 0.5 * X.size
    pred = np.dot(state.S, state.A)
    b = 1. + 0.5 * np.sum((X - pred) ** 2)
    return distributions.InverseGammaDistribution(a, b)

def cond_S(X, state):
    N, K, D = X.shape[0], state.Z.shape[1], X.shape[1]
    prior_term = gaussians.Potential.from_moments_diag(np.zeros((N, K)), np.exp(state.Z))
    Lambda = np.dot(state.A, state.A.T) / state.sigma_sq_N
    h = np.dot(X, state.A.T) / state.sigma_sq_N
    evidence = gaussians.Potential(h, psd_matrices.FullMatrix(Lambda[nax, :, :]), 0.)  # ignore normalization
    return (prior_term + evidence).to_distribution()

def cond_S_row(X, state, i, AAT):
    N, K, D = X.shape[0], state.Z.shape[1], X.shape[1]
    prior_term = gaussians.Potential.from_moments_diag(np.zeros(K), np.exp(state.Z[i, :]))
    Lambda = AAT / state.sigma_sq_N
    h = np.dot(state.A, X[i, :]) / state.sigma_sq_N
    evidence = gaussians.Potential(h, psd_matrices.FullMatrix(Lambda), 0.)  # ignore normalization
    return (prior_term + evidence).to_distribution()
    
def cond_A_transp(X, state, subset=None):
    N, K, D = X.shape[0], state.Z.shape[1], X.shape[1]
    if subset is None:
        S = state.S
    else:
        S = state.S[subset, :]
        X = X[subset, :]
        
    prior_term = gaussians.Potential.from_moments_iso(np.zeros(K), state.sigma_sq_A)
    Lambda = np.dot(S.T, S) / state.sigma_sq_N
    h = np.dot(S.T, X) / state.sigma_sq_N
    evidence = gaussians.Potential(h.T, psd_matrices.FullMatrix(Lambda[nax, :, :]), 0.)  # ignore normalization
    return (prior_term + evidence).to_distribution()

def optimize_bases(X, state, subset):
    N, K, D = X.shape[0], state.Z.shape[1], X.shape[1]
    if subset is None:
        S = state.S
    else:
        S = state.S[subset, :]
        X = X[subset, :]

    fobj = ols_field.BasesDualObjective(X, S, 1.)
    lam_0 = np.ones(K)
    bounds = [(1e-4, None)] * K

    lam = scipy.optimize.fmin_tnc(fobj, lam_0, bounds=bounds, messages=0)[0]
    A = np.zeros((K, D))
    for j in range(D):
        A[:,j] = np.linalg.solve(np.dot(S.T, S) + np.diag(lam), np.dot(S.T, X[:,j]))

    if hasattr(debugger, 'after_optimize_bases'):
        debugger.after_optimize_bases(vars())

    return A
    


class LogFCollapsed:
    def __init__(self, lam, h):
        self.lam = lam
        self.h = h

    def __call__(self, z):
        sigma_sq = np.exp(z) + 1. / self.lam
        mu = self.h / self.lam

        return -0.5 * np.log(sigma_sq) + \
               -0.5 * mu ** 2 / sigma_sq

class LogFUncollapsed:
    def __init__(self, s):
        self.s = s

    def __call__(self, z):
        return -0.5 * z + \
               -0.5 * self.s ** 2 / np.exp(z)


def sample_Z(state):
    N, K= state.S.shape[0], state.Z.shape[1]
    for i in range(N):
        for k in range(K):
            log_f = LogFUncollapsed(state.S[i, k])
            if np.isscalar(state.mu_Z):
                mu_Z = state.mu_Z
            else:
                mu_Z = state.mu_Z[k]
            state.Z[i, k] = slice_sampling.slice_sample_gauss(log_f, mu_Z, state.sigma_sq_Z, state.Z[i, k])

    if hasattr(debugger, 'after_sample_Z'):
        debugger.after_sample_Z(vars())
    

def sample_S_Z_uncollapsed(X, state, subset=None, clear=False):
    N, K, D = X.shape[0], state.Z.shape[1], X.shape[1]
    if subset is None:
        subset = np.arange(N)

    if clear:
        state.S[subset, :] = 0.
        state.Z[subset, :] = 0.
    
    Lambda = np.dot(state.A, state.A.T) / state.sigma_sq_N
    for count, i in enumerate(subset):
        if (count+1) % 100 == 0:
            print count+1
        h = np.dot(state.A, X[i, :]) / state.sigma_sq_N
        s = state.S[i, :].copy()
        z = state.Z[i, :].copy()

        order = np.random.permutation(K)
        for k in order:
            # Gaussian over S[i, k] proportional to the evidence, holding the other variables fixed
            s[k] = 0.
            lam_k = Lambda[k, k]
            h_k = h[k] - np.dot(Lambda[k, :], s)

            log_f = LogFCollapsed(lam_k, h_k)
            if np.isscalar(state.mu_Z):
                mu_Z = state.mu_Z
            else:
                mu_Z = state.mu_Z[k]
            z[k] = slice_sampling.slice_sample_gauss(log_f, mu_Z, state.sigma_sq_Z, z[k])

            lam_post = lam_k + np.exp(-z[k])
            sigma_sq_post = 1. / lam_post
            mu_post = sigma_sq_post * h_k
            s[k] = np.random.normal(mu_post, np.sqrt(sigma_sq_post))

        state.S[i, :] = s
        state.Z[i, :] = z

    if hasattr(debugger, 'after_sample_S_Z_uncollapsed'):
        debugger.after_sample_S_Z_uncollapsed(vars())

def gibbs_sweep(X, state, fixed_variance=False):
    N = state.S.shape[0]
    subset = np.arange(state.it % NUM_MINIBATCHES, N, NUM_MINIBATCHES)
    
    sample_S_Z_uncollapsed(X, state, subset=subset, clear=False)

    AAT = np.dot(state.A, state.A.T)
    for i in range(X.shape[0]):
        state.S[i, :] = cond_S_row(X, state, i, AAT).sample()

    # temporary
    if OPTIMIZE_BASES:
        state.A = optimize_bases(X, state, subset)
    else:
        state.A = cond_A_transp(X, state, subset).sample().T
    state.A /= np.sqrt(np.sum(state.A**2, axis=1)[:, nax])

    if NEW_HYPERPARAMETER_METHOD:
        sample_hyperparameters(X, state, subset)
        assert not fixed_variance
    else:
        state.mu_Z = cond_mu_Z(state).sample()
        state.sigma_sq_Z = cond_sigma_sq_Z(state).sample()

        if not fixed_variance:
            state.sigma_sq_N = cond_sigma_sq_N(X, state).sample()

    if hasattr(debugger, 'after_gibbs_sweep'):
        debugger.after_gibbs_sweep(vars())


def get_K(N, D):
    return int(min(0.125 * N, 1.2 * D))


def fit_model(data_matrix, K=None, num_iter=NUM_ITER):
    N, D = data_matrix.m, data_matrix.n
    if K is None:
        K = get_K(N, D)

    if INIT_OLS_FIELD:
        S, A = ols_field.compute_decomposition(data_matrix, K, target=None)
        temp = np.sqrt(np.sum(A**2, axis=1))
        A /= temp[:, nax]
        S *= temp[nax, :]
    else:
        A = np.random.normal(0., 1. / np.sqrt(D), size=(K, D))
        S = np.zeros((N, K))
        
    mu_Z = INIT_MU_Z
    sigma_sq_Z = INIT_SIGMA_SQ_Z
    sigma_sq_N = INIT_SIGMA_SQ_N
    sigma_sq_A = 1. / D
    Z = np.random.normal(mu_Z, sigma_sq_Z, size=(N, K))
    fixed_variance = data_matrix.fixed_variance()

    pred = np.dot(S, A)
    X = data_matrix.sample_latent_values(pred, sigma_sq_N)

    state = SparseCodingState(S, A, Z, sigma_sq_N, mu_Z, sigma_sq_Z, sigma_sq_A)

    for it in range(num_iter):
        state.it = it
        gibbs_sweep(X, state, fixed_variance)
        pred = np.dot(state.S, state.A)
        X = data_matrix.sample_latent_values(pred, state.sigma_sq_N)
        misc.print_dot(it+1, num_iter)

        if hasattr(debugger, 'each_iter'):
            debugger.each_iter(vars())

    return state


############# estimating hyperparameters with slice sampling ###################

def hyp_evidence(s, z):
    return -0.5 * np.log(2*np.pi) + \
           -0.5 * z + \
           -0.5 * s**2 / np.exp(z)

class EvidenceDiff:
    def __init__(self, s, cutoff):
        self.s = s
        self.cutoff = cutoff

    def __call__(self, z):
        return hyp_evidence(self.s, z) - self.cutoff


def get_intervals(state, subset):
    N, K = state.S.shape
    num = len(subset)

    left = np.zeros((num, K))
    right = np.zeros((num, K))
    for idx, i in enumerate(subset):
        for j in range(K):
            z, s = state.Z[i, j], state.S[i, j]
            cutoff = hyp_evidence(s, z) + np.log(np.random.uniform(0., 1.))
            ediff = EvidenceDiff(s, cutoff)
            left[idx, j] = scipy.optimize.brentq(ediff, -10000., z)
            right[idx, j] = scipy.optimize.brentq(ediff, z, 10000.)
        misc.print_dot(idx+1, num)
    return left, right

def fobj_hyp(mu_z, sigma_sq_z, left, right):
    return observations.log_prob_between(mu_z, sigma_sq_z, left, right).sum()

class FObjMu:
    def __init__(self, sigma_sq_Z, left, right):
        self.sigma_sq_Z = sigma_sq_Z
        self.left = left
        self.right = right

    def __call__(self, mu_Z):
        return fobj_hyp(mu_Z, self.sigma_sq_Z, self.left, self.right)

class FObjSigmaSq:
    def __init__(self, mu_Z, left, right):
        self.mu_Z = mu_Z
        self.left = left
        self.right = right

    def __call__(self, log_sigma_sq_Z):
        return fobj_hyp(self.mu_Z, np.exp(log_sigma_sq_Z), self.left, self.right)

class FObjSigmaSqN:
    def __init__(self, Q, d, X):
        # Q: N x D x D
        # d: N x D
        N, D = d.shape
        self.Q = Q
        self.d = d
        self.X = X
        self.QTX = np.array([np.dot(Q[i, :, :].T, X[i, :])
                             for i in range(N)])

    def __call__(self, log_sigma_sq_N):
        sigma_sq_N = np.exp(log_sigma_sq_N)
        d_with_noise = self.d + sigma_sq_N
        return -0.5 * np.log(d_with_noise).sum() + \
               -0.5 * (self.QTX ** 2 / d_with_noise).sum()


def sample_noise(X, state, subset):
    subset = subset[:200]
    N = len(subset)

    D = X.shape[1]
    Q = np.zeros((N, D, D))
    d = np.zeros((N, D))
    for i in range(N):
        Sigma = np.dot(state.A.T, np.exp(state.Z[i, :])[:, nax] * state.A)
        d[i, :], Q[i, :, :] = scipy.linalg.eigh(Sigma)

    f = FObjSigmaSqN(Q, d, X)
    for i in range(10):
        state.sigma_sq_N = np.exp(slice_sampling.slice_sample(f, np.log(state.sigma_sq_N), -100., 100.))
    

def sample_hyperparameters(X, state, subset=None):
    if subset is None:
        subset = np.arange(state.S.shape[0])
    left, right = get_intervals(state, subset)

    f = FObjMu(state.sigma_sq_Z, left, right)
    for i in range(10):
        state.mu_Z = slice_sampling.slice_sample(f, state.mu_Z, -100., 100.)

    f = FObjSigmaSq(state.mu_Z, left, right)
    for i in range(10):
        state.sigma_sq_Z = np.exp(slice_sampling.slice_sample(f, np.log(state.sigma_sq_Z), -100., 100.))

    if X is not None:
       sample_noise(X, state, subset)

    if hasattr(debugger, 'after_sample_hyperparameters'):
        debugger.after_sample_hyperparameters(vars())

    

    




########################## rotation moves ######################################

def log_prob_s(s, mu_Z, sigma_sq_Z):
    zp = np.linspace(-25., 25., 1000)
    log_f = distributions.gauss_loglik(zp, mu_Z, sigma_sq_Z) + \
            distributions.gauss_loglik(s, 0., np.exp(zp))
    f = np.exp(log_f)
    return np.log(scipy.integrate.simps(f, zp))

def log_p_s_interp(mu_Z, sigma_sq_Z):
    #sp = np.linspace(-200., 200., 10000)
    sp = np.concatenate([np.linspace(-200., -1., 500)[:-1], np.linspace(-1., 1., 500), np.linspace(1., 200., 500)[1:]])
    log_prob = [log_prob_s(si, -1., 2.) for si in sp]
    return scipy.interpolate.interp1d(sp, log_prob, kind='cubic')

class RotationFObj:
    def __init__(self, s1, s2, log_p_s):
        self.s1 = s1
        self.s2 = s2
        self.log_p_s = log_p_s

    def __call__(self, theta):
        s1_prime = self.s1 * np.cos(theta) - self.s2 * np.sin(theta)
        s2_prime = self.s1 * np.sin(theta) + self.s2 * np.cos(theta)
        return self.log_p_s(s1_prime).sum() + self.log_p_s(s2_prime).sum()

def temp_plot_prob():
    sp = np.linspace(-200., 200., 10000)
    prob = [log_prob_s(si, -1., 2.) for si in sp]
    pylab.figure()
    pylab.plot(sp, prob)

def temp_plot_interp_fobj(state, j1, j2):
    log_p_s = log_p_s_interp(state.mu_Z, state.sigma_sq_Z)
    thetas = np.linspace(-np.pi, np.pi, 1000.)
    rot_fobj = RotationFObj(state.S[:, j1], state.S[:, j2], log_p_s)
    values = [rot_fobj(theta) for theta in thetas]
    pylab.plot(thetas, values)


########################## debugging code ######################################

def random_state(N, K, D):
    mu_Z = np.random.normal(0., 0.1)
    sigma_sq_Z = 0.7
    sigma_sq_N = 0.8
    sigma_sq_A = 0.9
    Z = np.random.normal(mu_Z, sigma_sq_Z, size=(N, K))
    S = np.random.normal(0., np.exp(0.5 * Z))
    A = np.random.normal(0., 1., size=(K, D))
    return SparseCodingState(S, A, Z, sigma_sq_N, mu_Z, sigma_sq_Z, sigma_sq_A)

def random_data(state):
    return np.random.normal(np.dot(state.S, state.A), np.sqrt(state.sigma_sq_N))

def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        raise RuntimeError('a=%f, b=%f' % (a, b))

def check_hyperparameters():
    N, K, D = 30, 10, 20
    state = random_state(N, K, D)
    X = random_data(state)

    cond = cond_mu_Z(state)
    for i in range(5):
        new_mu_Z = np.random.normal()
        new_state = state.copy()
        new_state.mu_Z = new_mu_Z
        assert np.allclose(cond.loglik(new_mu_Z) - cond.loglik(state.mu_Z),
                           log_p(X, new_state) - log_p(X, state))

    cond = cond_sigma_sq_Z(state)
    for i in range(5):
        new_sigma_sq_Z = np.random.gamma(1., 1.)
        new_state = state.copy()
        new_state.sigma_sq_Z = new_sigma_sq_Z
        check_close(cond.loglik(new_sigma_sq_Z) - cond.loglik(state.sigma_sq_Z),
                    log_p(X, new_state) - log_p(X, state))

    cond = cond_sigma_sq_N(X, state)
    for i in range(5):
        new_sigma_sq_N = np.random.gamma(1., 1.)
        new_state = state.copy()
        new_state.sigma_sq_N = new_sigma_sq_N
        check_close(cond.loglik(new_sigma_sq_N) - cond.loglik(state.sigma_sq_N),
                    log_p(X, new_state) - log_p(X, state))
        

def check_factors():
    N, K, D = 30, 10, 20
    state = random_state(N, K, D)
    X = random_data(state)

    ## for i in range(N):
    ##     cond = cond_S(X, state, i)
    ##     new_s = np.random.normal(0, 1., size=K)
    ##     new_state = state.copy()
    ##     new_state.S[i, :] = new_s
    ##     check_close(cond.loglik(new_s) - cond.loglik(state.S[i, :]),
    ##                 log_p(X, new_state) - log_p(X, state))

    cond = cond_S(X, state)
    for tr in range(5):
        new_S = np.random.normal(0., 1., size=(N, K))
        new_state = state.copy()
        new_state.S = new_S
        check_close(cond.loglik(new_S).sum() - cond.loglik(state.S).sum(),
                    log_p(X, new_state) - log_p(X, state))

    cond = cond_A_transp(X, state)
    for tr in range(5):
        new_A = np.random.normal(0., 1., size=(K, D))
        new_state = state.copy()
        new_state.A = new_A
        check_close(cond.loglik(new_A.T).sum() - cond.loglik(state.A.T).sum(),
                    log_p(X, new_state) - log_p(X, state))


def check_log_f():
    for tr in range(5):
        lam_ev = np.random.gamma(1., 1.)
        h_ev = np.random.normal(0., 1.)
        evidence = gaussians.Potential(np.array([h_ev]), psd_matrices.EyeMatrix(lam_ev, 1), 0.)


        def f1(z):
            # prior over Z
            s_dist = gaussians.Potential.from_moments_iso(np.array([0.]), np.exp(z))
            return (evidence + s_dist).integral().sum()

        f2 = LogFCollapsed(lam_ev, h_ev)

        z1 = np.random.normal(0., 1.)
        z2 = np.random.normal(0., 1.)
        check_close(f1(z1) - f1(z2), f2(z1) - f2(z2))

class DataInfo:
    def __init__(self, X, X_noiseless, A, S, Z, mu_Z, sigma_sq_Z, sigma_sq_N):
        self.data_matrix = observations.DataMatrix.from_real_values(X)
        self.X = X
        self.X_noiseless = X_noiseless
        self.A = A
        self.S = S
        self.Z = Z
        self.mu_Z = mu_Z
        self.sigma_sq_Z = sigma_sq_Z
        self.sigma_sq_N = sigma_sq_N
        self.K, self.D = self.A.shape
        self.N = self.X.shape[0]

def load_toy_data():
    def square(i, j):
        result = np.zeros((12, 12))
        result[i-2:i+3, j-2:j+3] = np.random.normal(size=(5, 5))
        return result
    def cross(i, j):
        result = np.zeros((12, 12))
        result[i, j-2:j+3] = 1.
        result[i-2:i+3, j] = 1.
        return result

    squares = [square(i, j).ravel() for i in range(2, 9, 2) for j in range(2, 9)]
    crosses = [cross(i, j).ravel() for i in range(2, 9, 2) for j in range(2, 9)]
    A = np.array(squares + crosses)
    A /= np.sqrt(np.sum(A**2, axis=1)[:, nax])
    K, D = A.shape
    N = 10000

    mu_Z = -1.
    # temporary
    #sigma_sq_Z = 1.
    sigma_sq_Z = 4.
    sigma_sq_N = 0.1

    Z = np.random.normal(mu_Z, np.sqrt(sigma_sq_Z), size=(N, K))
    S = np.random.normal(0., np.exp(0.5 * Z))
    X_noiseless = np.dot(S, A)
    X = np.random.normal(X_noiseless, np.sqrt(sigma_sq_N))

    return DataInfo(X, X_noiseless, A, S, Z, mu_Z, sigma_sq_Z, sigma_sq_N)
        
def check_hyperparameter_learning():
    info = load_toy_data()
    state = SparseCodingState(info.S, info.A, info.Z, info.sigma_sq_N, info.mu_Z, info.sigma_sq_Z, 1. / info.D)

    state.mu_Z = 5.
    state.sigma_sq_Z = 0.1

    state.S = state.S[:200, :]
    state.Z = state.Z[:200, :]

    Z_true = state.Z.copy()

    print 'mean(Z)', np.mean(state.Z)
    print 'var(Z)', np.var(state.Z)

    mu_Z_all = [state.mu_Z]
    sigma_sq_Z_all = [state.sigma_sq_Z]
    for i in range(20):
        for tr in range(5):
            sample_Z(state)

        #state.mu_Z = cond_mu_Z(state).sample()
        #state.sigma_sq_Z = cond_sigma_sq_Z(state).sample()
        sample_hyperparameters(None, state)
        
        mu_Z_all.append(state.mu_Z)
        sigma_sq_Z_all.append(state.sigma_sq_Z)

        vis.figure('mu_Z')
        pylab.clf()
        pylab.plot(mu_Z_all)
        pylab.title('mu_Z')

        vis.figure('sigma_sq_Z')
        pylab.clf()
        pylab.semilogy(sigma_sq_Z_all)
        pylab.title('sigma_sq_Z')

        pylab.draw()

        print 'mean(Z)', np.mean(state.Z)
        print 'var(Z)', np.var(state.Z)

        for i_ in range(5):
            for j_ in range(5):
                print state.Z[i_, j_], Z_true[i_, j_]


