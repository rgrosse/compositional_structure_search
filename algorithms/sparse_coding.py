import numpy as np
nax = np.newaxis

import slice_sampling
from utils import distributions

debugger = None

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
    
