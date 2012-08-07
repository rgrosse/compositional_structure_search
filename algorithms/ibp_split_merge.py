import numpy as np
nax = np.newaxis
import scipy.special

import ibp
import observations
from utils import distributions, gaussians, psd_matrices

def poisson(k, lam):
    return -lam * k * np.log(lam) - scipy.special.gammaln(k+1)

def evidence(model, data, state):
    K, D = state.Z.shape[1], state.X.shape[1]

    Lambda = np.dot(state.Z.T, state.Z) / state.sigma_sq_n + np.eye(K) / state.sigma_sq_f
    h = np.dot(state.Z.T, state.X) / state.sigma_sq_n

    # we can ignore the constant factors because they don't depend on Z
    pot = gaussians.Potential(h.T, psd_matrices.FullMatrix(Lambda[nax, :, :]), 0.)
    return pot.integral().sum()

def sample_features(model, data, state):
    K, D = state.Z.shape[1], state.X.shape[1]

    Lambda = np.dot(state.Z.T, state.Z) / state.sigma_sq_n + np.eye(K) / state.sigma_sq_f
    h = np.dot(state.Z.T, state.X) / state.sigma_sq_n

    # we can ignore the constant factors because they don't depend on Z
    pot = gaussians.Potential(h.T, psd_matrices.FullMatrix(Lambda[nax, :, :]), 0.)
    return pot.to_distribution().sample().T


def next_assignment_proposal(model, data, state, cache, Sigma_info, i, k):
    assert not cache.rows_included[i]
    x = state.X[i, :]

    evidence = np.zeros(2)
    for assignment in [0, 1]:
        mu = Sigma_info.mu_for(k, assignment)
        ssq = Sigma_info.sigma_sq_for(k, assignment) + state.sigma_sq_n
        evidence[assignment] = ibp.gauss_loglik_vec_C2(x, mu, ssq)
    data_odds = evidence[1] - evidence[0]

    if cache.counts[k] > 0:
        prior_odds = np.log(cache.counts[k]) - np.log(cache.num_included - cache.counts[k] + 1)
    else:
        #prior_odds = poisson(1, 0.5 * model.alpha / (i+1)) - poisson(0, 0.5 * model.alpha / (i+1))
        prior_odds = np.log(model.alpha) - np.log(cache.num_included + 1)

    return distributions.BernoulliDistribution.from_odds(data_odds + prior_odds)
    

def propose_assignments(model, data, state, update=False):
    """Generate the proposal for K columns using sequential Monte Carlo. Assumes the remaining
    features have been sampled, the remaining assignments are fixed, and the other features' contributions
    are subtracted from the data matrix. Generally K = 2."""
    N, K = state.Z.shape
    state = state.copy()
    cache = ibp.IBPCache.from_state(model, data, state, np.zeros(N, dtype=bool))

    proposal_prob = 0.

    for i in range(N):
        Sigma_info = cache.fpost.Sigma_info(np.zeros(K, dtype=int))
        for k in range(K):
            cond = next_assignment_proposal(model, data, state, cache, Sigma_info, i, k)
            if update:
                state.Z[i, k] = cond.sample()
            proposal_prob += cond.loglik(state.Z[i, k])
            Sigma_info.update(k, state.Z[i, k])
        cache.add(i, state.Z[i, :], state.X[i, :])

    return state, proposal_prob

CHOICES = [(0, 0), (0, 1), (1, 0), (1, 1)]

def propose_assignments2(model, data, state, update=False):
    N, K = state.Z.shape
    state = state.copy()
    cache = ibp.IBPCache.from_state(model, data, state, np.zeros(N, dtype=bool))

    proposal_prob = 0.

    for i in range(N):
        obs = data.mask[i, :]
        x = state.X[i, :]
        
        evidence = np.zeros(4)
        prior_odds = np.zeros(4)
        for c, (z1, z2) in enumerate(CHOICES):
            z = np.array([z1, z2])
            mu = cache.fpost.predictive_mu(z)
            ssq = cache.fpost.predictive_ssq(z) + state.sigma_sq_n
            evidence[c] = ibp.gauss_loglik_vec_C2(x[obs], mu[obs], ssq)

            for k in [0, 1]:
                if cache.counts[k] > 0:
                    prior_odds[c] += np.log(cache.counts[k]) - np.log(cache.num_included - cache.counts[k] + 1)
                else:
                    prior_odds[c] += np.log(model.alpha) - np.log(cache.num_included + 1)

        odds = evidence + prior_odds
        dist = distributions.MultinomialDistribution.from_odds(odds)
        if update:
            state.Z[i, :] = CHOICES[dist.sample().argmax()]
        proposal_prob += dist.loglik(CHOICES.index(tuple(state.Z[i, :])))
        cache.add(i, state.Z[i, :], state.X[i, :])

        assert np.isfinite(proposal_prob)

    return state, proposal_prob
        

def ibp_loglik(Z, alpha):
    N = Z.shape[0]
    idxs = np.where(Z.any(0))[0]
    K = idxs.size
    
    total = -alpha * (1. / np.arange(1, N+1)).sum()
    total += alpha * K

    if K > 0:
        m = Z[:, idxs].sum(0)
        total += scipy.special.gammaln(N - m + 1).sum()
        total += scipy.special.gammaln(m).sum()
        total -= K * scipy.special.gammaln(N + 1)

    assert np.isfinite(total)
    
    return total


def choose_columns(K):
    if np.random.binomial(1, 0.5):
        k1 = 'new'
    else:
        k1 = np.random.randint(0, K)

    if np.random.binomial(1, 0.5):
        k2 = 'new'
    else:
        k2 = np.random.randint(0, K)
        if k2 == k1:
            k2 = 'new'

    return k1, k2

def column_probability(K, k1, k2):
    total = 0.
    if k1 == 'new':
        total += np.log(0.5)
    else:
        total += np.log(0.5) - np.log(K)

    if k2 == 'new':
        total += np.log(0.5 + 0.5 / K)
    else:
        assert k1 != k2
        total += np.log(0.5) - np.log(K)

    return total


def backward_move_info(K_orig, k1, k2, new_reduced_state):
    any_ones = new_reduced_state.Z.any(0)

    K_back = K_orig
    if k1 == 'new':
        K_back += 1
    if k2 == 'new':
        K_back += 1
    if not any_ones[0]:
        K_back -= 1
    if not any_ones[1]:
        K_back -= 1
    
    if any_ones[0]:
        k1_back = 0
    else:
        k1_back = 'new'

    if any_ones[1]:
        k2_back = 1
    else:
        k2_back = 'new'

    return K_back, k1_back, k2_back


def split_merge_step(model, data, state):
    N, K, D = state.X.shape[0], state.Z.shape[1], state.X.shape[1]

    if K <= 2:
        return    # this case is awkward to deal with, and if it occurs, the model probably isn't too good anyway

    # choose random columns
    k1, k2 = choose_columns(K)

    # generate reduced problem
    prod = np.zeros(state.X.shape)
    for k in range(K):
        if k not in (k1, k2):
            prod += np.outer(state.Z[:, k], state.A[k, :])
    reduced_data = observations.RealObservations(state.X - prod, np.ones(state.X.shape, dtype=bool))
    reduced_X = state.X - prod
    reduced_state = ibp.CollapsedIBPState(reduced_X, np.zeros((N, 2), dtype=int), state.sigma_sq_f, state.sigma_sq_n)
    if k1 != 'new':
        reduced_state.Z[:, 0] = state.Z[:, k1]
    if k2 != 'new':
        reduced_state.Z[:, 1] = state.Z[:, k2]

    # propose assignments
    new_reduced_state, forward_prob = propose_assignments2(model, reduced_data, reduced_state, True)
    forward_prob += column_probability(K, k1, k2)

    # score the states
    old_score = ibp_loglik(reduced_state.Z, model.alpha) + evidence(model, reduced_data, reduced_state)
    new_score = ibp_loglik(new_reduced_state.Z, model.alpha) + evidence(model, reduced_data, new_reduced_state)

    # backward proposal probability
    K_back, k1_back, k2_back = backward_move_info(K, k1, k2, new_reduced_state)
    backward_prob = column_probability(K_back, k1_back, k2_back)
    _, proposal_prob = propose_assignments2(model, reduced_data, reduced_state, False)
    backward_prob += proposal_prob

    mh_score = new_score - old_score + backward_prob - forward_prob
    if mh_score > 0.:
        acceptance_prob = 1.
    else:
        acceptance_prob = np.exp(mh_score)

    accept = np.random.binomial(1, acceptance_prob)

    if accept:
        A = sample_features(model, reduced_data, new_reduced_state)
        
        if k1 == 'new':
            if np.any(new_reduced_state.Z[:, 0] > 0):
                state.Z = np.hstack([state.Z, new_reduced_state.Z[:, 0][:, nax]])
                state.A = np.vstack([state.A, A[0, :][nax, :]])
        else:
            state.Z[:, k1] = new_reduced_state.Z[:, 0]
            state.A[k1, :] = A[0, :]

        if k2 == 'new':
            if np.any(new_reduced_state.Z[:, 1] > 0):
                state.Z = np.hstack([state.Z, new_reduced_state.Z[:, 1][:, nax]])
                state.A = np.vstack([state.A, A[1, :][nax, :]])
        else:
            state.Z[:, k2] = new_reduced_state.Z[:, 1]
            state.A[k2, :] = A[1, :]

    else:
        pass


    
