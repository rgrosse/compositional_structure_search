import numpy as np
nax = np.newaxis
import pylab

import predictive_distributions
from utils import distributions
import variational

SIGMOID_SCHEDULE = True

def p_s_given_z(S, Z, t, sigma_sq_approx):
    return (1. - t) * distributions.gauss_loglik(S, 0., sigma_sq_approx[nax, :]) + \
           t * distributions.gauss_loglik(S, 0., np.exp(Z))

def log_odds_to_prob(log_odds):
    prob = np.exp(log_odds - np.logaddexp.reduce(log_odds, axis=1)[:, nax])
    prob /= prob.sum(1)[:, nax]   # redundant in principle, but numerical error makes np.random.multinomial unhappy
    return prob

def get_schedule(num_steps, first_odds):
    tau = np.linspace(-first_odds, first_odds, num_steps)
    temp = 1. / (1. + np.exp(-tau))
    return (temp - temp[0]) / (temp[-1] - temp[0])

    

class MultinomialSampler:
    def __init__(self, pi, A, Sigma):
        self.pi = pi
        self.A = A
        self.Sigma = Sigma
        self.nlat, nvis = A.shape
        if Sigma.ndim == 2:
            self.Lambda = np.linalg.inv(self.Sigma)[nax, :, :]
        else:
            self.Lambda = np.array([np.linalg.inv(self.Sigma[i, :, :])
                                    for i in range(self.Sigma.shape[0])])

    def random_initialization(self, variational_reps):
        for vr in variational_reps:
            assert isinstance(vr, variational.MultinomialRepresentation)
        return np.array([rep.sample() for rep in variational_reps])

    def step(self, targets, t, U):
        N = targets.shape[0]
        diff = self.A[nax, :, :] - targets[:, nax, :]
        obs_term = -0.5 * np.sum(np.sum(diff[:, :, :, nax] * diff[:, :, nax, :] *
                                        self.Lambda[:, nax, :, :], axis=3), axis=2)
        prob = log_odds_to_prob(obs_term + np.log(self.pi)[nax, :])
        return np.array([np.random.multinomial(1, prob[i, :])
                         for i in range(N)])

    def p_star(self, t, U):
        # constant with respect to time, so it doesn't affect AIS output
        return 0

    def contribution(self, U):
        return np.dot(U, self.A)

class InnerMultinomialSampler:
    def __init__(self, pi, A, sigma_sq_approx):
        self.pi = pi
        self.A = A
        self.sigma_sq_approx = sigma_sq_approx

    def random_initialization(self, N):
        return np.random.multinomial(1, self.pi, size=N)

    def step(self, Z0, S, t, U):
        N, nspars, nclusters = S.shape[0], S.shape[1], self.pi.size
        ev = np.zeros((N, nclusters))
        for k in range(nclusters):
            Z = Z0 + self.A[k, :][nax, :]
            ev[:, k] = p_s_given_z(S, Z, t, self.sigma_sq_approx).sum(1)
        prob = log_odds_to_prob(ev + np.log(self.pi)[nax, :])
        #return np.random.multinomial(1, prob)
        return np.array([np.random.multinomial(1, prob[i, :])
                         for i in range(N)])

    def contribution(self, Z):
        return np.dot(Z, self.A)

class BernoulliSampler:
    def __init__(self, pi, A, Sigma):
        self.pi = pi
        self.A = A
        self.Sigma = Sigma
        self.nlat, nvis = A.shape
        if Sigma.ndim == 2:
            self.Lambda = np.linalg.inv(self.Sigma)[nax, :, :]
        else:
            self.Lambda = np.array([np.linalg.inv(self.Sigma[i, :, :])
                                    for i in range(self.Sigma.shape[0])])

    def random_initialization(self, variational_reps):
        for vr in variational_reps:
            assert isinstance(vr, variational.BernoulliRepresentation)
        return np.array([rep.sample() for rep in variational_reps])

    def step(self, targets, t, U):
        U = U.copy()
        N, K = U.shape
        for i in range(N):
            x = np.dot(U[i, :], self.A)
            curr_targets = targets[i, :]
            if self.Lambda.ndim == 2:
                Lambda = self.Lambda
            else:
                Lambda = self.Lambda[i, :, :]
            
            for k in range(K):
                if U[i, k]:
                    x -= self.A[k, :]
                off_score = -0.5 * np.dot(x - curr_targets, np.dot(Lambda, x - curr_targets))
                x_on = x + self.A[k, :]
                on_score = -0.5 * np.dot(x_on - curr_targets, np.dot(Lambda, x_on - curr_targets))

                log_odds = np.log(self.pi[k]) + on_score - off_score
                prob = 1. / (1. + np.exp(-log_odds))
                U[i, k] = np.random.binomial(1, prob)
                x += U[i, k] * self.A[k, :]

        return U

    def p_star(self, t, u):
        # constant with respect to time, so it doesn't affect AIS output
        return 0

    def contribution(self, Z):
        return np.dot(Z, self.A)

class InnerBernoulliSampler:
    def __init__(self, pi, A, sigma_sq_approx):
        self.pi = pi
        self.A = A
        self.sigma_sq_approx = sigma_sq_approx

    def step(self, Z0, S, t, U):
        U = U.copy()
        ndata, nspars, nfea = S.shape[0], S.shape[1], U.shape[1]
        for i in range(ndata):
            z = Z0[i, :] + np.dot(U[i, :], self.A)

            for k in range(nfea):
                if U[i, k]:
                    z -= self.A[k, :]
                off_score = p_s_given_z(S[i, :], z, t, self.sigma_sq_approx).sum()
                z_on = z + self.A[k, :]
                on_score = p_s_given_z(S[i, :], z_on, t, self.sigma_sq_approx).sum()

                log_odds = np.log(self.pi[k]) + on_score - off_score
                prob = 1. / (1. + np.exp(-log_odds))
                U[i, k] = np.random.binomial(1, prob)
                z += U[i, k] * self.A[k, :]

        return U

    def contribution(self, U):
        return np.dot(U, self.A)

    def random_initialization(self, N):
        return np.random.binomial(1, self.pi[nax, :], size=(N, self.pi.size))

def mh_multivariate_gaussian(U, f, mu, Sigma, epsilon):
    N, K = U.shape
    perturbation = np.array([np.random.multivariate_normal(np.zeros(K), Sigma)
                             for i in range(N)])
    proposal = mu[nax, :] + \
               np.sqrt(1. - epsilon ** 2) * (U - mu[nax, :]) + \
               epsilon * perturbation
    L0 = f(U)
    L1 = f(proposal)
    accept = np.random.binomial(1, np.where(L1 > L0, 1., np.exp(L1 - L0)))
    if np.isscalar(accept): # np.random.binomial converts length 1 arrays to scalars
        accept = np.array([accept])
    return np.where(accept[:, nax], proposal, U)


class InnerGaussianSampler:
    def __init__(self, mu, Sigma, sigma_sq_approx):
        self.mu = mu
        self.Sigma = Sigma
        self.sigma_sq_approx = sigma_sq_approx

    def step(self, Z0, S, t, U):
        EPSILON = 0.5
        N, K = S.shape
        U = U.copy()

        def f(U):
            return p_s_given_z(S, Z0 + U, t, self.sigma_sq_approx).sum(1)

        return mh_multivariate_gaussian(U, f, self.mu, self.Sigma, EPSILON)

        
    def contribution(self, U):
        return U

    def random_initialization(self, N):
        return np.array([np.random.multivariate_normal(self.mu, self.Sigma)
                         for i in range(N)])


class GSMRepresentation:
    def __init__(self, S, U_all):
        self.S = S
        self.U_all = U_all

    def copy(self):
        return GSMRepresentation(self.S.copy(), self.U_all[:])

class GSMSampler:
    def __init__(self, scale_samplers, sigma_sq_approx, evidence_Sigma, A):
        self.scale_samplers = scale_samplers
        self.sigma_sq_approx = sigma_sq_approx
        self.evidence_Sigma = evidence_Sigma
        if evidence_Sigma.ndim == 2:
            self.evidence_Lambda = np.linalg.inv(evidence_Sigma)
        else:
            self.evidence_Lambda = np.array([np.linalg.inv(evidence_Sigma[i, :, :])
                                             for i in range(evidence_Sigma.shape[0])])
        self.A = A

    def random_initialization(self, S):
        N = S.shape[0]
        U_all = [sampler.random_initialization(N) for sampler in self.scale_samplers]
        return GSMRepresentation(S.copy(), U_all)

    def step(self, targets, t, rep):
        N, D = targets.shape
        K = rep.S.shape[1]
        rep = rep.copy()

        
        # sample S
        if self.evidence_Lambda.ndim == 2:
            Lambda_ev = np.dot(self.A, np.dot(self.evidence_Lambda, self.A.T))
        else:
            Lambda_ev = np.array([np.dot(self.A, np.dot(self.evidence_Lambda[i, :, :], self.A.T))
                                  for i in range(N)])
        h_ev = np.dot(self.A, np.dot(self.evidence_Lambda, targets.T)).T
        
        Z = np.zeros((N, K))
        for comp, samp in zip(rep.U_all, self.scale_samplers):
            Z += samp.contribution(comp)
        #sigma_sq_pri = np.exp((1. - t) * np.log(self.sigma_sq_approx)[nax, :] +
        #                      t * Z)
        lam_pri = (1. - t) / self.sigma_sq_approx[nax, :] + \
                  t * np.exp(-Z)



        rep.S = np.zeros((N, K))
        for i in range(N):
            Lambda_pri = np.diag(lam_pri[i, :])
            if Lambda_ev.ndim == 2:
                Lambda = Lambda_pri + Lambda_ev
            else:
                Lambda = Lambda_pri + Lambda_ev[i, :, :]
            Sigma = np.linalg.inv(Lambda)
            mu = np.dot(Sigma, h_ev[i, :])
            rep.S[i, :] = np.random.multivariate_normal(mu, Sigma)

        # sample components of Z
        rep = rep.copy()
        for c in range(len(self.scale_samplers)):
            Z0 = np.zeros(Z.shape)
            for d in range(len(self.scale_samplers)):
                if d != c:
                    Z0 += self.scale_samplers[d].contribution(rep.U_all[d])
            
            rep.U_all[c] = self.scale_samplers[c].step(Z0, rep.S, t, rep.U_all[c])

        # temporary
        #assert not stop

        return rep

    def p_star(self, t, rep):
        N, K = rep.S.shape
        Z = np.zeros((N, K))
        for comp, samp in zip(rep.U_all, self.scale_samplers):
            Z += samp.contribution(comp)
        return p_s_given_z(rep.S, Z, t, self.sigma_sq_approx).sum(1)

    def contribution(self, rep):
        return np.dot(rep.S, self.A)

# temporary
stop = False

class AISModel:
    def __init__(self, samplers, X, Sigma, init_partition_function):
        self.samplers = samplers
        self.X = X
        self.Sigma = Sigma
        self.init_partition_function = init_partition_function

    def step(self, reps, t):
        reps = reps[:]
        for i in range(len(self.samplers)):
            targets = self.X.copy()
            for j in range(len(self.samplers)):
                if j != i:
                    targets -= self.samplers[j].contribution(reps[j])
            reps[i] = self.samplers[i].step(targets, t, reps[i])
        return reps

    def init_sample(self, variational_reps):
        N, D = self.X.shape
        is_gsm = np.array([isinstance(s, GSMSampler) for s in self.samplers])
        gsm_idxs = np.where(is_gsm)[0]
        non_gsm_idxs = np.where(-is_gsm)[0]

        if len(gsm_idxs) == 0:
            raise RuntimeError('No GSM components; problem with module reloading?')
        if len(gsm_idxs) > 1:
            raise RuntimeError('Cannot handle multiple GSM components yet')
        gsm_sampler = self.samplers[gsm_idxs[0]]
            
        reps = [None for i in range(len(self.samplers))]
        discrete_part = np.zeros((N, D))
        for vr_idx, sampler_idx in enumerate(non_gsm_idxs):
            curr_reps = [vr[vr_idx] for vr in variational_reps]
            reps[sampler_idx] = self.samplers[sampler_idx].random_initialization(curr_reps)
            discrete_part += self.samplers[sampler_idx].contribution(reps[sampler_idx])

        # S = coefficients
        # G = GSM part = SA
        # E = Gaussian part
        # C = continuous part = S + E
        A = gsm_sampler.A
        C = self.X - discrete_part
        Sigma_S = np.diag(gsm_sampler.sigma_sq_approx)
        Sigma_E = self.Sigma
        Sigma_C = np.dot(A.T, np.dot(Sigma_S, A)) + Sigma_E
        Sigma_C_inv = np.linalg.inv(Sigma_C)
        temp = np.dot(Sigma_S, A)
        mu_S_given_C = np.dot(temp, np.dot(Sigma_C_inv, C.T)).T
        Sigma_S_given_C = np.dot(temp, np.dot(Sigma_C_inv, temp.T))
        S = np.array([np.random.multivariate_normal(mu_S_given_C[i, :], Sigma_S_given_C)
                      for i in range(N)])

        reps[gsm_idxs[0]] = gsm_sampler.random_initialization(S)

        return reps

    def p_star(self, reps, t):
        total = 0.
        for sampler, rep in zip(self.samplers, reps):
            total += sampler.p_star(t, rep)
        return total

def ais(ais_model, t_schedule, variational_representations):
    init_partition_function = ais_model.init_partition_function
    total = init_partition_function.copy()
    reps = ais_model.init_sample(variational_representations)

    all_deltas = []

    count = 0
    for t0, t1 in zip(t_schedule[:-1], t_schedule[1:]):
        if count == 1:
            for i in range(100):
                reps = ais_model.step(reps, t0)
        else:
            reps = ais_model.step(reps, t0)
        delta = ais_model.p_star(reps, t1) - ais_model.p_star(reps, t0)

        # temporary
        if count > 0:
            total += delta
        
        all_deltas.append(delta)

        # temporary
        global stop
        if delta > 1.:
            stop = True

        count += 1

    return total

# __init__(self, scale_samplers, sigma_sq_approx, evidence_Sigma):
def compute_likelihood(X, components, Sigma, variational_representations, init_partition_function,
                       t_schedule=None, num_steps=1000):

    samplers = []
    for comp in components:
        if isinstance(comp, predictive_distributions.MultinomialPredictiveDistribution):
            sampler = MultinomialSampler(comp.pi, comp.centers, Sigma)
            
        elif isinstance(comp, predictive_distributions.BernoulliPredictiveDistribution):
            sampler = BernoulliSampler(comp.pi, comp.A, Sigma)
            
        elif isinstance(comp, predictive_distributions.GSMPredictiveDistribution):
            inner_samplers = []
            for sc in comp.scale_components:
                if isinstance(sc, predictive_distributions.MultinomialPredictiveDistribution):
                    inner_sampler = InnerMultinomialSampler(sc.pi, sc.centers, comp.sigma_sq_approx)
                    
                elif isinstance(sc, predictive_distributions.BernoulliPredictiveDistribution):
                    inner_sampler = InnerBernoulliSampler(sc.pi, sc.A, comp.sigma_sq_approx)
                    
                else:
                    raise RuntimeError("Can't convert to inner sampler: %s" % sc.__class__)
                
                inner_samplers.append(inner_sampler)

            igs = InnerGaussianSampler(comp.scale_mu, comp.scale_Sigma, comp.sigma_sq_approx)
            inner_samplers.append(igs)

            sampler = GSMSampler(inner_samplers, comp.sigma_sq_approx, Sigma,
                                 comp.A)

        samplers.append(sampler)

    ais_model = AISModel(samplers, X, Sigma, init_partition_function)
            
    if t_schedule is None:
        if SIGMOID_SCHEDULE:
            t_schedule = get_schedule(num_steps, 10.)
        else:
            t_schedule = np.linspace(0., 1., num_steps)

    return ais(ais_model, t_schedule, variational_representations)



