import numpy as np
nax = np.newaxis
import scipy.optimize

import grammar
import models
import slice_sampling
import sparse_coding
from utils import distributions, misc


def sample_variance(node):
    if node.isleaf() and node.distribution() == 'g':
        node.sample_variance()
    elif node.issum():
        for child in node.children:
            sample_variance(child)
    elif node.isproduct():
        for child in node.children:
            sample_variance(child)





class GenericGibbsSampler:
    def __init__(self, node):
        self.node = node

    def step(self, niter=1, maximize=False):
        if not maximize:
            self.node.gibbs_update2()

    def __str__(self):
        return 'GenericGibbsSampler(%d)' % self.node.model.id

    def preserves_root_value(self):
        return True


class GaussianSampler:
    def __init__(self, product_node, noise_node, side, maximize):
        self.product_node = product_node
        self.noise_node = noise_node
        self.side = side
        self.maximize = maximize

    def step(self):
        left, right = self.product_node.children
        m, n = self.noise_node.m, self.noise_node.n

        old = np.dot(left.value(), right.value()) + self.noise_node.value()

        if self.side == 'left' and ((left.isleaf() and left.distribution() == 'g')
                                    or left.isgsm()):
            A = np.eye(m)
            B = right.value()
            X_node = left
        elif self.side == 'left' and left.issum():
            A = np.eye(m)
            B = right.value()
            X_node = left.children[-1]
        elif self.side == 'right' and ((right.isleaf() and right.distribution() == 'g')
                                       or right.isgsm()):
            A = left.value()
            B = np.eye(n)
            X_node = right
        elif self.side == 'right' and right.issum():
            A = left.value()
            B = np.eye(n)
            X_node = right.children[-1]
        else:
            raise RuntimeError("shouldn't get here")

        X = X_node.value()
        N_node = self.noise_node
        N = N_node.value()
        C = np.dot(np.dot(A, X), B) + N
        obs = np.ones((m, n), dtype=bool)

        if X_node.has_rank1_variance() and N_node.has_rank1_variance():
            ssq_row_N, ssq_col_N = N_node.row_col_variance()
            ssq_row_X, ssq_col_X = X_node.row_col_variance()
            d_1 = 1. / np.sqrt(ssq_row_N)
            d_2 = 1. / np.sqrt(ssq_col_N)
            d_3 = 1. / np.sqrt(ssq_row_X)
            d_4 = 1. / np.sqrt(ssq_col_X)

            if self.maximize:
                X_new = misc.map_gaussian_matrix_em(A, B, C, d_1, d_2, d_3, d_4, obs, X)
            else:
                X_new = misc.sample_gaussian_matrix_em(A, B, C, d_1, d_2, d_3, d_4, obs, X)

        else:
            if self.maximize:
                fn = misc.map_gaussian_matrix2
            else:
                fn = misc.sample_gaussian_matrix2

            if self.side == 'left':
                X_new = fn(B.T, C.T, 1. / X_node.variance().T, obs.T / N_node.variance().T).T
            else:
                X_new = fn(A, C, 1. / X_node.variance(), obs / N_node.variance())


        X_node.set_value(X_new)
        N_new = C - np.dot(np.dot(A, X_new), B)
        N_node.set_value(N_new)

        new = np.dot(left.value(), right.value()) + self.noise_node.value()
        assert np.allclose(old, new)

    def __str__(self):
        return 'GaussianSampler(prod=%d, noise=%d, side=%s, maximize=%s)' % \
               (self.product_node.model.id, self.noise_node.model.id, self.side, self.maximize)

    def preserves_root_value(self):
        return True


class LatentValueSampler:
    def __init__(self, data_matrix, node):
        self.data_matrix = data_matrix
        self.node = node

    def step(self):
        pred = self.node.value() - self.node.children[-1].value()
        new_X = self.data_matrix.sample_latent_values(pred, self.node.children[-1].variance())
        self.node.children[-1].set_value(new_X - pred)

    def __str__(self):
        return 'LatentValueSampler(%d)' % self.node.model.id

    def preserves_root_value(self):
        return False

    

class LatentValueMaximizer:
    def __init__(self, data_matrix, node):
        self.data_matrix = data_matrix
        self.node = node

    def step(self):
        pred = self.node.value() - self.node.children[-1].value()
        new_X = np.where(self.data_matrix.observations.mask, self.node.value(), pred)
        self.node.children[-1].set_value(new_X - pred)

    def __str__(self):
        return 'LatentValueMaximizer(%d)' % self.node.model.id

    def preserves_root_value(self):
        return False

class VarianceSampler:
    def __init__(self, node):
        self.node = node

    def step(self):
        self.node.sample_variance()

    def __str__(self):
        return 'VarianceSampler(%d)' % self.node.model.id

    def preserves_root_value(self):
        return True

class GSMScaleSampler:
    def __init__(self, gsm_node, maximize=False):
        self.gsm_node = gsm_node
        self.maximize = maximize

    def step(self):
        # S ~ N(0, exp(Z / 2))
        #
        # Z = signal_node + noise_node + bias
        #   = signal_node +   gaussian_term
        #   =       scale_node         + bias
        #
        # resample gaussian_term conditioned on signal_node
         
        scale_node = self.gsm_node.scale_node
        S = self.gsm_node.value()
        N, K = S.shape

        # resample Z
        Z = scale_node.value() + self.gsm_node.bias
        if scale_node.isleaf():
            mu = self.gsm_node.bias * np.ones((N, K))
            sigma_sq = scale_node.variance()
        else:
            assert scale_node.issum()
            mu = self.gsm_node.bias + scale_node.value() - scale_node.children[-1].value()
            sigma_sq = scale_node.children[-1].variance()

        for i in range(N):
            for k in range(K):
                log_f = sparse_coding.LogFUncollapsed(S[i, k])
                if self.maximize:
                    temp = lambda z: -log_f(z) - distributions.gauss_loglik(z, mu[i, k], sigma_sq[i, k])
                    Z[i, k] = scipy.optimize.fmin(temp, Z[i, k], disp=False)
                else:
                    Z[i, k] = slice_sampling.slice_sample_gauss(log_f, mu[i, k], sigma_sq[i, k], Z[i, k])

        # resample bias
        if scale_node.isleaf():
            gaussian_term = Z
        else:
            signal = scale_node.value() - scale_node.children[-1].value()
            gaussian_term = Z - signal

        if not self.maximize:
            if self.gsm_node.bias_type == 'scalar':
                mu = gaussian_term.mean()
                lam = (1. / sigma_sq).sum()
                self.gsm_node.bias = np.random.normal(mu, 1. / lam)
            elif self.gsm_node.bias_type == 'row':
                mu = gaussian_term.mean(1)
                lam = (1. / sigma_sq).sum(1)
                self.gsm_node.bias = np.random.normal(mu, 1. / lam)[:, nax]
            elif self.gsm_node.bias_type == 'col':
                mu = gaussian_term.mean(0)
                lam = (1. / sigma_sq).sum(0)
                self.gsm_node.bias = np.random.normal(mu, 1. / lam)[nax, :]

        # set noise node
        noise_term = gaussian_term - self.gsm_node.bias
        if scale_node.isleaf():
            scale_node.set_value(noise_term)
        else:
            scale_node.children[-1].set_value(noise_term)
    
    def __str__(self):
        return 'GSMScaleSampler(%d, maximize=%s)' % (self.gsm_node.model.id, self.maximize)

    def preserves_root_value(self):
        return True


def get_samplers(data_matrix, node, maximize):
    samplers = []
    if data_matrix is not None and not maximize:
        samplers.append(LatentValueSampler(data_matrix, node))
    if data_matrix is not None and maximize:
        samplers.append(LatentValueMaximizer(data_matrix, node))
        
    if node.isleaf() and not node.model.fixed and not maximize:
        samplers.append(GenericGibbsSampler(node))
        
    if node.isleaf() and node.distribution() == 'g' and not node.model.fixed_variance and not maximize:
        samplers.append(VarianceSampler(node))
        
    if node.issum():
        children = node.children[:-1]
        noise_node = node.children[-1]
        for child in children:
            left, right = child.children
            if ((left.isleaf() and left.distribution() == 'g') or left.issum() or left.isgsm()) and \
                   not left.model.fixed and not noise_node.model.fixed:
                samplers.append(GaussianSampler(child, noise_node, 'left', maximize))
            if ((right.isleaf() and right.distribution() == 'g') or right.issum() or left.isgsm()) and \
                   not right.model.fixed and not noise_node.model.fixed:
                samplers.append(GaussianSampler(child, noise_node, 'right', maximize))

    if node.isgsm():
        samplers.append(GSMScaleSampler(node, maximize=maximize))
                
    for child in node.children:
        samplers += get_samplers(None, child, maximize)
        
    return samplers

def list_samplers(model, maximize=False):
    node = model.dummy()
    models.align(node, model)
    samplers = get_samplers('dummy', node, maximize)
    node.model.display()
    print
    for s in samplers:
        print s


def sweep(data_matrix, root, num_iter=100, maximize=False):
    samplers = get_samplers(data_matrix, root, maximize)

    if num_iter > 1:
        print 'Dumb Gibbs sampling on %s...' % grammar.pretty_print(root.structure())
        pbar = misc.pbar(num_iter)
    else:
        pbar = None
        
    for it in range(num_iter):
        for sampler in samplers:
            if sampler.preserves_root_value():
                old = root.value()
            sampler.step()
            if sampler.preserves_root_value():
                assert np.allclose(old, root.value())

        if pbar is not None:
            pbar.update(it)
    if pbar is not None:
        pbar.finish()
        
