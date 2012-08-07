import numpy as np
nax = np.newaxis

from utils import misc

class PredictiveDistribution:
    def __slice__(self, slc):
        return self.__getitem__(slc)


class GaussianPredictiveDistribution(PredictiveDistribution):
    def __init__(self, mu, Sigma):
        self.mu = mu.copy()
        self.Sigma = Sigma.copy()

    def __getitem__(self, slc):
        return GaussianPredictiveDistribution(self.mu, self.Sigma)

    def generate_data(self, N):
        return np.array([np.random.multivariate_normal(self.mu, self.Sigma)
                         for i in range(N)])

class MultinomialPredictiveDistribution(PredictiveDistribution):
    def __init__(self, pi, centers):
        self.pi = pi.copy()
        self.centers = centers.copy()

    def __getitem__(self, slc):
        return MultinomialPredictiveDistribution(self.pi, self.centers[:, slc])

    @staticmethod
    def random(K, N):
        pi = np.random.uniform(0., 1., size=K)
        pi /= pi.sum()
        centers = np.random.normal(size=(K, N))
        return MultinomialPredictiveDistribution(pi, centers)

    def generate_data(self, N):
        Z = np.random.multinomial(1, self.pi, size=N)
        return np.dot(Z, self.centers)

class BernoulliPredictiveDistribution(PredictiveDistribution):
    def __init__(self, pi, A):
        self.pi = pi.copy()
        self.A = A.copy()

    def __getitem__(self, slc):
        return BernoulliPredictiveDistribution(self.pi, self.A[:, slc])

    @staticmethod
    def random(K, N):
        pi = np.random.uniform(0., 1., size=K)
        A = np.random.normal(size=(K, N))
        return BernoulliPredictiveDistribution(pi, A)

    def generate_data(self, N):
        Z = np.random.binomial(1, self.pi[nax, :], size=(N, self.pi.size))
        return np.dot(Z, self.A)


class PredictiveInfo:
    def __init__(self, components, mu, Sigma):
        self.components = components
        self.mu = mu
        self.Sigma = Sigma

    def predictive_for_row(self, i, idxs):
        components = [c[idxs] for c in self.components]
        if self.mu.ndim == 2:
            return components, self.mu[i, idxs], self.Sigma[i, :, :][idxs[:, nax], idxs[nax, :]]
        else:
            assert self.mu.ndim == 1
            return components, self.mu[idxs], self.Sigma[idxs[:, nax], idxs[nax, :]]

    def predictive_for_rows(self, rows):
        if self.mu.ndim == 1:
            N, D = rows.size, self.mu.size
            return self.components, np.tile(self.mu[nax, :], (N, 1)), np.tile(self.Sigma[nax, :, :], (N, 1, 1))
        else:
            return self.components, self.mu[rows], self.Sigma[rows, :, :]

    def generate_data(self, N):
        D = self.Sigma.shape[0]
        X = np.zeros((N, D))
        for c in self.components:
            X += c.generate_data(N)
        X += np.array([np.random.multivariate_normal(self.mu, self.Sigma)
                       for i in range(N)])
        return X

class GSMPredictiveDistribution(PredictiveDistribution):
    def __init__(self, scale_components, scale_mu, scale_Sigma, sigma_sq_approx, A):
        self.scale_components = scale_components
        self.scale_mu = scale_mu
        self.scale_Sigma = scale_Sigma
        self.sigma_sq_approx = sigma_sq_approx
        self.A = A.copy()

    def __getitem__(self, slc):
        return GSMPredictiveDistribution(self.scale_components, self.scale_mu, self.scale_Sigma,
                                         self.sigma_sq_approx, self.A[:, slc])

    def generate_data(self, N):
        K, D = self.A.shape
        Z = np.zeros((N, K))
        for sc in self.scale_components:
            Z += sc.generate_data(N)
        Z += np.array([np.random.multivariate_normal(self.scale_mu, self.scale_Sigma)
                                  for i in range(N)])
        S = np.random.normal(0., np.exp(0.5 * Z))
        return np.dot(S, self.A)





######################## computing the predictive distributions ################
        
class FixedTerm:
    def __init__(self, values):
        self.values = values

class GaussianTerm:
    def __init__(self, values, mu, Sigma):
        self.values = values
        self.mu = mu
        self.Sigma = Sigma

class ChainTerm:
    def __init__(self, values, mu_delta, Sigma_delta):
        self.values = values
        self.mu_delta = mu_delta
        self.Sigma_delta = Sigma_delta

def extract_terms(node):
    if node.isleaf():
        assert node.distribution() in ['g', 'm', 'b']
        if node.distribution() == 'g':
            mu = np.zeros(node.n)
            sigma_sq_row, sigma_sq_col = node.row_col_variance()
            Sigma = np.diag(sigma_sq_row.mean() * sigma_sq_col)
            return [GaussianTerm(node.value(), mu, Sigma)]
        else:
            return [FixedTerm(node.value())]
        
    elif node.issum():
        child_terms = [extract_terms(child) for child in node.children]
        return reduce(list.__add__, child_terms)

    elif node.isgsm():
        return [FixedTerm(node.value())]

    elif node.isproduct():
        left, right = node.children
        
        if left.isleaf() and left.distribution() == 'c':
            child_terms = extract_terms(right)
            terms = []
            for ct in child_terms:
                if isinstance(ct, FixedTerm):
                    # fixed terms inside chains remain fixed
                    terms.append(FixedTerm(ct.values.cumsum(0)))
                elif isinstance(ct, GaussianTerm):
                    # Gaussians become chains
                    terms.append(ChainTerm(ct.values.cumsum(0), ct.mu, ct.Sigma))
                elif isinstance(ct, ChainTerm):
                    # freeze nested chains since these are annoying
                    terms.append(FixedTerm(ct.values.cumsum(0)))
                else:
                    raise RuntimeError('Unknown term')
            return terms

        else:
            child_terms = extract_terms(left)
            V = right.value()
            terms = []
            for ct in child_terms:
                # same distribution, but multiplied by V on the right
                if isinstance(ct, FixedTerm):
                    terms.append(FixedTerm(np.dot(ct.values, V)))
                elif isinstance(ct, GaussianTerm):
                    mu = np.dot(ct.mu, V)
                    Sigma = np.dot(V.T, np.dot(ct.Sigma, V))
                    terms.append(GaussianTerm(np.dot(ct.values, V), mu, Sigma))
                elif isinstance(ct, ChainTerm):
                    mu = np.dot(ct.mu_delta, V)
                    Sigma = np.dot(V.T, np.dot(ct.Sigma_delta, V))
                    terms.append(ChainTerm(np.dot(ct.values, V), mu, Sigma))
                else:
                    raise RuntimeError('Unknown term')
            return terms

def collect_terms(terms):
    fixed_values = 0.
    gaussian_values = 0.
    gaussian_mu = 0.
    gaussian_Sigma = 0.
    chain_values = 0.
    chain_mu = 0.
    chain_Sigma = 0.
    has_fixed = has_gaussian = has_chain = False

    for term in terms:
        if isinstance(term, FixedTerm):
            fixed_values += term.values
            has_fixed = True
        elif isinstance(term, GaussianTerm):
            gaussian_values += term.values
            gaussian_mu += term.mu
            gaussian_Sigma += term.Sigma
            has_gaussian = True
        elif isinstance(term, ChainTerm):
            chain_values += term.values
            chain_mu += term.mu_delta
            chain_Sigma += term.Sigma_delta
            has_chain = True
        else:
            raise RuntimeError('Unknown term')

    if has_fixed:
        fixed_term = FixedTerm(fixed_values)
    else:
        fixed_term = None

    if has_gaussian:
        gaussian_term = GaussianTerm(gaussian_values, gaussian_mu, gaussian_Sigma)
    else:
        gaussian_term = None
        
    if has_chain:
        chain_term = ChainTerm(chain_values, chain_mu, chain_Sigma)
    else:
        chain_term = None

    return fixed_term, gaussian_term, chain_term


def compute_gaussian_part(training_data_matrix, root, N):
    fixed_term, gaussian_term, chain_term = collect_terms(extract_terms(root))
    assert gaussian_term is not None
    D = gaussian_term.values.shape[1]

    if chain_term is None:
        return gaussian_term.mu, gaussian_term.Sigma

    X = training_data_matrix.sample_latent_values(root.predictions(), root.children[-1].sigma_sq)

    mu_0 = np.zeros(D)
    Sigma_0 = 1000. * np.eye(D)
    A = np.eye(D)
    mu_v = chain_term.mu_delta
    Sigma_v = chain_term.Sigma_delta
    B = np.eye(D)

    Lambda_n = np.zeros((D, D, N))
    y = np.zeros((D, N))
    Lambda = np.linalg.inv(gaussian_term.Sigma)
    for i, row in enumerate(training_data_matrix.row_ids):
        Lambda_n[:, :, row] = Lambda
        if fixed_term is not None:
            y[:, row] = X[i, :] - gaussian_term.mu - fixed_term.values[i, :]
        else:
            y[:, row] = X[i, :] - gaussian_term.mu

    mu_chains, Sigma_chains = misc.kalman_filter(mu_0, Sigma_0, A, mu_v, Sigma_v, B, Lambda_n, y)
    mu_total = mu_chains.T + gaussian_term.mu[nax, :]
    Sigma_total = np.zeros((N, D, D))
    for i in range(N):
        Sigma_total[i, :, :] = Sigma_chains[:, :, i] + gaussian_term.Sigma

    return mu_total, Sigma_total
    


def extract_non_gaussian_part(node):
    if node.isleaf():
        assert node.distribution() in ['g', 'm', 'b']
        if node.distribution() == 'g':
            return []
        elif node.distribution() == 'm':
            pi = (1. + node.value().sum(0)) / (node.n + node.m)
            return [MultinomialPredictiveDistribution(pi, np.eye(node.n))]
        elif node.distribution() == 'b':
            pi = (1. + node.value().sum(0)) / (2. + node.m)
            return [BernoulliPredictiveDistribution(pi, np.eye(node.n))]

    elif node.issum():
        child_components = [extract_non_gaussian_part(child) for child in node.children]
        return reduce(list.__add__, child_components)

    elif node.isproduct():
        left, right = node.children

        if left.isleaf() and left.distribution() == 'c':
            return []

        else:
            child_components = extract_non_gaussian_part(left)
            components = []
            for cp in child_components:
                if isinstance(cp, MultinomialPredictiveDistribution):
                    components.append(MultinomialPredictiveDistribution(cp.pi, np.dot(cp.centers, right.value())))
                elif isinstance(cp, BernoulliPredictiveDistribution):
                    components.append(BernoulliPredictiveDistribution(cp.pi, np.dot(cp.A, right.value())))
                elif isinstance(cp, GSMPredictiveDistribution):
                    components.append(GSMPredictiveDistribution(cp.scale_components, cp.scale_mu, cp.scale_Sigma,
                                                                cp.sigma_sq_approx, np.dot(cp.A, right.value())))
            return components

    elif node.isgsm():
        scale_node = node.scale_node
        scale_components = extract_non_gaussian_part(scale_node)
        fixed_term, gaussian_term, chain_term = collect_terms(extract_terms(scale_node))
        assert chain_term is None
        scale_mu, scale_Sigma = gaussian_term.mu, gaussian_term.Sigma
        assert node.bias_type == 'col'
        scale_mu += node.bias.ravel()
        sigma_sq_approx = (node.value() ** 2).mean(0)
        return [GSMPredictiveDistribution(scale_components, scale_mu, scale_Sigma,
                                          sigma_sq_approx, np.eye(node.n))]
    
        


def compute_predictive_info(train_data_matrix, root, N):
    components = extract_non_gaussian_part(root)
    mu, Sigma = compute_gaussian_part(train_data_matrix, root, N)
    return PredictiveInfo(components, mu, Sigma)


def remove_gsm(predictive_info):
    new_components = []
    new_mu, new_Sigma = predictive_info.mu.copy(), predictive_info.Sigma.copy()
    for c in predictive_info.components:
        if isinstance(c, GSMPredictiveDistribution):
            #new_Sigma += np.diag(c.sigma_sq_approx)
            new_Sigma += np.dot(c.A.T, np.dot(np.diag(c.sigma_sq_approx), c.A))
        else:
            new_components.append(c)
    return PredictiveInfo(new_components, new_mu, new_Sigma)

def has_gsm(predictive_info):
    for c in predictive_info.components:
        if isinstance(c, GSMPredictiveDistribution):
            return True
    return False

