import numpy as np
nax = np.newaxis

from utils import distributions, misc

class DataMatrix:
    def __init__(self, observations, row_ids=None, col_ids=None, row_labels=None, col_labels=None,
                 m_orig=None, n_orig=None):
        self.m, self.n = observations.shape
        self.observations = observations
        
        if row_ids is None:                    # indices from the original matrix (used for chain models)
            row_ids = np.arange(self.m)
        self.row_ids = np.array(row_ids)
        if col_ids is None:
            col_ids = np.arange(self.n)
        self.col_ids = np.array(col_ids)
        
        if row_labels is None:                 # e.g. entity or attribute names
            row_labels = range(self.m)
        self.row_labels = list(row_labels)     # make sure it's not an array
        if col_labels is None:
            col_labels = range(self.n)
        self.col_labels = list(col_labels)
        
        if m_orig is None:                     # size of the original matrix (used for chain models)
            m_orig = self.m
        self.m_orig = m_orig
        if n_orig is None:
            n_orig = self.n
        self.n_orig = n_orig

    def transpose(self):
        return DataMatrix(self.observations.transpose(), self.col_ids, self.row_ids, self.col_labels, self.row_labels,
                          self.n_orig, self.m_orig)

    def copy(self):
        return DataMatrix(self.observations.copy(), self.row_ids.copy(), self.col_ids.copy(), list(self.row_labels),
                          list(self.col_labels), self.n_orig, self.m_orig)

    def __getitem__(self, slc):
        rslc, cslc = misc.extract_slices(slc)
        return DataMatrix(self.observations[slc], self.row_ids[rslc], self.col_ids[cslc],
                          misc.slice_list(self.row_labels, rslc), misc.slice_list(self.col_labels, cslc),
                          self.m_orig, self.n_orig)

    def sample_latent_values(self, predictions, noise):
        return self.observations.sample_latent_values(predictions, noise)

    def loglik(self, predictions, noise):
        return self.observations.loglik(predictions, noise)

    def fixed_variance(self):
        return self.observations.fixed_variance()

    @staticmethod
    def from_decomp(decomp):
        obs = RealObservations(decomp.root.value(), decomp.obs)
        return DataMatrix(obs, decomp.row_ids, decomp.col_ids, decomp.row_labels, decomp.col_labels)

    @staticmethod
    def from_real_values(values, mask=None, **kwargs):
        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        observations = RealObservations(values, mask)
        return DataMatrix(observations, **kwargs)

    @staticmethod
    def from_binary_values(values, mask=None, **kwargs):
        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        p = values[mask].mean()
        cutoffs = -misc.probit(p) * np.ones(values.shape)
        observations = BinaryObservations(values, mask, cutoffs)
        return DataMatrix(observations, **kwargs)

    @staticmethod
    def from_integer_values(values, mask=None, **kwargs):
        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        nbins = values.max() + 1
        cutoffs = []
        for i in range(nbins - 1):
            p = (values[mask] <= i).mean()
            cutoffs.append(misc.probit(p))
        observations = IntegerObservations(values, mask, np.array(cutoffs))
        return DataMatrix(observations, **kwargs) 

class RealObservations:
    def __init__(self, values, mask):
        self.values = values
        self.mask = mask
        self.shape = values.shape
        assert isinstance(self.values, np.ndarray) and self.values.dtype == float
        assert isinstance(self.mask, np.ndarray) and self.mask.dtype == bool
    
    def sample_latent_values(self, predictions, noise):
        missing_values = np.random.normal(predictions, np.sqrt(noise))
        return np.where(self.mask, self.values, missing_values)

    def copy(self):
        return RealObservations(self.values.copy(), self.mask.copy())

    def transpose(self):
        return RealObservations(self.values.T, self.mask.T)

    def loglik(self, predictions, noise):
        if not np.isscalar(noise):
            noise = noise[self.mask]
        return distributions.gauss_loglik(self.values[self.mask], predictions[self.mask], noise).sum()

    def loglik_each(self, predictions, noise):
        return np.where(self.mask,
                        distributions.gauss_loglik(self.values, predictions, noise),
                        0.)

    def fixed_variance(self):
        return False

    def variance_estimate(self):
        return (self.values[self.mask] ** 2).mean()

    def __getitem__(self, slc):
        return RealObservations(self.values[slc], self.mask[slc])


#EXACT = True
EXACT = False

def sample_cond_pos(mu, sigma_sq):
    if EXACT:
        log_p = np.log(misc.inv_probit(mu / np.sqrt(sigma_sq)))
    else:
        log_p = misc.log_inv_probit(mu / np.sqrt(sigma_sq))

    tmp = np.log(np.random.uniform(0., 1., size=mu.shape))
    return mu - np.sqrt(sigma_sq) * misc.log_probit(log_p + tmp)

def sample_cond_above(mu, sigma_sq, cutoff):
    return sample_cond_pos(mu - cutoff, sigma_sq) + cutoff

def sample_cond_below(mu, sigma_sq, cutoff):
    return -sample_cond_pos(-mu + cutoff, sigma_sq) + cutoff

def sample_cond_between_helper(mu, sigma_sq, L, U):
    sigma = np.sqrt(sigma_sq)
    #assert np.all(L < U)
    Lz = (L - mu) / sigma
    Uz = (U - mu) / sigma

    Lp = misc.log_inv_probit(Lz)
    Up = misc.log_inv_probit(Uz)
    ratio = np.exp(Lp - Up)
    log_p = Up + np.log(np.random.uniform(ratio, 1.))

    return mu + sigma * misc.log_probit(log_p)

def sample_cond_between(mu, sigma_sq, L, U):
    Lz = (L - mu) / np.sqrt(sigma_sq)
    return np.where(Lz > 0., -sample_cond_between_helper(-mu, sigma_sq, -U, -L),
                    sample_cond_between_helper(mu, sigma_sq, L, U))


def log_prob_below(mu, sigma_sq, cutoff):
    if EXACT:
        return np.log(misc.inv_probit((cutoff - mu) / np.sqrt(sigma_sq)))
    else:
        return misc.log_inv_probit((cutoff - mu) / np.sqrt(sigma_sq))

def log_prob_above(mu, sigma_sq, cutoff):
    if EXACT:
        return np.log(misc.inv_probit((mu - cutoff) / np.sqrt(sigma_sq)))
    else:
        return misc.log_inv_probit((mu - cutoff) / np.sqrt(sigma_sq))

def log_prob_between_helper(mu, sigma_sq, L, U):
    sigma = np.sqrt(sigma_sq)
    Lz = (L - mu) / sigma
    Uz = (U - mu) / sigma

    Lp = misc.log_inv_probit(Lz)
    Up = misc.log_inv_probit(Uz)
    ratio = np.exp(Lp - Up)

    return Up + np.log(1. - ratio)

def log_prob_between(mu, sigma_sq, L, U):
    Lz = (L - mu) / np.sqrt(sigma_sq)
    return np.where(Lz > 0., log_prob_between_helper(-mu, sigma_sq, -U, -L),
                    log_prob_between_helper(mu, sigma_sq, L, U))


class BinaryObservations:
    def __init__(self, values, mask, cutoffs):
        self.values = values
        self.mask = mask
        self.cutoffs = cutoffs
        self.shape = values.shape
        assert isinstance(self.values, np.ndarray) and self.values.dtype == bool
        assert isinstance(self.mask, np.ndarray) and self.mask.dtype == bool

    def copy(self):
        return BinaryObservations(self.values.copy(), self.mask.copy(), self.cutoffs.copy())

    def sample_latent_values(self, predictions, noise):
        pos = self.mask * self.values
        missing = -self.mask

        pos_values = sample_cond_above(predictions, noise, self.cutoffs)
        neg_values = sample_cond_below(predictions, noise, self.cutoffs)
        missing_values = np.random.normal(predictions, np.sqrt(noise))

        return np.where(missing, missing_values,
                        np.where(pos, pos_values, neg_values))

    def transpose(self):
        return BinaryObservations(self.values.T, self.mask.T, self.cutoffs.T)

    def loglik(self, predictions, noise):
        pos = self.mask * self.values
        neg = self.mask * -self.values
        if np.isscalar(noise):
            noise = noise * np.ones(predictions.shape)

        return log_prob_below(predictions[neg], noise[neg], self.cutoffs[neg]).sum() + \
               log_prob_above(predictions[pos], noise[pos], self.cutoffs[pos]).sum()

    def loglik_each(self, predictions, noise):
        pos = self.mask * self.values
        neg = self.mask * -self.values
        if np.isscalar(noise):
            noise = noise * np.ones(predictions.shape)

        result = np.zeros(self.values.shape)
        result[neg] = log_prob_below(predictions[neg], noise[neg], self.cutoffs[neg])
        result[pos] = log_prob_above(predictions[pos], noise[pos], self.cutoffs[pos])
        return result

    def fixed_variance(self):
        return True

    def variance_estimate(self):
        return 1.

    def __getitem__(self, slc):
        return BinaryObservations(self.values[slc], self.mask[slc], self.cutoffs[slc])



class IntegerObservations:
    def __init__(self, values, mask, cutoffs):
        self.values = values
        self.mask = mask
        self.cutoffs = cutoffs
        self.nbins = self.cutoffs.size + 1
        self.shape = values.shape
        assert isinstance(self.values, np.ndarray) and self.values.dtype == int
        assert isinstance(self.mask, np.ndarray) and self.mask.dtype == bool

    def copy(self):
        return IntegerObservations(self.values.copy(), self.mask.copy(), self.cutoffs.copy())

    def sample_latent_values(self, predictions, noise):
        result = np.zeros(self.shape)
        if np.isscalar(noise):
            noise = noise * np.ones(self.values.shape)

        missing_idxs = np.where(-self.mask)
        if missing_idxs[0].size > 0:
            result[missing_idxs] = np.random.normal(predictions[missing_idxs], np.sqrt(noise[missing_idxs]))

        smallest_idxs = np.where(self.mask * (self.values == 0))
        if smallest_idxs[0].size > 0:
            result[smallest_idxs] = sample_cond_below(predictions[smallest_idxs], np.sqrt(noise[smallest_idxs]), self.cutoffs[0])

        largest_idxs = np.where(self.mask * (self.values == self.nbins - 1))
        if largest_idxs[0].size > 0:
            result[largest_idxs] = sample_cond_above(predictions[largest_idxs], np.sqrt(noise[largest_idxs]), self.cutoffs[-1])

        idxs = np.where(self.mask * (self.values > 0) * (self.values < self.nbins - 1))
        if idxs[0].size > 0:
            vals = self.values[idxs]
            result[idxs] = sample_cond_between(predictions[idxs], np.sqrt(noise[idxs]), self.cutoffs[vals-1], self.cutoffs[vals])

        return result

    def transpose(self):
        return IntegerObservations(self.values.T, self.mask.T, self.cutoffs)

    def fixed_variance(self):
        return False

    def variance_estimate(self):
        return 1.

    def loglik(self, predictions, noise):
        if np.isscalar(noise):
            noise = noise * np.ones(self.values.shape)
        
        smallest = self.mask * (self.values == 0)
        total = log_prob_below(predictions[smallest], noise[smallest], self.cutoffs[0]).sum()

        middle = self.mask * (self.values > 0) * (self.values < self.nbins - 1)
        vals = self.values[middle]
        total += log_prob_between(predictions[middle], noise[middle], self.cutoffs[vals-1], self.cutoffs[vals]).sum()

        largest = self.mask * (self.values == self.nbins - 1)
        total += log_prob_above(predictions[largest], noise[largest], self.cutoffs[-1]).sum()

        return total

    def loglik_each(self, predictions, noise):
        if np.isscalar(noise):
            noise = noise * np.ones(self.values.shape)
        result = np.zeros(self.values.shape)
        
        smallest = self.mask * (self.values == 0)
        result[smallest] = log_prob_below(predictions[smallest], noise[smallest], self.cutoffs[0])

        middle = self.mask * (self.values > 0) * (self.values < self.nbins - 1)
        vals = self.values[middle]
        result[middle] = log_prob_between(predictions[middle], noise[middle], self.cutoffs[vals-1], self.cutoffs[vals])

        largest = self.mask * (self.values == self.nbins - 1)
        result[largest] = log_prob_above(predictions[largest], noise[largest], self.cutoffs[-1])
        return result

    def __getitem__(self, slc):
        return IntegerObservations(self.values[slc], self.mask[slc], self.cutoffs)



class MixedObservations:
    def __init__(self, real_values, real_mask, below_values, below_mask, above_values, above_mask):
        assert real_values.dtype == float
        assert real_mask.dtype == bool
        assert below_values.dtype == float
        assert below_mask.dtype == bool
        assert above_values.dtype == float
        assert above_mask.dtype == bool

        self.real_values = real_values
        self.real_mask = real_mask
        self.below_values = below_values
        self.below_mask = below_mask
        self.above_values = above_values
        self.above_mask = above_mask
        self.mask = real_mask + below_mask + above_mask
        assert not np.any(real_mask * below_mask) and not np.any(real_mask * above_mask)
        assert real_values.shape == below_values.shape == above_values.shape
        self.shape = self.real_values.shape

    def copy(self):
        return MixedObservations(self.real_values.copy(), self.real_mask.copy(),
                                 self.below_values.copy(), self.below_mask.copy(),
                                 self.above_values.copy(), self.above_mask.copy())

    def sample_latent_values(self, predictions, noise):
        noise = noise * np.ones(self.shape)
        
        result = np.zeros(self.shape)
        only_below = self.below_mask * -self.above_mask
        only_above = -self.below_mask * self.above_mask
        both = self.below_mask * self.above_mask

        result[self.real_mask] = self.real_values[self.real_mask]
        result[only_below] = sample_cond_below(predictions[only_below], noise[only_below],
                                               self.below_values[only_below])
        result[only_above] = sample_cond_above(predictions[only_above], noise[only_above],
                                               self.above_values[only_above])
        result[both] = sample_cond_between(predictions[both], noise[both], self.above_values[both],
                                           self.below_values[both])

        return result

    def transpose(self):
        return MixedObservations(self.real_values.T, self.real_mask.T, self.below_values.T,
                                 self.below_mask.T, self.above_values.T, self.above_mask.T)

    def fixed_variance(self):
        return False

    def variance_estimate(self):
        return self.sample_latent_values(np.zeros(self.shape), 1.).var()

    def loglik_each(self, predictions, noise):
        noise = noise * np.ones(self.shape)

        result = np.zeros(self.shape)
        real = self.real_mask
        only_below = self.below_mask * -self.above_mask
        only_above = -self.below_mask * self.above_mask
        both = self.below_mask * self.above_mask

        result[real] = distributions.gauss_loglik(self.real_values[real], predictions[real], noise[real])
        result[only_below] = log_prob_below(predictions[only_below], noise[only_below],
                                            self.below_values[only_below])
        result[only_above] = log_prob_above(predictions[only_above], noise[only_above],
                                            self.above_values[only_above])
        result[both] = log_prob_between(predictions[both], noise[both], self.above_values[both],
                                        self.below_values[both])

        return result
    
    def loglik(self, predictions, noise):
        return self.loglik_each(predictions, noise).sum()

    def __getitem__(self, slc):
        return MixedObservations(self.real_values[slc], self.real_mask[slc], self.below_values[slc],
                                 self.below_mask[slc], self.above_values[slc], self.above_mask[slc])

