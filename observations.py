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

