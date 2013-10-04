import numpy as np
nax = np.newaxis

from algorithms import ais_gsm, variational
import observations
import predictive_distributions
from utils import misc


CACHE = False
cached_pi = None

def score_row_predictive_variational(train_data_matrix, root, test_data_matrix, num_steps_ais=2000):
    N = test_data_matrix.m_orig
    predictive_info_orig = predictive_distributions.compute_predictive_info(train_data_matrix, root, N)
    predictive_info = predictive_distributions.remove_gsm(predictive_info_orig)

    result = np.zeros(test_data_matrix.m)
    pbar = misc.pbar(test_data_matrix.m)
    for i, row in enumerate(test_data_matrix.row_ids):
        idxs = np.where(test_data_matrix.observations.mask[i, :])[0]

        components, mu, Sigma = predictive_info.predictive_for_row(row, idxs)

        estimators = []
        for comp in components:
            if isinstance(comp, predictive_distributions.MultinomialPredictiveDistribution):
                estimators.append(variational.MultinomialEstimator(comp.pi, comp.centers))
            elif isinstance(comp, predictive_distributions.BernoulliPredictiveDistribution):
                estimators.append(variational.BernoulliEstimator(comp.pi, comp.A))
            else:
                raise RuntimeError('Unknown predictive distribution')

        assert isinstance(test_data_matrix.observations, observations.RealObservations)
        
        problem = variational.VariationalProblem(estimators, test_data_matrix.observations.values[i, idxs] - mu,
                                                 Sigma)
        reps = problem.solve()
        result[i] = problem.objective_function(reps)

        if predictive_distributions.has_gsm(predictive_info_orig):
            components, mu, Sigma = predictive_info_orig.predictive_for_row(row, idxs)
            assert np.allclose(mu, 0.)   # can't do chains yet
            X = test_data_matrix.observations.values[i, idxs]
            X = X[nax, :]
            result[i] = ais_gsm.compute_likelihood(X, components, Sigma, [reps], np.array([result[i]]),
                                                   num_steps=num_steps_ais)[0]

        pbar.update(i)
    pbar.finish()


    return result

def score_col_predictive_variational(train_data_matrix, root, test_data_matrix, num_steps_ais=2000):
    return score_row_predictive_variational(train_data_matrix.transpose(), root.transpose(),
                                            test_data_matrix.transpose(), num_steps_ais=num_steps_ais)



def no_structure_row_loglik(train_data, row_test_data):
    sigma_sq = train_data.observations.variance_estimate()
    return np.array([row_test_data.observations[i, :].loglik(np.zeros(row_test_data.n), sigma_sq)
                     for i in range(row_test_data.m)])


def no_structure_col_loglik(train_data, col_test_data):
    return no_structure_row_loglik(train_data.transpose(), col_test_data.transpose())
    
def evaluate_model(train_data, root, row_test_data, col_test_data, label='', avg_col_mean=True,
                   init_row_loglik=None, init_col_loglik=None, num_steps_ais=2000, max_dim=None):

    print 'Scoring row predictive likelihood...'
    row_loglik_all = score_row_predictive_variational(
        train_data[:, :max_dim], root[:, :max_dim], row_test_data[:, :max_dim], num_steps_ais=num_steps_ais)
    if avg_col_mean:
        if init_row_loglik is None:
            init_row_loglik = no_structure_row_loglik(train_data[:, :max_dim], row_test_data[:, :max_dim])
        row_loglik_all = np.logaddexp(row_loglik_all + np.log(0.99),
                                      init_row_loglik + np.log(0.01))

    print 'Scoring column predictive likelihood...'
    col_loglik_all = score_col_predictive_variational(
        train_data[:max_dim, :], root[:max_dim, :], col_test_data[:max_dim, :], num_steps_ais=num_steps_ais)
    if avg_col_mean:
        if init_col_loglik is None:
            init_col_loglik = no_structure_col_loglik(train_data[:max_dim, :], col_test_data[:max_dim, :])
        col_loglik_all = np.logaddexp(col_loglik_all + np.log(0.99),
                                      init_col_loglik + np.log(0.01))

    return row_loglik_all, col_loglik_all



