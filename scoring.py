import gc
import numpy as np
nax = np.newaxis

from algorithms import ais_gsm, variational
import config
import observations
import predictive_distributions
from utils import misc


CACHE = False
cached_pi = None

AIS_VERSION = 2
COLLAPSE = 'x'

MAX_IDXS = 500


def score_row_predictive_variational(train_data_matrix, root, test_data_matrix):
    #N = train_data_matrix.m + test_data_matrix.m
    N = test_data_matrix.m_orig
    predictive_info_orig = predictive_distributions.compute_predictive_info(train_data_matrix, root, N)
    predictive_info = predictive_distributions.remove_gsm(predictive_info_orig)

    result = np.zeros(test_data_matrix.m)
    #reps_all = []
    for i, row in enumerate(test_data_matrix.row_ids):
        idxs = np.where(test_data_matrix.observations.mask[i, :])[0]

        if MAX_IDXS is not None and idxs.size > MAX_IDXS:
            idxs = idxs[:MAX_IDXS]
        
        components, mu, Sigma = predictive_info.predictive_for_row(row, idxs)

        estimators = []
        for comp in components:
            if isinstance(comp, predictive_distributions.MultinomialPredictiveDistribution):
                estimators.append(variational.MultinomialEstimator(comp.pi, comp.centers))
            elif isinstance(comp, predictive_distributions.BernoulliPredictiveDistribution):
                estimators.append(variational.BernoulliEstimator(comp.pi, comp.A))
            else:
                raise RuntimeError('Unknown predictive distribution')

        if isinstance(test_data_matrix.observations, observations.RealObservations):
            problem = variational.MeanFieldProblem(estimators, test_data_matrix.observations.values[i, idxs] - mu,
                                                  Sigma)
            reps = problem.solve()
            result[i] = problem.objective_function(reps)
            #reps_all.append(reps)
        elif isinstance(test_data_matrix.observations, observations.BinaryObservations):
            Sigma_xgz = Sigma - np.eye(idxs.size)     # Sigma doesn't include the white Gaussian noise
            values = test_data_matrix.observations.values[i, idxs]
            cutoffs = test_data_matrix.observations.cutoffs[i, idxs]
            _, _, result[i] = variational_binary.solve(estimators, Sigma_xgz, values, cutoffs)
        elif isinstance(test_data_matrix.observations, observations.MixedObservations):
            ssq_N = root.children[-1].sigma_sq
            Sigma_xgz = Sigma - ssq_N * np.eye(idxs.size)     # Sigma doesn't include the white Gaussian noise
            _, _, result[i] = variational_mixed.solve(estimators, Sigma_xgz, test_data_matrix.observations[i, idxs],
                                                     ssq_N)


        if predictive_distributions.has_gsm(predictive_info_orig):
            components, mu, Sigma = predictive_info_orig.predictive_for_row(row, idxs)
            X = test_data_matrix.observations.values[i, idxs]
            X = X[nax, :]
            result[i] = ais_gsm.compute_likelihood(X, predictive_info_orig, [reps], np.array([result[i]]))[0]
            
            

        if config.USE_AMAZON_S3:
            amazon.check_visibility()

        misc.print_dot(i+1, test_data_matrix.m)


    return result

def score_col_predictive_variational(train_data_matrix, root, test_data_matrix):
    return score_row_predictive_variational(train_data_matrix.transpose(), root.transpose(), test_data_matrix.transpose())

def get_log_schedule(num_steps):
    t = np.linspace(-8., 8., num_steps)
    s = 1. / (1. + np.exp(-t))
    return (s - s[0]) / (s[-1] - s[0])



def score_row_predictive_ais(train_data_matrix, root, test_data_matrix, num_steps=2000, collapse=COLLAPSE,
                             logspace=True, plot=False):
    #N = train_data_matrix.m + test_data_matrix.m
    N = test_data_matrix.m_orig

    global cached_pi
    if CACHE:
        if cached_pi is None:
            cached_pi = predictive_distributions.compute_predictive_info(train_data_matrix, root, N)
        predictive_info = cached_pi
    else:
        predictive_info = predictive_distributions.compute_predictive_info(train_data_matrix, root, N)

    test_rows = test_data_matrix.row_ids
    components, mu, Sigma = predictive_info.predictive_for_rows(test_rows)

    # push a small amount of the noise into the "signal" so it's not degenerate
    sigma_sq_n = 0.99 * root.children[-1].sigma_sq
    Sigma_r = Sigma - sigma_sq_n * np.eye(Sigma.shape[-1])

    if logspace:
        schedule = get_log_schedule(num_steps)
    else:
        schedule = np.linspace(0., 1., num_steps)

    if AIS_VERSION == 2:
        problem = ais2.AISProblem(test_data_matrix.observations, components, mu, Sigma_r, sigma_sq_n, schedule,
                                  collapse)
        del Sigma_r
        del Sigma
        gc.collect()
        return ais2.run_ais(problem, plot=plot)
    elif AIS_VERSION == 1:
        problem = ais.AISProblem(test_data_matrix.observations, components, mu, Sigma_r, sigma_sq_n, schedule)
        return ais.run_ais(problem, collapse=collapse, plot=plot)
                          


def score_col_predictive_ais(train_data_matrix, root, test_data_matrix, *args, **kwargs):
    return score_row_predictive_ais(train_data_matrix.transpose(), root.transpose(), test_data_matrix.transpose(),
                                    *args, **kwargs)





## def no_structure_row_loglik_real(train_data, row_test_data):
##     assert isinstance(train_data, observations.DataMatrix) and isinstance(row_test_data, observations.DataMatrix)
##     var = np.sum(train_data.observations.mask * train_data.observations.values**2) / np.sum(train_data.observations.mask)
##     row_loglik_all = np.zeros(row_test_data.m)
##     for i in range(row_test_data.m):
##         obs = np.where(row_test_data.observations.mask[i,:])[0]
##         x = row_test_data.observations.values[i,obs]
##         row_loglik_all[i] = misc.gauss_loglik(x, np.zeros(x.shape), var * np.ones(x.shape))
##     return row_loglik_all

def no_structure_row_loglik(train_data, row_test_data):
    sigma_sq = train_data.observations.variance_estimate()
    return np.array([row_test_data.observations[i, :].loglik(np.zeros(row_test_data.n), sigma_sq)
                     for i in range(row_test_data.m)])


def no_structure_col_loglik(train_data, col_test_data):
    return no_structure_row_loglik(train_data.transpose(), col_test_data.transpose())
    
def evaluate_model(train_data, root, row_test_data, col_test_data, verbose=False, label='', avg_col_mean=True,
                   init_row_loglik=None, init_col_loglik=None, num_steps=2000):
    #use_ais = not isinstance(row_test_data.observations, observations.RealObservations)
    use_ais = isinstance(row_test_data.observations, observations.IntegerObservations)
    
    if use_ais:
        row_loglik_all = score_row_predictive_ais(train_data, root, row_test_data, num_steps=num_steps)
    else:
        row_loglik_all = score_row_predictive_variational(train_data, root, row_test_data)
    if avg_col_mean:
        if init_row_loglik is None:
            init_row_loglik = no_structure_row_loglik(train_data, row_test_data)
        row_loglik_all = misc.logsumexp(row_loglik_all + np.log(0.99),
                                         init_row_loglik + np.log(0.01))
    row_loglik = np.mean(row_loglik_all)
    if verbose:
        print '    %s row log-likelihood: %1.2f' % (label, row_loglik)

    if use_ais:
        col_loglik_all = score_col_predictive_ais(train_data, root, col_test_data, num_steps=num_steps)
    else:
        col_loglik_all = score_col_predictive_variational(train_data, root, col_test_data)
    if avg_col_mean:
        if init_col_loglik is None:
            init_col_loglik = no_structure_col_loglik(train_data, col_test_data)
        col_loglik_all = misc.logsumexp(col_loglik_all + np.log(0.99),
                                         init_col_loglik + np.log(0.01))
    col_loglik = np.mean(col_loglik_all)
    if verbose:
        print '    %s column log-likelihood: %1.2f' % (label, col_loglik)

    return row_loglik_all, col_loglik_all



