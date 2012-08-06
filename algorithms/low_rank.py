import numpy as np
nax = np.newaxis
import scipy.linalg
import time

from utils import misc

def sample_variance(values, axis):
    a = 0.01 + 0.5 * np.ones(values.shape).sum(axis)
    b = 0.01 + 0.5 * (values ** 2).sum(axis)
    prec = np.random.gamma(a, 1. / b)
    return 1. / prec

NUM_ITER = 200


def fit_model(data_matrix, K, num_iter=NUM_ITER, rotation_trick=True):
    N, D = data_matrix.m, data_matrix.n
    X = data_matrix.sample_latent_values(np.zeros((N, D)), 1.)

    if rotation_trick:
        U_, s_, V_ = scipy.linalg.svd(X, full_matrices=False)
        U = U_[:, :K] * np.sqrt(s_[:K][nax, :])
        V = V_[:K, :] * np.sqrt(s_[:K][:, nax])
    else:
        U = np.random.normal(size=(N, K))
        V = np.random.normal(size=(K, D))

    ssq_U = np.mean(U**2, axis=0)
    ssq_V = np.mean(V**2, axis=1)

    pred = np.dot(U, V)
    if data_matrix.observations.fixed_variance():
        ssq_N = 1.
    else:
        ssq_N = np.mean((X - pred) ** 2)

    t0 = time.time()
    for it in range(num_iter):
        if np.any(-data_matrix.observations.mask):
            obs = data_matrix.observations.mask
            U_var = np.outer(np.ones(N), ssq_U)
            V_var = np.outer(ssq_V, np.ones(D))
            U = misc.sample_gaussian_matrix2(V.T, X.T, 1. / U_var.T, obs.T / ssq_N).T
            V = misc.sample_gaussian_matrix2(U, X, 1. / V_var, obs / ssq_N)
        else:
            U = misc.sample_gaussian_matrix(np.eye(N), V, X, np.ones(N) / ssq_N, np.ones(D), np.ones(N), 1. / ssq_U)
            V = misc.sample_gaussian_matrix(U, np.eye(D), X, np.ones(N) / ssq_N, np.ones(D), 1. / ssq_V, np.ones(D))


        # rotation trick (to speed up learning the variances)
        if rotation_trick and it < num_iter // 4:
            UtU = np.dot(U.T, U)
            _, Q = scipy.linalg.eigh(UtU)
            Q = Q[:, ::-1]
            U = np.dot(U, Q)
            V = np.dot(Q.T, V)


        ssq_U = sample_variance(U, 0)
        ssq_V = sample_variance(V, 1)
        ssq_U = np.sqrt(ssq_U * ssq_V)
        ssq_V = ssq_U.copy()
        
        pred = np.dot(U, V)
        if not data_matrix.observations.fixed_variance():
            ssq_N = sample_variance(X - pred, None)

        X = data_matrix.sample_latent_values(pred, ssq_N)

        misc.print_dot(it+1, num_iter)

        if time.time() - t0 > 3600.:   # 1 hour
            break

    return U, V, ssq_U, ssq_V, ssq_N, X


