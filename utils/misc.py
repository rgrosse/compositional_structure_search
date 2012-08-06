import collections
from PIL import Image
import itertools
import math
import numpy as np
nax = np.newaxis
import pylab
import scipy.linalg, scipy.integrate
import sys
import termcolor
import time

def override(dicts, add=True):
    result = {}
    for dict in dicts:
        for key, val in dict.items():
            if add or dicts[0].has_key(key):
                result[key] = val
    return result

def arr2img(arr, rescale=True):
    if rescale:
        assert np.all(0. <= arr) and np.all(arr <= 1.)
        return Image.fromarray((arr*255).astype('uint8'))
    else:
        return Image.fromarray(arr.astype('uint8'))

def group(list, size, mode='all'):
    """
    Partitions list into groups of size, where the final
    group possibly contains fewer elements.

    Example:
        group(range(8), 3) ==> [[0, 1, 2], [3, 4, 5], [6, 7]]
    """

    # Force a copy of numpy array/matrix
    if type(list) in [np.array, np.core.defmatrix.matrix]:
        list = list.copy()
        
    result = []
    start = 0
    while True:
        end = min(start + size, len(list))
        result.append(list[start:end])
        if end == len(list):
            break
        start += size

    if mode=='truncate' and len(result[-1]) < size:
        result = result[:-1]
        
    return result


def logsumexp(A, B):
    """Compute the entrywise log-sum-exp of A and B"""
    mx = np.max((A, B), 0)
    A_ = A-mx
    B_ = B-mx
    C_ = np.log(np.exp(A_) + np.exp(B_))
    return C_+mx

def logsumexp_vec(a, axis=None):
    """Compute the log-sum-exp of a along a given axis"""
    a = np.asarray(a)
    ndim = a.ndim
    mx = np.max(a, axis=axis)
    if axis is not None:
        mx_ = shape_to_cons('*'*axis + '1' + '*'*(ndim-axis-1), mx)
    else:
        mx_ = mx
    return np.log(np.sum(np.exp(a - mx_), axis=axis)) + mx

def multinomial_entropy(q):
    assert False
    q = np.asarray(q)
    return np.sum(np.where(q > 1e-10, -q*np.log2(q), 0.))

def bernoulli_entropy(q):
    assert False
    q = np.asarray(q)
    return np.sum(np.where((q > 1e-10) * (1.-q > 1e-10), -q*np.log2(q) - (1-q)*np.log2(1-q), 0.))

def harmonic_mean_log(values):
    """Compute the harmonic mean in log space"""
    values = np.array(values)
    return -(logsumexp_vec(-values) - np.log(values.size))

def mean_log(values):
    """Compute the mean in log space"""
    return logsumexp_vec(values) - np.log(len(values))


def kronecker_product(A, B):
    ma, na = A.shape
    mb, nb = B.shape
    m = ma * mb
    n = na * nb
    result = np.zeros((m, n))
    for i in range(ma):
        for j in range(na):
            result[i*mb:(i+1)*mb, j*nb:(j+1)*nb] = A[i,j] * B
    return result

def vec(X):
    return X.T.ravel()
def vec2mat(x, m, n):
    assert x.shape == (m*n,)
    return x.reshape(n, m).T.copy()

def solve_for_interactions(X, U, V, obs_noise, R_var, W=None):
    """Compute the expected value of R given X, U, and V, under the model
    where X = URV^T + noise."""
    if W is None:
        W = np.ones(X.shape)

    # TODO: a better way would be to sample this
    R_var = max(R_var, 0.01)
    
    m, k1 = U.shape
    n, k2 = V.shape
    assert X.shape == (m, n)
    Z = kronecker_product(V, U)
    reg_weight = obs_noise / R_var



    left = np.dot(Z.T, st('*1', vec(W)) * Z) + reg_weight * np.eye(k1 * k2)
    right = np.dot(Z.T, vec(W) * vec(X))
    r = np.linalg.solve(left, right)
    return vec2mat(r, k1, k2)

def is_diag(A):
    return A.shape[0] == A.shape[1] and np.all(A == np.diag(np.diag(A)))

def my_svd(A):
    m, n = A.shape
    if is_diag(A):
        return np.eye(m), np.diag(A), np.eye(m)
    else:
        return scipy.linalg.svd(A, full_matrices=False)

def map_gaussian_matrix(A, B, C, d_1, d_2, d_3, d_4):
    """sample X, where P(X) \propto e^{-J(X)} and
    J(X) = 1/2 \|D_1(AXB - C)D_2\|^2 + 1/2 \|D_3 X D_4\|^2."""
    A_tilde = st('*1', d_1) * A * st('1*', 1./d_3)
    B_tilde = st('*1', 1./d_4) * B * st('1*', d_2)
    C_tilde = st('*1', d_1) * C * st('1*', d_2)

    U_A, lambda_A, Vt_A = my_svd(A_tilde)
    V_A = Vt_A.T
    
    U_B, lambda_B, Vt_B = my_svd(B_tilde)
    V_B = Vt_B.T

    Lambda = st('*1', lambda_A) * st('1*', lambda_B)
    Y = Lambda * np.dot(np.dot(U_A.T, C_tilde), V_B) / (1. + Lambda**2)
    X_tilde = np.dot(np.dot(V_A, Y), U_B.T)
    X = st('*1', 1./d_3) * X_tilde * st('1*', 1./d_4)

    return X

def map_gaussian_matrix_em(A, B, C, d_1, d_2, d_3, d_4, obs, X):
    C_ = np.where(obs, C, np.dot(np.dot(A, X), B))
    return map_gaussian_matrix(A, B, C_, d_1, d_2, d_3, d_4)
    

def sample_gaussian_matrix(A, B, C, d_1, d_2, d_3, d_4):
    """sample X, where P(X) \propto e^{-J(X)} and
    J(X) = 1/2 \|D_1(AXB - C)D_2\|^2 + 1/2 \|D_3 X D_4\|^2."""
    A_tilde = st('*1', d_1) * A * st('1*', 1./d_3)
    B_tilde = st('*1', 1./d_4) * B * st('1*', d_2)
    C_tilde = st('*1', d_1) * C * st('1*', d_2)

    U_A, lambda_A, Vt_A = my_svd(A_tilde)
    V_A = Vt_A.T

    U_B, lambda_B, Vt_B = my_svd(B_tilde)
    V_B = Vt_B.T

    Lambda = st('*1', lambda_A) * st('1*', lambda_B)
    Y_mean = Lambda * np.dot(np.dot(U_A.T, C_tilde), V_B) / (1. + Lambda**2)
    Y_var = 1. / (1. + Lambda**2)
    Y = np.random.normal(Y_mean, np.sqrt(Y_var))
    X_tilde = np.dot(np.dot(V_A, Y), U_B.T)
    X = st('*1', 1./d_3) * X_tilde * st('1*', 1./d_4)

    return X

def sample_gaussian_matrix_em(A, B, C, d_1, d_2, d_3, d_4, obs, X):
    C_ = np.where(obs, C, np.dot(np.dot(A, X), B))
    return sample_gaussian_matrix(A, B, C_, d_1, d_2, d_3, d_4)



def sample_gaussian_matrix2(A, B, W_X, W_N):
    nrows, ncols = A.shape[1], B.shape[1]
    X = np.zeros((nrows, ncols))
    for j in range(ncols):
        Lambda = np.dot(np.dot(A.T, np.diag(W_N[:,j])), A) + np.diag(W_X[:,j])
        Sigma = np.linalg.inv(Lambda)
        mu = mult([Sigma, A.T, W_N[:,j] * B[:,j]])
        X[:,j] = np.random.multivariate_normal(mu, Sigma)
    return X

def map_gaussian_matrix2(A, B, W_X, W_N):
    nrows, ncols = A.shape[1], B.shape[1]
    X = np.zeros((nrows, ncols))
    for j in range(ncols):
        Lambda = np.dot(np.dot(A.T, np.diag(W_N[:,j])), A) + np.diag(W_X[:,j])
        Sigma = np.linalg.inv(Lambda)
        mu = mult([Sigma, A.T, W_N[:,j] * B[:,j]])
        X[:,j] = mu
    return X





def integral_from_information(Lambda, J, c):
    """Given the information form of an unnormalized gaussian
            f(x) = -0.5 * x^T Lambda x - J^T x - c,
    compute the integral."""
    if np.isscalar(Lambda):
        return 0.5 * np.log(2*np.pi) - 0.5 * np.log(Lambda) - c + 0.5 * J**2 / Lambda
    else:
        n = J.size
        d, Q = scipy.linalg.eigh(Lambda)
        return 0.5 * n * np.log(2*np.pi) - 0.5 * np.sum(np.log(d)) - c + 0.5 * np.dot(np.dot(J, np.linalg.inv(Lambda)), J)

def information_to_expectation(Lambda, J, c=0.):
    if np.isscalar(Lambda):
        Sigma = 1. / Lambda
        mu = -Sigma * J
    else:
        Sigma = np.linalg.inv(Lambda)
        mu = -np.dot(Sigma, J)
    log_Z = integral_from_information(Lambda, J, c)
    return Sigma, mu, log_Z

def expectation_to_information(Sigma, mu, log_Z=0.):
    if np.isscalar(Sigma):
        Lambda = 1. / Sigma
        J = -Lambda * mu
        c = -log_Z + 0.5 * np.log(2 * np.pi * Sigma) + 0.5 * mu**2 * Lambda
    else:
        Lambda = np.linalg.inv(Sigma)
        J = -np.dot(Lambda, mu)
        d, Q = np.linalg.eigh(Sigma)
        c = -log_Z + 0.5 * np.sum(np.log(2 * np.pi * d)) + 0.5 * np.dot(np.dot(mu, Lambda), mu)
    return Lambda, J, c

    


def st(shape_str, arr):
    """Takes a string of 1's and *'s, and an array. Returns an array with dummy axes
    in the positions specified by 1's. For instance, if a is a 1-D vector,
    st('*1', a) is a column vector and st('1*', a) is a row vector."""
    new_shape = []
    count = 0
    for c in shape_str:
        assert c == '*' or c == '1'
        if c == '*':
            new_shape.append(arr.shape[count])
            count += 1
        else:
            new_shape.append(1)
    return arr.reshape(tuple(new_shape))

def mult(matrices):
    """Matrix multiplication"""
    prod = matrices[0]
    for mat in matrices[1:]:
        prod = np.dot(prod, mat)
    return prod

def gauss_ml_diag(X):
    """Maximum likelihood for a gaussian with diagonal covariance. Each row is an observation."""
    mu = X.mean(0)
    sigma = np.mean((X - st('1*', mu)) ** 2, axis=0)
    return mu, sigma

def gauss_loglik(X, mu, Sigma):
    """Log-likelihood under a gaussian observation model."""
    was_vector = (X.ndim == 1)
    if was_vector:
        X = st('1*', X)
    pi_term = -0.5 * X.shape[1] * np.log(2*np.pi)
    diff = X - st('1*', mu)
    if Sigma.ndim == 1:
        det_term = -0.5 * np.sum(np.log(Sigma))
        obs_term = -0.5 * np.sum(diff**2 / st('1*', Sigma), axis=1)
    else:
        d, Q = scipy.linalg.eigh(Sigma)
        det_term = -0.5 * np.sum(np.log(d))
        obs_term = -0.5 * np.sum(np.sum(st('**1', diff) * st('*1*', diff) * st('1**', np.linalg.inv(Sigma)),
                                 axis=2), axis=1)
    if was_vector:
        obs_term = obs_term[0]
    return  pi_term + det_term + obs_term

def gauss_loglik2(X, mu, d, Q):
    """Log-likelihood under a gaussian observation model, where d and Q are the eigenvalues
    and eigenvectors of the covariance matrix."""
    was_vector = (X.ndim == 1)
    if was_vector:
        X = st('1*', X)
    pi_term = -0.5 * X.shape[1] * np.log(2*np.pi)
    det_term = -0.5 * np.sum(np.log(d))
    diff = X - st('1*', mu)
    QTdiff = np.dot(diff, Q)
    obs_term = -0.5 * np.sum(QTdiff**2 / st('1*', d), axis=1)
    if was_vector:
        obs_term = obs_term[0]
    return pi_term + det_term + obs_term

def gauss_loglik3(x, mu, L):
    """Log-likelihood under a gaussian observation model, where L is the Cholesky factor
    for the covariance matrix."""
    assert x.ndim == 1
    pi_term = -0.5 * x.size * np.log(2*np.pi)
    det_term = -np.sum(np.log(np.diag(L)))
    diff = x - mu
    #Sigma_inv_diff = scipy.linalg.cho_solve((L, True), diff)
    L_inv_diff = scipy.linalg.lu_solve((L, np.arange(x.size)), diff)
    obs_term = -0.5 * np.dot(L_inv_diff, L_inv_diff)
    return pi_term + det_term + obs_term

def gauss_loglik4(X, mu, L, Sigma_inv):
    """Log-likelihood under a gaussian observation model, where d and Q are the eigenvalues
    and eigenvectors of the covariance matrix."""
    was_vector = (X.ndim == 1)
    if was_vector:
        X = st('1*', X)
    pi_term = -0.5 * X.shape[1] * np.log(2*np.pi)
    det_term = -np.sum(np.log(np.diag(L)))
    diff = X - st('1*', mu)
    diff_Sigma_inv = np.dot(diff, Sigma_inv)
    obs_term = -0.5 * np.sum(diff * diff_Sigma_inv, axis=1)
    if was_vector:
        obs_term = obs_term[0]
    return pi_term + det_term + obs_term

    

def gauss_entropy(Sigma):
    if np.isscalar(Sigma):
        return 0.5 * (1 + np.log(2*np.pi)) + 0.5 * np.log(sigma_sq)
    elif Sigma.ndim == 1:
        n = Sigma.size
        return 0.5 * n * (1 + np.log(2*np.pi)) + 0.5 * np.sum(np.log(Sigma))
    else:
        assert Sigma.ndim == 2
        n = Sigma.shape[0]
        d, Q = scipy.linalg.eigh(Sigma)
        return 0.5 * n * (1 + np.log(2*np.pi)) + 0.5 * np.sum(np.log(d))

def gauss_condition(b, mu_a, mu_b, Sigma_aa, Sigma_ab, Sigma_bb):
    """Compute P(a|b) where a and b are jointly gaussian."""
    mu = mu_a + np.dot(Sigma_ab, np.linalg.solve(Sigma_bb, b-mu_b))
    Sigma = Sigma_aa - np.dot(Sigma_ab, np.dot(np.linalg.inv(Sigma_bb), Sigma_ab.T))
    return mu, Sigma

TOL = 1e-10
EPS = 1e-8
NUM_TRIES = 20
def verify_optimum(fobj, x, verbose=False, eps=EPS, tol=TOL):
    """Check that x minimizes fobj to a specified tolerance."""
    fobj_opt = fobj.value(x)
    if verbose:
        print 'fobj_opt', fobj_opt
    for i in range(NUM_TRIES):
        eps_ = eps*np.random.normal(size=x.shape)
        fobj_curr = fobj.value(x+eps_)
        if verbose:
            print 'fobj_curr', fobj_curr
        assert fobj_curr > fobj_opt - tol
    if verbose:
        print 'Passed.'


def mean_field(J, Lambda, z_init=None):
    n = J.size
    assert J.shape == (n,) and Lambda.shape == (n, n)
    if z_init is not None:
        z = z_init.copy()
    else:
        z = np.zeros(n)

    # move quadratic potentials for one variable to unary terms
    J = J + 0.5 * Lambda[range(n), range(n)]
    Lambda[range(n), range(n)] = 0.

    for tr in range(100):
        for j in range(n):
            Lambda_term = np.dot(Lambda, z)
            odds = -J - Lambda_term
            odds = odds.clip(-100., 100.)   # to avoid the overflow warnings
            z_new = 1. / (1. + np.exp(-odds))
            z[j] = 0.8*z[j] + 0.2*z_new[j]

    return z

NEWLINE_EVERY = 50
dummy_count = [0]
def print_dot(count=None, max=None):
    print_count = (count is not None)
    if count is None:
        dummy_count[0] += 1
        count = dummy_count[0]
    sys.stdout.write('.')
    sys.stdout.flush()
    if count % NEWLINE_EVERY == 0:
        if print_count:
            if max is not None:
                sys.stdout.write(' [%d/%d]' % (count, max))
            else:
                sys.stdout.write(' [%d]' % count)
        sys.stdout.write('\n')
    elif count == max:
        sys.stdout.write('\n')
    sys.stdout.flush()

def match_paren(string, pos):
    assert string[pos] == '('
    depth = 0
    while pos < len(string):
        if string[pos] == '(':
            depth += 1
        if string[pos] == ')':
            depth -= 1
        if depth == 0:
            return pos
        pos += 1
    return None




def estimate_noise(N):
    nrows, ncols = N.shape
    sigma_sq_rows = np.ones(nrows)
    sigma_sq_cols = np.ones(ncols)

    for tr in range(10):
        sigma_sq_rows = np.mean((N**2 / sigma_sq_cols[nax,:]), axis=1)
        sigma_sq_cols = np.mean((N**2 / sigma_sq_rows[:,nax]), axis=0)

    return sigma_sq_rows, sigma_sq_cols

def sample_noise(N, obs=None, b0=1.):
    if obs is None:
        obs = np.ones(N.shape, dtype=bool)
        
    nrows, ncols = N.shape
    ssq_rows, ssq_cols = sample_noise_tied(N, obs, b0)
    lambda_rows = 1. / ssq_rows
    lambda_cols = 1. / ssq_cols

    a0 = 1.

    for tr in range(10):
        a = a0 + 0.5 * obs.sum(1)
        b = b0 + 0.5 * np.sum(obs * N**2 * lambda_cols[nax,:], axis=1)
        lambda_rows = np.random.gamma(a, 1. / b)

        if np.isscalar(lambda_rows):  # np.random.gamma converts singleton arrays into scalars
            lambda_rows = np.array([lambda_rows])

        a = a0 + 0.5 * obs.sum(0)
        b = b0 + 0.5 * np.sum(obs * N**2 * lambda_rows[:,nax], axis=0)
        lambda_cols = np.random.gamma(a, 1. / b)

        if np.isscalar(lambda_cols):
            lambda_cols = np.array([lambda_cols])

    return 1. / lambda_rows, 1. / lambda_cols

def sample_noise_tied(N, obs=None, b0=1.):
    if obs is None:
        obs = np.ones(N.shape, dtype=bool)
    nrows, ncols = N.shape
    a0 = 1.
    a = a0 + 0.5 * obs.sum()
    b = b0 + 0.5 * np.sum(obs * N**2)
    prec = np.random.gamma(a, 1. / b)

    return np.ones(nrows) / np.sqrt(prec), np.ones(ncols) / np.sqrt(prec)

def sample_col_noise(N):
    nrows, ncols = N.shape
    A0 = 1.
    B0 = 1.
    B0 = np.mean(N**2)   # UNDO
    a = A0 + 0.5 * nrows
    b = B0 + 0.5 * np.sum(N**2, axis=0)
    return 1. / np.random.gamma(a, 1. / b)
    

def greedy_match(scores):
    m, n = scores.shape
    remaining_rows = range(m)
    remaining_cols = range(n)

    match_rows = []
    match_cols = []
    while len(remaining_rows) > 0 and len(remaining_cols) > 0:
        rr, rc = np.array(remaining_rows), np.array(remaining_cols)
        rem = scores[rr[:,nax], rc[nax,:]]
        loc = np.argmax(rem)
        r_ind, c_ind = np.unravel_index(loc, rem.shape)
        row, col = remaining_rows[r_ind], remaining_cols[c_ind]
        match_rows.append(row)
        match_cols.append(col)
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    return match_rows, match_cols
    

def permute_to_match(U_true, U):
    diff = np.array([[np.sum((U_true[:,i] - U[:,j]) ** 2)
                      for j in range(U.shape[1])]
                     for i in range(U_true.shape[1])])
    rows, cols = greedy_match(-diff)
    inds = np.argsort(rows)
    cols = [cols[i] for i in inds]
    unmatched = [i for i in range(U.shape[1]) if i not in cols]
    U_inds = cols + unmatched
    return U[:, U_inds]

def permute_to_match2(U_true, U):
    diff = np.array([[np.sum((U_true[:,i] - U[:,j]) ** 2)
                      for j in range(U.shape[1])]
                     for i in range(U_true.shape[1])])
    best = np.argmin(diff, axis=0)
    inds = np.argsort(best)
    return U[:,inds]


def update_cholesky(L0, a):
    m = L0.shape[0]
    L = np.zeros((m+1, m+1))
    L[:m, :m] = L0
    # update 7-17-11: scipy changed the input format to lu_solve
    # L[m, :m] = scipy.linalg.lu_solve((L0, np.arange(m)), a[:m])
    if scipy.version.version in ['0.9.0rc2', '0.9.0', '0.8.0']:
        L[m, :m] = scipy.linalg.lu_solve((L0[::-1, ::-1], np.arange(m)), a[:m][::-1])[::-1]
    elif scipy.version.version == '0.7.1':
        L[m, :m] = scipy.linalg.lu_solve((L0, np.arange(m)), a[:m])
    else:
        raise RuntimeError('Unknown SciPy version: %s' % scipy.version.version)
    L[m, m] = np.sqrt(a[m] - np.dot(L[m, :m], L[m, :m]))
    assert np.all(np.isfinite(L))
    return L


    
def log_integrate(fn, xmin, xmax, npts):
    assert False






def get_arg_list(arguments, args, kwargs):
    assert type(args) == tuple
    assert type(kwargs) == dict
    nargs = len(args)
    args = list(args)
    for argument in arguments[nargs:]:
        assert type(argument) in [str, tuple], 'Invalid argument: %s' % argument
        if type(argument) == tuple:
            assert len(argument) == 2, 'Invalid argument: %s' % argument
            has_default = True
            name, default = argument
        elif type(argument) == str:
            has_default = False
            name = argument
            
        if name in kwargs:
            args.append(kwargs[name])
        else:
            assert has_default, 'No value given for argument %s' % argument
            args.append(default)
    return args

    

class Memoized:
    """Utility class for memoizing functions, either in memory or to disk."""
    def __init__(self, fn, arguments, keys=None, filename=None, filename_args=None):
        """Memoize a function. arguments should be a list containing either
        strings or (string, value) pairs, the string being the argument name
        and the value being the default value. Keys should be a list of strings
        or (string, function) pairs, where the string gives the argument name
        and the function is optionally called on the value of the argument."""
        self.fn = fn
        
        self.arg_names = []
        for arg in arguments:
            assert type(arg) in [str, tuple], 'Invalid argument name: %s' % arg
            if type(arg) == str:
                self.arg_names.append(arg)
            elif type(arg) == tuple:
                assert len(arg) == 2, 'Invalid argument name: %s' % arg
                self.arg_names.append(arg[0])
        self.arguments = arguments
        assert len(set(self.arg_names)) == len(self.arg_names), 'Names must be unique.'
        
        if keys is None:
            keys = self.arg_names
        for k in keys:
            assert type(k) in [str, tuple], 'Invalid key: %s' % k
            if type(k) == tuple:
                assert len(k) == 2, 'Invalid key: %s' % k
        self.keys = keys

        self.filename = filename
        self.filename_args = filename_args
        if filename_args is not None:
            assert filename is not None
        if filename is None:
            self.cache = {}
            

    def __call__(self, *args, **kwargs):
        arg_list = get_arg_list(self.arguments, args, kwargs)
        arg_dict = dict(zip(self.arg_names, arg_list))
        tag = []
        for k in self.keys:
            assert type(k) in [str, tuple], 'Invalid key: %s' % k
            if type(k) == str:
                tag.append(arg_dict[k])
            elif type(k) == tuple:
                assert len(k) == 2, 'Invalid key: %s' % k
                name, fn = k
                tag.append(fn(arg_dict[name]))
        tag = tuple(tag)

        if self.filename is not None:
            if self.filename_args is not None:
                if not os.path.exists(self.filename):
                    os.mkdir(self.filename)
                parts = []
                for k in self.filename_args:
                    assert type(k) in [str, tuple], 'Invalid key: %s' % k
                    if type(k) == str:
                        parts.append(arg_dict[k])
                    elif type(k) == tuple:
                        assert len(k) == 2, 'Invalid key: %s' % k
                        name, fn = k
                        parts.append(fn(arg_dict[name]))
                parts_str = '_'.join(parts)
                cache_file = os.path.join(self.filename, parts_str)
            else:
                cache_file = self.filename
            if not os.path.exists(cache_file):
                cPickle.dump({}, open(cache_file, 'w'))
            cache = cPickle.load(open(cache_file))
        else:
            cache_file = None
            cache = self.cache
        

        if tag in cache:
            return cache[tag]
        ans = self.fn(*args, **kwargs)
        cache[tag] = ans
        if cache_file is not None:
            cPickle.dump(cache, open(cache_file, 'w'))
        return ans

def print_time(seconds):
    if seconds > 3600:
        return '%dh %dm' % (int(seconds/3600), int(seconds/60)%60)
    elif seconds > 60:
        return '%dm %ds' % (int(seconds/60), int(seconds)%60)
    elif seconds > 0:
        return '%ds' % int(seconds)
    else:
        return '???'
    
    
class TimeEstimate:
    def __init__(self, fname):
        self.fname = fname
        self.start_time = None
        self.done_ = 0

    def start(self):
        self.start_time = time.time()

    def done(self, done_, out_of):
        self.done_ = done_
        self.out_of = out_of
        self.last_time = time.time()
        

    def eta_string(self):
        if self.start_time is None or self.done_ == 0:
            return '???'
        else:
            t = (self.last_time - self.start_time) * (self.out_of / float(self.done_))
            estimate = self.start_time + t
            delta = estimate - time.time()
            return print_time(delta)


def combine_precisions(Sigma_v, Lambda_y):
    """compute (Sigma_v + Lambda_y^{-1})^{-1}, where Lambda_y may be low rank"""
    d, Q = scipy.linalg.eigh(Lambda_y)
    zero_count = np.sum(d < 1e-10)
    if zero_count == Lambda_y.shape[0]:
        return np.zeros(Lambda_y.shape)
    Q = Q[:,zero_count:]

    Sigma_v_proj = np.dot(np.dot(Q.T, Sigma_v), Q)
    Lambda_y_proj = np.dot(np.dot(Q.T, Lambda_y), Q)
    ans_proj = scipy.linalg.inv(Sigma_v_proj + scipy.linalg.inv(Lambda_y_proj))
    return np.dot(np.dot(Q, ans_proj), Q.T)



def kalman_filter(mu_0, Sigma_0, A, mu_v, Sigma_v, B, Lambda_n, y):
    nlat = mu_0.size
    nvis, ntime = y.shape

    assert np.allclose(mu_v, 0.)
    
    mu_forward = np.zeros((nlat, ntime))
    Sigma_forward = np.zeros((nlat, nlat, ntime))
    mu = np.zeros((nlat, ntime))
    Sigma = np.zeros((nlat, nlat, ntime))
    mu_forward[:,0] = mu_0
    Sigma_forward[:,:,0] = Sigma_0
    
    # forward propagation
    for t in range(ntime):
        # execute dynamics
        if t > 0:
            mu_forward[:,t] = np.dot(A, mu[:,t-1]) + mu_v
            Sigma_forward[:,:,t] = np.dot(np.dot(A, Sigma[:,:,t-1]), A.T) + Sigma_v

        # account for observations
        Lambda, J, _ = expectation_to_information(Sigma_forward[:,:,t], mu_forward[:,t])
        Lambda += np.dot(np.dot(B.T, Lambda_n[:,:,t]), B)
        J -= np.dot(B.T, np.dot(Lambda_n[:,:,t], y[:,t]))
        Sigma[:,:,t], mu[:,t], _ = information_to_expectation(Lambda, J)

    J_backward = np.zeros((nlat, ntime))
    Lambda_backward = np.zeros((nlat, nlat, ntime))


    # backward propagation
    for t in range(ntime-1)[::-1]:
        Lambda_obs = np.dot(np.dot(B.T, Lambda_n[:,:,t]), B)
        Lambda_prime = Lambda_backward[:,:,t+1] + Lambda_obs
        J_prime = -np.dot(B.T, np.dot(Lambda_n[:,:,t], y[:,t])) + J_backward[:,t+1]
        mu_prime = -np.linalg.lstsq(Lambda_prime, J_prime)[0]

        Lambda_prime = combine_precisions(Sigma_v, Lambda_prime)

        Lambda_backward[:,:,t] = np.dot(np.dot(A.T, Lambda_prime), A)
        mu_backward = np.linalg.lstsq(A, mu_prime)[0]
        J_backward[:,t] = -np.dot(Lambda_backward[:,:,t], mu_backward)

    # combine both directions
    for t in range(ntime):
        Lambda, J, _ = expectation_to_information(Sigma[:,:,t], mu[:,t])
        J += J_backward[:,t]
        Lambda += Lambda_backward[:,:,t]
        Sigma[:,:,t], mu[:,t], _ = information_to_expectation(Lambda, J)

    return mu, Sigma


def check_kalman_filter():
    sigma_sq_n = 0.5
    
    x = np.linspace(-2., 2., 100)
    y_true = x**3 - 2*x
    y = y_true[nax,:] + np.random.normal(0., np.sqrt(sigma_sq_n), size=(2, 100))

    vis.figure('Kalman filter test')
    pylab.clf()
    pylab.plot(x, y_true, 'k-')
    pylab.plot(x[:25], y[0,:25], 'bx')
    pylab.plot(x[:25], y[1,:25], 'bx')
    pylab.plot(x[75:], y[0,75:], 'bx')
    pylab.plot(x[75:], y[1,75:], 'bx')

    mu_0 = np.zeros(3)
    Sigma_0 = 1000 * np.eye(3)
    A = np.array([[1., 1., 0.],
                  [0., 1., 1.],
                  [0., 0., 1.]])
    mu_v = np.zeros(3)
    Sigma_v = np.diag([0., 0., 0.000001])
    B = np.array([[1., 0., 0.],
                  [1., 0., 0.]])
    Lambda_n = np.zeros((2, 2, 100))
    for i in range(25) + range(75, 100):
        Lambda_n[:,:,i] = np.eye(2) / sigma_sq_n

    mu, Sigma = kalman_filter(mu_0, Sigma_0, A, mu_v, Sigma_v, B, Lambda_n, y)

    pylab.plot(x, mu[0,:], 'r-')
    pylab.plot(x, mu[0,:] - 2 * np.sqrt(Sigma[0,0,:]), 'r--')
    pylab.plot(x, mu[0,:] + 2 * np.sqrt(Sigma[0,0,:]), 'r--')
    
    
    
def logdet(A):
    """Compute the log-determinant of a symmetric positive definite matrix A using the Cholesky factorization."""
    L = np.linalg.cholesky(A)
    return 2 * np.sum(np.log(np.diag(L)))


def log_integrate(f, xp):
    diff = np.max(f)
    f = f - diff
    intl = np.log(scipy.integrate.simps(np.exp(f), xp))
    return intl + diff

class LogDistribution:
    def __init__(self, log_prob, locs):
        self.log_prob = log_prob
        self.locs = locs

    def integral(self):
        return scipy.integrate.simps(np.exp(self.log_prob), self.locs)

    def expected_value(self):
        return scipy.integrate.simps(np.exp(self.log_prob) * self.locs, self.locs)

    def variance(self):
        return scipy.integrate.simps(np.exp(self.log_prob) * self.locs**2, self.locs) - self.expected_value() ** 2

    def entropy(self):
        return scipy.integrate.simps(np.exp(self.log_prob) * -self.log_prob, self.locs)

    def sample(self):
        dist - np.exp(self.log_prob)
        dist /= np.sum(dist)
        return self.locs[np.argmax(np.random.multinomial(1, dist))]

    def expectation_of(self, f):
        return scipy.integrate.simps(np.exp(self.log_prob) * f, self.locs)



def inverse_rank_one_update(A_inv, u):
    """Given the inverse A_inv of a symmetric matrix A, compute the inverse of A + uu^T using the
    Sherman-Morrison formula."""
    v = np.dot(A_inv, u)
    denom = 1 + np.dot(u, v)
    return A_inv - np.outer(v, v) / denom



def slice_list(lst, slc):
    """Slice a Python list as if it were an array."""
    if isinstance(slc, np.ndarray):
        slc = slc.ravel()
    idxs = np.arange(len(lst))[slc]
    return [lst[i] for i in idxs]

## def slice_projected(arr, slc):
##     """Slice a 1-D array, but convert the slice to a 1-D array if it is a multi-dimensional array.
##     Useful when an object maintains metadata about rows and columns, and responds to slices of the form
##     obj[row_idxs[:, nax], col_idxs[nax, :]]."""
##     if isinstance(slc, np.ndarray):
##         slc = slc.ravel()
##     return arr[slc]

def extract_slices(slc):
    if type(slc) == tuple:
        result = []
        for s in slc:
            if isinstance(s, np.ndarray):
                result.append(s.ravel())
            else:
                result.append(s)
        return tuple(result)
    else:
        return slc



def _err_string(arr1, arr2):
    try:
        if np.allclose(arr1, arr2):
            return 'OK'
        elif arr1.shape == arr2.shape:
            return 'off by %s' % np.abs(arr1 - arr2).max()
        else:
            return 'incorrect shapes: %s and %s' % (arr1.shape, arr2.shape)
    except:
        return 'error comparing'

err_info = collections.defaultdict(list)
def set_err_info(key, info):
    err_info[key] = info

def summarize_error(key):
    """Print a helpful description of the reason a condition was not satisfied. Intended usage:
        assert pot1.allclose(pot2), summarize_error()"""
    if type(err_info[key]) == str:
        return '    ' + err_info[key]
    else:
        return '\n' + '\n'.join(['    %s: %s' % (name, err) for name, err in err_info[key]]) + '\n'


def broadcast(idx, shape):
    result = []
    for i, d in zip(idx, shape):
        if d == 1:
            result.append(0)
        else:
            result.append(i)
    return tuple(result)

def full_shape(shapes):
    """The shape of the full array that results from broadcasting the arrays of the given shapes."""
    return tuple(np.array(shapes).max(0))


def array_map(fn, arrs, n):
    """Takes a list of arrays a_1, ..., a_n where the elements of the first n dimensions line up. For every possible
    index into the first n dimensions, apply fn to the corresponding slices, and combine the results into
    an n-dimensional array. Supports broadcasting but does not prepend 1's to the shapes."""
    # we shouldn't need a special case for n == 0, but NumPy complains about indexing into a zero-dimensional
    # array a using a[(Ellipsis,)].
    if n == 0:
        return fn(*arrs)
    
    full_shape = tuple(np.array([a.shape[:n] for a in arrs]).max(0))
    result = None
    for full_idx in itertools.product(*map(range, full_shape)):
        inputs = [a[broadcast(full_idx, a.shape[:n]) + (Ellipsis,)] for a in arrs]
        curr = fn(*inputs)
        
        if result is None:
            if type(curr) == tuple:
                result = tuple(np.zeros(full_shape + np.asarray(c).shape) for c in curr)
            else:
                result = np.zeros(full_shape + np.asarray(curr).shape)

        if type(curr) == tuple:
            for i, c in enumerate(curr):
                result[i][full_idx + (Ellipsis,)] = c
        else:
            result[full_idx + (Ellipsis,)] = curr
    return result

def extend_slice(slc, n):
    if not isinstance(slc, tuple):
        slc = (slc,)
    if any([isinstance(s, np.ndarray) for s in slc]):
        raise NotImplementedError('Advanced slicing not implemented yet')
    return slc + (slice(None),) * n

def process_slice(slc, shape, n):
    """Takes a slice and returns the appropriate slice into an array that's being broadcast (i.e. by
    converting the appropriate entries to 0's and :'s."""
    if not isinstance(slc, tuple):
        slc = (slc,)
    slc = list(slc)
    ndim = len(shape) - n
    assert ndim >= 0
    shape_idx = 0
    for slice_idx, s in enumerate(slc):
        if s == nax:
            continue
        if shape[shape_idx] == 1:
            if type(s) == int:
                slc[slice_idx] = 0
            else:
                slc[slice_idx] = slice(None)
        shape_idx += 1
    if shape_idx != ndim:
        raise IndexError('Must have %d terms in the slice object' % ndim)
    return extend_slice(tuple(slc), n)

def my_sum(a, axis, count):
    """For an array a which might be broadcast, return the value of a.sum() were a to be expanded out in full."""
    if a.shape[axis] == count:
        return a.sum(axis)
    elif a.shape[axis] == 1:
        return count * a.sum(axis)
    else:
        raise IndexError('Cannot be broadcast: a.shape=%s, axis=%d, count=%d' % (a.shape, axis, count))
        
    

def match_shapes(arrs):
    """Prepend 1's to the shapes so that the dimensions line up."""
    #temp = [(name, np.asarray(a), deg) for name, a, deg in arrs]
    #ndim = max([a.ndim - deg for _, a, deg in arrs])

    temp = [a for name, a, deg in arrs]
    for i in range(len(temp)):
        if np.isscalar(temp[i]):
            temp[i] = np.array(temp[i])
    ndim = max([a.ndim - deg for a, (_, _, deg) in zip(temp, arrs)])

    prep_arrs = []
    for name, a, deg in arrs:
        if np.isscalar(a):
            a = np.asarray(a)
        if a.ndim < deg:
            raise RuntimeError('%s.ndim must be at least %d' % (name, deg))
        if a.ndim < ndim + deg:
            #a = a.reshape((1,) * (ndim + deg - a.ndim) + a.shape)
            slc = (nax,) * (ndim + deg - a.ndim) + (Ellipsis,)
            a = a[slc]
        prep_arrs.append(a)

    return prep_arrs
    
def lstsq(A, b):
    # do this rather than call lstsq to support efficient broadcasting
    P = array_map(np.linalg.pinv, [A], A.ndim - 2)
    return array_map(np.dot, [P, b], A.ndim - 2)

def dot(A, b):
    return array_map(np.dot, [A, b], A.ndim - 2)

def vdot(x, y):
    return (x*y).sum(-1)

def my_inv(A):
    """Compute the inverse of a symmetric positive definite matrix."""
    cho = scipy.linalg.flapack.dpotrf(A)
    choinv = scipy.linalg.flapack.dtrtri(cho[0])
    upper = scipy.linalg.flapack.dlauum(choinv[0])[0]

    # upper is the upper triangular entries of A^{-1}, so need to fill in the
    # lower triangular ones; unfortunately this has nontrivial overhead
    temp = np.diag(upper)
    return upper + upper.T - np.diag(temp)


def transp(A):
    return A.swapaxes(-2, -1)


def get_counts(array, n):
    result = np.zeros(n, dtype=int)
    ans = np.bincount(array)
    result[:ans.size] = ans
    return result


def log_erfc_helper(x):
    p = 0.47047
    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556
    t = 1. / (1 + p*x)
    return np.log(a1 * t + a2 * t**2 + a3 * t**3) - x ** 2

def log_erfc(x):
    return np.where(x > 0., log_erfc_helper(x), np.log(2. - np.exp(log_erfc_helper(-x))))

def log_inv_probit(x):
    return log_erfc(-x / np.sqrt(2.)) - np.log(2.)

def inv_probit(x):
    return 0.5 * scipy.special.erfc(-x / np.sqrt(2.))

def log_erfcinv(log_y):
    a = 0.140012
    log_term = log_y + np.log(2 - np.exp(log_y))

    temp1 = 2 / (np.pi * a) + 0.5 * log_term
    temp2 = temp1 ** 2 - log_term / a
    temp3 = np.sqrt(temp2) - temp1
    return np.sign(1. - np.exp(log_y)) * np.sqrt(temp3)

def log_probit(log_p):
    return -np.sqrt(2) * log_erfcinv(log_p + np.log(2))

def probit(p):
    return -np.sqrt(2) * scipy.special.erfcinv(2 * p)


def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        if np.isscalar(a) and np.isscalar(b):
            raise RuntimeError('a=%f, b=%f' % (a, b))
        else:
            raise RuntimeError('Off by %f' % np.max(np.abs(a - b)))

COLORS = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

def print_integers_colored(a):
    print '[',
    for ai in a:
        color = COLORS[ai % len(COLORS)]
        print termcolor.colored(str(ai), color, attrs=['bold']),
    print ']'
