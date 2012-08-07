import numpy as np
nax = np.newaxis

from algorithms import low_rank_poisson, crp, ibp, sparse_coding, chains
import grammar
import observations
import recursive
from utils import misc

debugger = None


def init_low_rank(data_matrix, num_iter=200):
    m, n = data_matrix.m, data_matrix.n
    state, X = low_rank_poisson.fit_model(data_matrix, 2, num_iter=num_iter)
    U, V, ssq_U, ssq_N = state.U, state.V, state.ssq_U, state.ssq_N

    U /= ssq_U[nax, :] ** 0.25
    V *= ssq_U[:, nax] ** 0.25

    left = recursive.GaussianNode(U, 'col', np.sqrt(ssq_U))
    
    right = recursive.GaussianNode(V, 'row', np.sqrt(ssq_U))

    pred = np.dot(U, V)
    X = data_matrix.sample_latent_values(pred, ssq_N)
    noise = recursive.GaussianNode(X - pred, 'scalar', ssq_N)

    return recursive.SumNode([recursive.ProductNode([left, right]), noise])

def init_row_clustering(data_matrix, isotropic, num_iter=200):
    m, n = data_matrix.m, data_matrix.n
    state = crp.fit_model(data_matrix, isotropic_w=isotropic, isotropic_b=isotropic, num_iter=num_iter)

    U = np.zeros((m, state.assignments.max() + 1), dtype=int)
    U[np.arange(m), state.assignments] = 1
    left = recursive.MultinomialNode(U)

    if isotropic:
        right = recursive.GaussianNode(state.centers, 'scalar', state.sigma_sq_b)
    else:
        right = recursive.GaussianNode(state.centers, 'col', state.sigma_sq_b)
    
    pred = state.centers[state.assignments, :]
    X = data_matrix.sample_latent_values(pred, state.sigma_sq_w * np.ones((m, n)))
    if isotropic:
        noise = recursive.GaussianNode(X - pred, 'scalar', state.sigma_sq_w)
    else:
        noise = recursive.GaussianNode(X - pred, 'col', state.sigma_sq_w)
    
    return recursive.SumNode([recursive.ProductNode([left, right]), noise])

def init_col_clustering(data_matrix, isotropic, num_iter=200):
    return init_row_clustering(data_matrix.transpose(), isotropic, num_iter=num_iter).transpose()

def init_row_binary(data_matrix, num_iter=200):
    state = ibp.fit_model(data_matrix, num_iter=num_iter)

    left = recursive.BernoulliNode(state.Z)
    
    right = recursive.GaussianNode(state.A, 'scalar', state.sigma_sq_f)
    
    pred = np.dot(state.Z, state.A)
    X = data_matrix.sample_latent_values(pred, state.sigma_sq_n)
    noise = recursive.GaussianNode(X - pred, 'scalar', state.sigma_sq_n)
    
    return recursive.SumNode([recursive.ProductNode([left, right]), noise])

def init_col_binary(data_matrix, num_iter=200):
    return init_row_binary(data_matrix.transpose(), num_iter=num_iter).transpose()

def init_row_chain(data_matrix, num_iter=200):
    states, sigma_sq_D, sigma_sq_N = chains.fit_model(data_matrix, num_iter=num_iter)

    integ = chains.integration_matrix(data_matrix.m_orig)[data_matrix.row_ids, :]
    left = recursive.IntegrationNode(integ)
    
    temp = np.vstack([states[0, :][nax, :],
                      states[1:, :] - states[:-1, :]])
    right = recursive.GaussianNode(temp, 'scalar', sigma_sq_D)

    pred = states[data_matrix.row_ids, :]
    X = data_matrix.sample_latent_values(pred, sigma_sq_N)
    noise = recursive.GaussianNode(X - pred, 'scalar', sigma_sq_N)

    return recursive.SumNode([recursive.ProductNode([left, right]), noise])

def init_col_chain(data_matrix, num_iter=200):
    return init_row_chain(data_matrix.transpose(), num_iter=num_iter).transpose()

def init_sparsity(data_matrix, mu_Z_mode, num_iter=200):
    if mu_Z_mode == 'row':
        return init_sparsity(data_matrix.transpose(), 'col', num_iter)
    elif mu_Z_mode == 'col':
        by_column = True
    elif mu_Z_mode == 'scalar':
        by_column = False
    
    # currently, data_matrix should always be real-valued with no missing values, so this just
    # passes on data_matrix.observations.values; we may want to replace it with interval observations
    # obtained from slice sampling
    S = data_matrix.sample_latent_values(np.zeros((data_matrix.m, data_matrix.n)),
                                         np.ones((data_matrix.m, data_matrix.n)))
    
    Z = np.random.normal(-1., 1., size=S.shape)

    # sparse_coding.py wants a full sparse coding problem, so pass in None for the things
    # that aren't relevant here
    state = sparse_coding.SparseCodingState(S, None, Z, None, -1., 1., None)

    pbar = misc.pbar(num_iter)
    for i in range(num_iter):
        sparse_coding.sample_Z(state)
        state.mu_Z = sparse_coding.cond_mu_Z(state, by_column).sample()
        state.sigma_sq_Z = sparse_coding.cond_sigma_sq_Z(state).sample()

        if hasattr(debugger, 'after_init_sparsity_iter'):
            debugger.after_init_sparsity_iter(locals())

        pbar.update(i)
    pbar.finish()

    scale_node = recursive.GaussianNode(state.Z, 'scalar', state.sigma_sq_Z)
    return recursive.GSMNode(state.S, scale_node, mu_Z_mode, state.mu_Z)

    

def initialize(data_matrix, root, old_structure, new_structure, num_iter=200):
    root = root.copy()
    if old_structure == new_structure:
        return root
    node, old_dist, rule = recursive.find_changed_node(root, old_structure, new_structure)
    rule_name = grammar.rule2name[old_dist, rule]

    old = root.value()

    # if we're replacing the root, pass on the observation model; otherwise, treat
    # the node we're factorizing as exact real-valued observations
    if node is root:
        inner_data_matrix = data_matrix
    else:
        row_ids = recursive.row_ids_for(data_matrix, node)
        col_ids = recursive.col_ids_for(data_matrix, node)
        m_orig, n_orig = recursive.orig_shape_for(data_matrix, node)
        frv = observations.DataMatrix.from_real_values
        inner_data_matrix = frv(node.value(), row_ids=row_ids, col_ids=col_ids,
                                m_orig=m_orig, n_orig=n_orig)

    print 'Initializing %s from %s...' % (grammar.pretty_print(new_structure), grammar.pretty_print(old_structure))

    if rule_name == 'low-rank':
        new_node = init_low_rank(inner_data_matrix, num_iter=num_iter)
    elif rule_name == 'row-clustering':
        isotropic = (node is root)
        new_node = init_row_clustering(inner_data_matrix, isotropic, num_iter=num_iter)
    elif rule_name == 'col-clustering':
        isotropic = (node is root)
        new_node = init_col_clustering(inner_data_matrix, isotropic, num_iter=num_iter)
    elif rule_name == 'row-binary':
        new_node = init_row_binary(inner_data_matrix, num_iter=num_iter)
    elif rule_name == 'col-binary':
        new_node = init_col_binary(inner_data_matrix, num_iter=num_iter)
    elif rule_name == 'row-chain':
        new_node = init_row_chain(inner_data_matrix, num_iter=num_iter)
    elif rule_name == 'col-chain':
        new_node = init_col_chain(inner_data_matrix, num_iter=num_iter)
    elif rule_name == 'sparsity':
        new_node = init_sparsity(inner_data_matrix, node.variance_type, num_iter=num_iter)
    else:
        raise RuntimeError('Unknown production rule: %s' % rule_name)

    root = recursive.splice(root, node, new_node)

    if isinstance(data_matrix.observations, observations.RealObservations):
        assert np.allclose(root.value()[data_matrix.observations.mask], old[data_matrix.observations.mask])

    return root




