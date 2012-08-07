
###################### debugging code ##########################################

def default_model(ndim=40, isotropic=True):
    alpha = 1.
    within_var_prior = distributions.InverseGammaDistribution(1., 1.)
    between_var_prior = distributions.InverseGammaDistribution(1., 1.)
    return CRPModel(alpha, ndim, within_var_prior, between_var_prior, isotropic, isotropic)

def random_instance(model, ndata, missing=True):
    """Generate a random instance for debugging purposes (not actually drawn from the model distribution)"""
    if model.isotropic_w:
        within_var = 0.5
    else:
        within_var = np.random.uniform(0.2, 1., model.ndim)
    if model.isotropic_b:
        between_var = 1.
    else:
        between_var = np.random.uniform(0.2, 1., model.ndim)
    alpha = np.ones(20) * 3. / 20
    temp = distributions.sample_dirichlet(alpha)
    temp2 = np.array([distributions.MultinomialDistribution(temp).sample()
                      for i in range(ndata)])
    temp2 = temp2[:, temp2.any(0)]  # eliminate empty clusters
    assignments = temp2.argmax(1)
    ncomp = assignments.max() + 1
    centers = np.random.normal(0., np.sqrt(between_var), size=(ncomp, model.ndim))
    X = np.random.normal(centers[assignments, :], np.sqrt(within_var))

    if missing:
        obs = np.random.binomial(1, 0.5, size=X.shape).astype(bool)
        #data = CRPData(X, obs)
        data = observations.RealObservations(X, obs)
    else:
        #data = CRPData(X)
        data = observations.RealObservations(X, np.ones(X.shape, dtype=bool))
    state = FullCRPState(X, assignments, centers, within_var, between_var)
    return data, state

def random_initialization(model, data):
    X = data.sample_latent_values(np.zeros(data.shape), 1.)
    assignments = np.random.randint(0, 10, size=data.num)
    if model.isotropic_w:
        sigma_sq_w = np.random.gamma(1., 1.)
    else:
        sigma_sq_w = np.random.gamma(1., 1., size=model.ndim)
    if model.isotropic_b:
        sigma_sq_b = np.random.gamma(1., 1.)
    else:
        sigma_sq_b = np.random.gamma(1., 1., size=model.ndim)
    return CollapsedCRPState(X, assignments, sigma_sq_w, sigma_sq_b)


def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        raise RuntimeError('a=%f, b=%f' % (a, b))

def check(missing=True, isotropic=True):
    model = default_model(isotropic=isotropic)
    NDATA = 50
    data, state = random_instance(model, NDATA, missing=missing)
    cache = CollapsedCRPCache.from_state(model, data, state)

    # check the conditional distribution over cluster centers
    for i in range(5):
        cond = cond_centers(model, data, state, cache)
        new_centers1, new_centers2 = cond.to_distribution().sample(), cond.to_distribution().sample()
        new_state1, new_state2 = state.copy(), state.copy()
        new_state1.centers = new_centers1; new_state2.centers = new_centers2

        check_close(cond.score(new_centers1).sum() - cond.score(new_centers2).sum(),
                    p_tilde(model, data, new_state1) - p_tilde(model, data, new_state2))


    # check hyperparameter posterior
    for i in range(5):
        cond = cond_sigma_sq_b(model, data, state)
        new_state = state.copy()
        new_sigma_sq_b = cond.sample()
        new_state.sigma_sq_b = new_sigma_sq_b
        check_close(np.sum(cond.loglik(new_sigma_sq_b)) - np.sum(cond.loglik(state.sigma_sq_b)),
                    p_tilde(model, data, new_state) - p_tilde(model, data, state))

        cond = cond_sigma_sq_w(model, data, state)
        new_state = state.copy()
        new_sigma_sq_w = cond.sample()
        new_state.sigma_sq_w = new_sigma_sq_w
        check_close(np.sum(cond.loglik(new_sigma_sq_w)) - np.sum(cond.loglik(state.sigma_sq_w)),
                    p_tilde(model, data, new_state) - p_tilde(model, data, state))

    # check consistency after Gibbs steps
    for i in range(data.shape[0]):
        gibbs_step_assignments_collapsed(model, data, state, cache, i)
        cache.check(data, state)

    # check consistency after adding clusters
    state2 = state.copy()
    cache2 = cache.copy()
    for i in range(data.shape[0]):
        new_assignment = state2.assignments.max()
        state2.assignments[i] = new_assignment
        cache2.replace(i, new_assignment)
        cache2.squeeze(state2)
        cache2.check(data, state2)

    # check consistency after deleting clusters
    state2 = state.copy()
    cache2 = cache.copy()
    for i in range(data.shape[0]):
        state2.assignments[i] = 0
        cache2.replace(i, 0)
        cache2.squeeze(state2)
        cache2.check(data, state2)

    # check that the marginals agree with the ratio of likelihoods
    ncomp = state.assignments.max() + 1
    for i in range(data.shape[0]):
        cond = cond_assignments_collapsed(model, data, state, cache, i)
        assert np.all(-np.isnan(cond.p))
        assert np.all(-np.isnan(cond.log_p))
        assert np.all(state.assignments >= 0)
        for k in range(ncomp + 1):
            if cache.counts[state.assignments[i]] == 1:
                continue    # the conditional need not agree with the likelihood when deleting a cluster

            new_state = state.copy()
            assert np.all(state.assignments >= 0)
            new_state.assignments[i] = k
            cache.check(data, state)
            check_close(cond.loglik(k) - cond.loglik(state.assignments[i]),
                        p_tilde_collapsed(model, data, new_state) - p_tilde_collapsed(model, data, state))


def try_synthetic(sigma_sq_w=1., ndata=50):
    model = default_model()
    true_assignments = np.array([i % 5 for i in range(ndata)])
    sigma_sq_b = 1.
    
    centers = np.random.normal(0, np.sqrt(sigma_sq_b), size=(5, model.ndim))
    X = np.random.normal(centers[true_assignments, :], np.sqrt(sigma_sq_w))
    data = observations.RealObservations(X, np.ones(X.shape, dtype=bool))

    state = random_initialization(model, data)

    for i in range(20):
        t0 = time.time()
        gibbs_sweep_collapsed(model, data, state)
        print '%1.1f seconds' % (time.time() - t0)
        print state.assignments


