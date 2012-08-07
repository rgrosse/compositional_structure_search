
########################## debugging code ######################################

def random_instance(simple, N=30, K=20, D=25, p=0.2):
    alpha = 2.
    feature_var_prior = distributions.InverseGammaDistribution(1., 1.)
    noise_var_prior = distributions.InverseGammaDistribution(1., 1.)
    model = IBPModel(alpha, feature_var_prior, noise_var_prior)

    # make sure Z doesn't have any empty columns
    Z = np.random.binomial(1, p, size=(N, K))
    for j in range(Z.shape[1]):
        if not np.any(Z[:, j]):
            i = np.random.randint(0, Z.shape[0])
            Z[i, j] = 1
    
    A = np.random.normal(size=(K, D))
    X = np.random.normal(np.dot(Z, A))
    if simple:
        #data = SimpleData(X)
        data = observations.RealObservations(X, np.ones(X.shape, dtype=bool))
    else:
        raise NotImplementedError()
        obs = np.random.binomial(1., 0.5, size=(N, D)).astype(bool)
        data = GeneralData(X, obs)

    state = FullIBPState(X, Z, A, 0.5, 0.8)

    return model, data, state



def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        raise RuntimeError('a=%f, b=%f' % (a, b))
    

def check_ibp_loglik(alpha):
    Z = np.array([[1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 1, 0],
                  [1, 1, 1, 1]])
    correct_answer = -alpha * (1. / np.arange(1, 5)).sum() + 4 * np.log(alpha) - np.log(13824)
    check_close(ibp_loglik(Z, alpha), correct_answer)

def check_cache(simple, new_dishes, delete):
    model, data, state = random_instance(simple)
    N, D = state.X.shape
    cache = IBPCache.from_state(model, data, state, np.ones(state.X.shape[0], dtype=bool))

    if delete:
        delete_cols = np.where(np.random.binomial(1, 0.5, size=state.Z.shape[1]))[0]

    for i in range(state.X.shape[0]):
        cache.remove(i)
        K = state.Z.shape[1]
        if delete:
            z = np.random.binomial(1, 0.5, size=K)
            z[delete_cols] = 0
        else:
            z = np.where(cache.counts > 0, np.random.binomial(1, 0.5, size=K), 0)
        state.Z[i, :] = z
        if new_dishes and np.random.binomial(1, 0.5):
            # add a dish
            state.Z = np.hstack([state.Z, np.zeros((N, 1))])
            state.Z[i, -1] = 1
            cache.add_dish()
        cache.add(i, state.Z[i, :], state.X[i, :])
        
    # check consistency of the posterior
    cache.fpost.check()

    # check that the cache matches
    cache.check(data, state)

def check_Sigma_info(simple):
    K, D = 20, 10

    u = np.random.binomial(1, 0.5, size=K)
    mu = np.random.normal(size=(K, D))
    if simple:
        A = np.random.normal(size=(K, K))
        Sigma = np.dot(A, A.T)
        Sigma_info = SimpleSigmaInfo(Sigma, mu, u)
    else:
        raise NotImplementedError()
        Sigma = np.zeros((K, K, D))
        for i in range(D):
            A = np.random.normal(size=(K, K))
            Sigma[:, :, i] = np.dot(A, A.T)
        Sigma_info = GeneralSigmaInfo(Sigma, mu, u)

    def sigma_sq(Sigma, u):
        if simple:
            return np.dot(u, np.dot(Sigma, u))
        else:
            return np.array([np.dot(u, np.dot(Sigma[:, :, j], u))
                             for j in range(D)])
    
    assert np.allclose(Sigma_info.sigma_sq(), sigma_sq(Sigma, u))
    assert np.allclose(Sigma_info.mu(), np.dot(u, mu))
    
    for it in range(20):
        k = np.random.randint(0, K)
        uk = np.random.binomial(1, 0.5)
        v = u.copy()
        v[k] = uk
        assert np.allclose(Sigma_info.sigma_sq_for(k, uk), sigma_sq(Sigma, v))
        assert np.allclose(Sigma_info.mu_for(k, uk), np.dot(v, mu))

        Sigma_info.update(k, uk)
        assert np.allclose(Sigma_info.sigma_sq(), sigma_sq(Sigma, v))
        assert np.allclose(Sigma_info.mu(), np.dot(v, mu))
        u = v



def check_assignment_conditional(simple):
    model, data, state = random_instance(simple)
    N, D = state.X.shape
    K = state.Z.shape[1]
    cache = IBPCache.from_state(model, data, state, np.ones(state.X.shape[0], dtype=bool))

    for tr in range(20):
        i = np.random.randint(0, N)
        k = np.random.randint(0, K)

        cache.remove(i)
        #Sigma_info = GeneralSigmaInfo(cache.fpost.Sigma, state.Z[i, :])
        Sigma_info = cache.fpost.Sigma_info(state.Z[i, :])
        cond = cond_assignment_collapsed(model, data, state, cache, Sigma_info, i, k)
        new_assignment = 1 - state.Z[i, k]
        new_state = state.copy()
        new_state.Z[i, k] = new_assignment
        check_close(cond.loglik(new_assignment) - cond.loglik(state.Z[i, k]),
                    p_tilde_collapsed(model, data, new_state) - p_tilde_collapsed(model, data, state))
        cache.add(i, state.Z[i, :], state.X[i, :])
        Sigma_info.update(k, state.Z[i, k])

def check_feature_conditional(simple):
    model, data, state = random_instance(simple)
    N, D = state.X.shape
    K = state.Z.shape[1]
    cache = IBPCache.from_state(model, data, state, np.ones(state.X.shape[0], dtype=bool))

    mu, Sigma = cache.fpost.mu.T, cache.fpost.Sigma.T
    if simple:
        pot = gaussians.Potential.from_moments_full(mu, Sigma[nax, :, :])
    else:
        pot = gaussians.Potential.from_moments_full(mu, Sigma)
    
    A1 = np.random.normal(size=(K, D))
    A2 = np.random.normal(size=(K, D))
    new_state1 = state.copy()
    new_state1.A = A1
    new_state2 = state.copy()
    new_state2.A = A2

    check_close(pot.score(A2.T).sum() - pot.score(A1.T).sum(),
                p_tilde_uncollapsed(model, data, new_state2) - p_tilde_uncollapsed(model, data, new_state1))



def check_hyperparameter_conditional(simple):
    model, data, state = random_instance(simple)
    cache = IBPCache.from_state(model, data, state, np.ones(state.X.shape[0], dtype=bool))
    state.A = cache.fpost.sample()

    cond = cond_sigma_sq_f(model, data, state)
    new_sigma_sq_f = np.random.gamma(1., 1.)
    new_state = state.copy()
    new_state.sigma_sq_f = new_sigma_sq_f
    check_close(cond.loglik(new_sigma_sq_f) - cond.loglik(state.sigma_sq_f),
                p_tilde_uncollapsed(model, data, new_state) - p_tilde_uncollapsed(model, data, state))

    cond = cond_sigma_sq_n(model, data, state)
    new_sigma_sq_n = np.random.gamma(1., 1.)
    new_state = state.copy()
    new_state.sigma_sq_n = new_sigma_sq_n
    check_close(cond.loglik(new_sigma_sq_n) - cond.loglik(state.sigma_sq_n),
                p_tilde_uncollapsed(model, data, new_state) - p_tilde_uncollapsed(model, data, state))


def check():
    check_ibp_loglik(1.5)
    #for simple in [True, False]:
    for simple in [True]:
        check_Sigma_info(simple)
        for new_dishes in [False, True]:
            for delete in [False, True]:
                check_cache(simple, new_dishes, delete)
        check_assignment_conditional(simple)
        check_feature_conditional(simple)
        check_hyperparameter_conditional(simple)



def solve(simple, N, K, D, p, niter, split_merge):
    model, data, _ = random_instance(simple, N, K, D, p)
    state = sequential_init(model, data)
    for it in range(niter):
        print it
        gibbs_sweep(model, data, state, split_merge)
        print '   k =', state.Z.shape[1]



def mean_field(h, Lambda, c, prior_odds):
    n = h.size
    z = np.zeros(n)

    h = h - 0.5 * Lambda[range(n), range(n)]
    Lambda = Lambda.copy()
    Lambda[range(n), range(n)] = 0.

    for it in range(100):
        Lambda_term = -np.dot(Lambda, z)
        odds = h + Lambda_term + prior_odds
        z_new = 1. / (1 + np.exp(-odds))
        z = 0.8*z + 0.2*z_new

    log_p = -np.logaddexp(0., -prior_odds)
    log_1mp = -np.logaddexp(0., prior_odds)

    obj = -0.5 * np.dot(z, np.dot(Lambda, z)) + np.dot(h, z) + c + np.dot(z, log_p) + np.dot(1-z, log_1mp)

    return z, obj

def score_predictive(model, data, state, maximize=False):
    N_train, N_test, K, D = state.Z.shape[0], state.X.shape[0], state.Z.shape[1], state.X.shape[1]
    counts = state.Z.sum(0)
    prior_odds = np.log(counts) - np.log(N_train - counts + 1)

    fallback_var = np.mean(state.X**2)

    if maximize:
        #old_state = state
        state = state.copy()
        cache = IBPCache.from_state(model, data, state, np.ones(N_train, dtype=bool))
        #assert False
        state.A = cache.fpost.mu
    

    total = 0.
    for i in range(N_test):
        x = state.X[i, :]
        Lambda = np.dot(state.A, state.A.T) / state.sigma_sq_n
        h = np.dot(state.A, x) / state.sigma_sq_n
        c = -0.5 * D * np.log(state.sigma_sq_n) + \
            -0.5 * D * np.log(2*np.pi) + \
            -0.5 * np.dot(x, x) / state.sigma_sq_n
        z, obj = mean_field(h, Lambda, c, prior_odds)

        fallback = -0.5 * D * np.log(2*np.pi) + \
                   -0.5 * D * np.log(fallback_var) + \
                   -0.5 * np.dot(x, x) / fallback_var
        
        total += np.logaddexp(obj + np.log(0.99), fallback + np.log(0.01))

    return total / N_test
    

    

def solve2(N, K, D, p, niter):
    raise NotImplementedError()
    model, both_data, _ = random_instance(True, N*2, K, D, p)
    train_data = SimpleData(both_data.X[:N, :])
    test_data = SimpleData(both_data.X[N:, :])
    state_no = random_initialization(model, train_data)
    state_yes = state_no.copy()

    scores_no = []
    scores_yes = []
    scores_no_max = []
    scores_yes_max = []

    for it in range(niter):
        print it
        gibbs_sweep(model, train_data, state_no, False)
        gibbs_sweep(model, train_data, state_yes, True)

        score_no = score_predictive(model, test_data, state_no)
        scores_no.append(score_no)
        print '    Without:', score_no
        print '        K =', state_no.Z.shape[1]
        score_yes = score_predictive(model, test_data, state_yes)
        print '    With:', score_yes
        print '        K =', state_yes.Z.shape[1]
        scores_yes.append(score_yes)

        score_no_max = score_predictive(model, test_data, state_no, True)
        scores_no_max.append(score_no_max)
        print '    Without (max):', score_no_max
        score_yes_max = score_predictive(model, test_data, state_yes, True)
        print '    With (max):', score_yes_max
        scores_yes_max.append(score_yes_max)

        if (it+1) % 5 == 0:
            pylab.figure(1)
            pylab.clf()
            pylab.plot(range(it+1), scores_no, 'r-', range(it+1), scores_yes, 'g-',
                       range(it+1), scores_no_max, 'r--', range(it+1), scores_yes_max, 'g--')
            pylab.legend(['without', 'with'], loc='lower right')
            pylab.draw()

        
        
def solve3(N, K, D, p, niter):
    raise NotImplementedError()
    model, both_data, _ = random_instance(True, N*2, K, D, p)
    train_data = SimpleData(both_data.X[:N, :])
    test_data = SimpleData(both_data.X[N:, :])
    state_no = random_initialization(model, train_data)
    state_yes = sparse_coding_init(train_data.X)

    scores_no = []
    scores_yes = []

    for it in range(niter):
        print it
        gibbs_sweep(model, train_data, state_no, True)
        gibbs_sweep(model, train_data, state_yes, True)

        score_no = score_predictive(model, test_data, state_no)
        scores_no.append(score_no)
        print '    Random:', score_no
        score_yes = score_predictive(model, test_data, state_yes)
        print '    Sparse coding:', score_yes
        scores_yes.append(score_yes)

        if (it+1) % 5 == 0:
            pylab.figure(1)
            pylab.clf()
            pylab.plot(range(it+1), scores_no, 'r-', range(it+1), scores_yes, 'b-')
            pylab.legend(['random', 'sparse coding'], loc='lower right')
            pylab.draw()


