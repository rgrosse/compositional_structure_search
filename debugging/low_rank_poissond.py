
def print_predictive(resid, obs, ssq_N, proposal, order=None):
    pi = ProposalInfo(resid, obs, ssq_N)
    N, D = resid.shape
    dist_without = distributions.GaussianDistribution(0., ssq_N)
    if order is None:
        order = range(N)

    total_with = 0.
    total_without = 0.
    for i in order:
        idxs = np.where(obs[i, :])[0]

        if PRINT_ALL and i % 20 == 0:
            print 'ssq_u', pi.ssq_u
            print 'ssq_N', ssq_N
            print 'pi.v', pi.v
            print 'proposal.v', proposal.v
            print 'u[i]', proposal.u[i]

        if i == order[0]:
            Sigma = ssq_N * np.eye(idxs.size)
        else:
            Sigma = pi.ssq_u * np.outer(pi.v[idxs], pi.v[idxs]) + ssq_N * np.eye(idxs.size)
        pot = gaussians.Potential.from_moments_full(np.zeros(idxs.size), Sigma)
        score_with = pot.score(resid[i, idxs])
        total_with += score_with
        
        score_without = dist_without.loglik(resid[i, :])[idxs].sum()
        total_without += score_without

        pi.update_u(i, proposal.u[i])
        pi.fit_v_and_var()


        if PRINT_ALL:
            print 'i =', i
            print 'Score with:', score_with
            print 'Score without:', score_without
            print

    print 'Total with:', total_with
    print 'Total without:', total_without



K_INIT_ALL = [2, 5, 10, 20, 50]

def fit_model_parallel(data_matrix, num_iter=NUM_ITER, name=None):
    N, D = data_matrix.m, data_matrix.n
    states = []
    for K in K_INIT_ALL:
        X, st = init_state(data_matrix, K)
        states.append(st)

    # temporary
    ssq_N_log = []

    K_log = []
    for it in range(num_iter):
        for j, state in enumerate(states):
            sample_U_V(state, X, data_matrix.observations.mask)

            givens_moves(state)
            scaling_moves(state)

            print 'ssq_U', sorted(state.ssq_U)


            state.ssq_U = sample_variance(state.U, 0)
            pred = np.dot(state.U, state.V)
            if not data_matrix.observations.fixed_variance():
                state.ssq_N = sample_variance(X - pred, None, mask=data_matrix.observations.mask)

            print 'ssq_N', state.ssq_N

            #state.ssq_N = 0.1

            X = data_matrix.sample_latent_values(pred, state.ssq_N)

            for i in range(10):
                states[j] = add_delete_move(states[j], X, data_matrix.observations.mask)

        misc.print_dot(it+1, num_iter)

        K_log.append([s.U.shape[1] for s in states])


        # temporary
        ssq_N_log.append([s.ssq_N for s in states])

        title = 'Convergence of low-rank model (%s)' % name
        vis.pw.figure(title)
        pylab.clf()
        pylab.plot(range(1, it+2), K_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('Number of dimensions')

        title = 'Convergence of low rank model (%s, log-log)' % name
        vis.pw.figure(title)
        pylab.clf()
        pylab.loglog(range(1, it+2), K_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('Number of dimensions')
        pylab.draw()

        # temporary
        title = 'sigma_sq_N'
        vis.pw.figure(title)
        pylab.clf()
        pylab.semilogy(range(1, it+2), ssq_N_log)
        pylab.title(title)
        pylab.xlabel('Iteration')
        pylab.ylabel('sigma_sq_N')
        pylab.draw()
        



def run():
    dm = data.load_senate_data(2008, real_valued=True)
    U, V, ssq_U, ssq_V, ssq_N, X = low_rank.fit_model(dm, 20, num_iter=2, rotation_trick=False)
    N, K, D = U.shape[0], U.shape[1], V.shape[1]

    #norm = (U**2).sum(0)
    #idxs = np.argsort(norm)
    #i, j = idxs[-2], idxs[-1]

    for tr in range(10):
        a = np.random.randint(0, K)
        b = np.random.randint(0, K)
        if a == b:
            continue

        theta = np.linspace(-np.pi, np.pi)
        uaa = np.dot(U[:, a], U[:, a])
        uab = np.dot(U[:, a], U[:, b])
        ubb = np.dot(U[:, b], U[:, b])

        sin, cos = np.sin(theta), np.cos(theta)
        uaa_prime_ssq = uaa * cos ** 2 + 2 * uab * cos * sin + ubb * sin ** 2
        ubb_prime_ssq = uaa * sin ** 2 - 2 * uab * cos * sin + ubb * cos ** 2
        prob = -(A + 0.5 * N) * (np.log(B + 0.5 * uaa_prime_ssq) + np.log(B + 0.5 * ubb_prime_ssq))

        pylab.figure()
        pylab.plot(theta, prob)


def run2():
    dm = data.load_senate_data(2008, real_valued=True)
    U, V, ssq_U, ssq_V, ssq_N, X = low_rank.fit_model(dm, 20, num_iter=2, rotation_trick=False)
    N, K, D = U.shape[0], U.shape[1], V.shape[1]

    vis.pw.figure('before')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))
    #print np.dot(U.T, U)

    state = State(U, V, None, None)
    givens_moves(state)

    vis.pw.figure('after')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))

    givens_moves(state)
    vis.pw.figure('after 2')
    vis.display(vis.norm01(np.abs(np.dot(U.T, U))))
    


def check_integral():
    u1 = np.random.normal(size=20)
    u2 = np.random.normal(size=20)
    exact1 = p_u(u1)
    exact2 = p_u(u2)

    #lam = np.logspace(-4., 4., 100000)
    npts = 10000
    lam = np.linspace(0., 4., npts)
    p_lam = A * np.log(lam) - B * lam
    p_u1_given_lam = distributions.gauss_loglik(u1[:, nax], 0., 1. / lam[nax, :]).sum(0)
    p_u2_given_lam = distributions.gauss_loglik(u2[:, nax], 0., 1. / lam[nax, :]).sum(0)
    approx1 = np.log(scipy.integrate.trapz(np.exp(p_lam + p_u1_given_lam), lam))
    approx2 = np.log(scipy.integrate.trapz(np.exp(p_lam + p_u2_given_lam), lam))
    
    print 'exact1 - exact2', exact1 - exact2
    print 'approx1 - approx2', approx1 - approx2

def check_integral2():
    exact = np.zeros(100)
    approx = np.zeros(100)
    u0 = np.random.normal(size=20)
    alpha_all = np.linspace(0., 1., 100)
    for i, alpha in enumerate(alpha_all):
        exact[i] = p_u(alpha * u0)
        
        npts = 1000000
        lam = np.linspace(0., 1000., npts)
        p_lam = A * np.log(lam) - B * lam
        p_u_given_lam = distributions.gauss_loglik(alpha * u0[:, nax], 0., 1. / lam[nax, :]).sum(0)
        approx[i] = np.log(scipy.integrate.simps(np.exp(p_lam + p_u_given_lam), lam))

        misc.print_dot(i+1, len(alpha_all))

    exact -= exact[-1]
    approx -= approx[-1]
    pylab.figure()
    pylab.plot(alpha_all, exact, 'b', alpha_all, approx, 'r')

def check_close(a, b):
    if not np.allclose([a], [b]):   # array brackets to avoid an error comparing inf and inf
        raise RuntimeError('a=%f, b=%f' % (a, b))
    
def check_cond_U_Vt(missing=False):
    N, K, D = 20, 15, 30
    ssq_U = np.random.uniform(0., 1., size=K)
    ssq_N = np.random.uniform(0., 1.)
    U = np.random.normal(0., np.sqrt(ssq_U[nax, :]), size=(N, K))
    V = np.random.normal(0., 1., size=(K, D))
    X = np.random.normal(np.dot(U, V), np.sqrt(ssq_N))
    state = State(U, V, ssq_U, ssq_N)
    if missing:
        mask = np.random.binomial(1, 0.5, size=(N, D)).astype(bool)
    else:
        mask = np.ones((N, D), dtype=bool)

    U_new = np.random.normal(size=(N, K))
    new_state = state.copy()
    new_state.U = U_new
    cond = cond_U(X, mask, V, ssq_U, ssq_N)
    check_close(cond.score(U_new).sum() - cond.score(U).sum(),
                p_star(new_state, X, mask) - p_star(state, X, mask))

    V_new = np.random.normal(size=(K, D))
    new_state = state.copy()
    new_state.V = V_new
    cond = cond_Vt(X, mask, U, ssq_N)
    check_close(cond.score(V_new.T).sum() - cond.score(V.T).sum(),
                p_star(new_state, X, mask) - p_star(state, X, mask))
    
    

def generate_synthetic(noise_var):
    #N, K, D = 1000, 10, 1050
    #N, K, D = 100, 10, 150
    N, K, D = 100, 10, 150
    U = np.random.normal(size=(N, K))
    V = np.random.normal(size=(K, D))
    X = np.dot(U, V)
    X /= np.sqrt(np.mean(X ** 2))
    X = np.random.normal(X, np.sqrt(noise_var))
    return observations.DataMatrix.from_real_values(X)
    
    

def run_real(name, parallel=False, run=1, num_iter=100):
    if name == 'senate':
        dm = data.load_senate_data(2008, real_valued=True)
    elif name == 'intel':
        dm = data.load_intel_data(real_valued=True)
    elif name == 'senate-binary':
        dm = data.load_senate_data(2008)
    elif name == 'intel-integer':
        dm = data.load_intel_data()
    elif name == 'senate-noiseless':
        dm = data.load_senate_data(2008, real_valued=True, noise_variance=1e-5)
    elif name == 'intel-noiseless':
        dm = data.load_intel_data(real_valued=True, noise_variance=1e-5)
    elif name[:9] == 'synthetic':
        noise_var = float(name[10:])
        dm = generate_synthetic(noise_var)
    elif name == 'movielens-integer':
        dm = movielens_data.load_data_matrix()
        dm = dm[:, :1000]
    elif name == 'movielens-real':
        dm = movielens_data.load_data_matrix(True)
        dm = dm[:, :1000]
    elif name == 'patches':
        X = data.load_image_patches(1000)
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'patches2':
        X = data.load_image_patches(1000)
        X = X[:, ::2]
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'temp-intel':
        dm = data.load_intel_data(real_valued=True, noise_variance=0.1)
    elif name == 'temp-patches':
        X = data.load_image_patches(4000)
        cols = np.random.permutation(144)[:115]
        X = X[:, cols]
        X /= X.std()
        X = np.random.normal(X, 0.1)
        dm = observations.DataMatrix.from_real_values(X)
    elif name == 'temp-patches2':
        dm = cPickle.load(open(experiments.data_file('image-patches-6-22')))
        splits = cPickle.load(open(experiments.splits_file('image-patches-6-22')))
        train_rows, train_cols, test_rows, test_cols = splits[0]
        train_cols = np.random.permutation(144)[:144*4//5]
        train_rows = np.arange(500)
        dm = dm[train_rows[:, nax], train_cols[nax, :]]
    name += ' run %d' % run
    
    if parallel:
        fit_model_parallel(dm, num_iter=num_iter, name=name)
    else:
        fit_model(dm, num_iter=num_iter, name=name)

def run_all():
    #NAMES = ['synthetic-1', 'synthetic-0.1', 'senate', 'senate-binary', 'intel', 'intel-integer']
    NAMES = ['intel', 'intel-integer']
    for name in NAMES:
        for i in range(1, 4):
            print name, i
            run_real(name, True, i)
        vis.save_all_figures('/tmp/roger/low-rank')


import initialization
import recursive
import grammar
import scoring

def compare_old_and_new(name):
    NUM_SAMPLES = 2
    
    if name == 'senate':
        dm = data.load_senate_data(2008, real_valued=True)
    elif name == 'intel':
        dm = data.load_intel_data(real_valued=True)
    elif name == 'senate-binary':
        dm = data.load_senate_data(2008)
    elif name == 'intel-binary':
        dm = data.load_intel_data()
    elif name[:9] == 'synthetic':
        noise_var = float(name[10:])
        dm = generate_synthetic(noise_var)

    train_rows = np.arange(0, dm.m, 2)
    test_rows = np.arange(1, dm.m, 2)
    train_cols = np.arange(0, dm.n, 2)
    test_cols = np.arange(1, dm.n, 2)

    X_train = dm[train_rows[:, nax], train_cols[nax, :]]
    X_row_test = dm[test_rows[:, nax], train_cols[nax, :]]
    X_col_test = dm[train_rows[:, nax], test_cols[nax, :]]

    old_row_loglik = []
    old_col_loglik = []
    new_row_loglik = []
    new_col_loglik = []
    


    for i in range(NUM_SAMPLES):
        initialization.USE_OLD_LOW_RANK = True
        old_root = recursive.fit_model(grammar.parse('gg+g'), X_train, gibbs_steps=50)
        rl, cl = scoring.evaluate_model(X_train, old_root, X_row_test, X_col_test)
        old_row_loglik.append(rl)
        old_col_loglik.append(cl)

        initialization.USE_OLD_LOW_RANK = False
        new_root = recursive.fit_model(grammar.parse('gg+g'), X_train, gibbs_steps=50)
        rl, cl = scoring.evaluate_model(X_train, new_root, X_row_test, X_col_test)
        new_row_loglik.append(rl)
        new_col_loglik.append(cl)

    old_row_loglik = np.array(old_row_loglik)
    old_col_loglik = np.array(old_col_loglik)
    new_row_loglik = np.array(new_row_loglik)
    new_col_loglik = np.array(new_col_loglik)

    print 'Average row log-likelihood (each sample)'
    print '    Old:', old_row_loglik.mean(1)
    print '    New:', new_row_loglik.mean(1)
    print 'Average column log-likelihood (each sample)'
    print '    Old:', old_col_loglik.mean(1)
    print '    New:', new_col_loglik.mean(1)
    print 'Average row log-likelihood (combined)'
    print '    Old:', np.logaddexp.reduce(old_row_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print '    New:', np.logaddexp.reduce(new_row_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print 'Average column log-likelihood (combined)'
    print '    Old:', np.logaddexp.reduce(old_col_loglik, 0).mean() - np.log(NUM_SAMPLES)
    print '    New:', np.logaddexp.reduce(new_col_loglik, 0).mean() - np.log(NUM_SAMPLES)


    
