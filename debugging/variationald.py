



def test_moments(rep):
    NUM_SAMPLES = 10000
    samples = np.array([rep.sample() for i in range(NUM_SAMPLES)])

    pred = rep.expected_value()
    actual = np.mean(samples, axis=0)
    assert np.max(np.abs(pred - actual)) < 0.01

    pred = rep.covariance()
    actual = np.mean(samples[:,:,nax] * samples[:,nax,:], axis=0) - np.outer(samples.mean(0), samples.mean(0))
    assert np.max(np.abs(pred - actual)) < 0.01
    
        
    

    

def test_variational(check_moments=False):
    m, k, n = 100, 15, 30
    ESTIMATOR_CLASSES = [MultinomialEstimator, BernoulliEstimator]
    
    num_estimators = len(ESTIMATOR_CLASSES)
    temp = np.random.normal(size=(n, n))
    Sigma_N = np.dot(temp.T, temp)
    Sigma_N /= np.mean(np.diag(Sigma_N))
    
    estimators = [EstimatorClass.random(k, n) for EstimatorClass in ESTIMATOR_CLASSES]
    right_multipliers = [e.A for e in estimators]
    u_all = [EstimatorClass.random_u(k) for EstimatorClass in ESTIMATOR_CLASSES]

    reps = [estimator.init_representation() for estimator in estimators]
    x = np.zeros(n)
    for i in range(num_estimators):
        x += np.dot(right_multipliers[i].T, u_all[i])
    x += np.random.multivariate_normal(np.zeros(n), Sigma_N)
    problem = VariationalProblem(estimators, x, Sigma_N)
    fobj = problem.objective_function(reps)
    print 'Initial objective function: %1.3f' % fobj

    NUM_ITER = 10
    for it in range(NUM_ITER):
        for i in range(num_estimators):
            reps = problem.update_one(reps, i)
            fobj = problem.objective_function(reps)
            print 'Updated %d; new objective function: %1.3f' % (i, fobj)

            if check_moments:
                test_moments(reps[i])

            EPS = 1e-8
            for p in range(10):
                reps_pert = reps[:]
                reps_pert[i] = reps_pert[i].perturb(EPS)
                fobj_pert = problem.objective_function(reps_pert)
                print 'Difference:', fobj - fobj_pert
