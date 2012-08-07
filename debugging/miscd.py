



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
    
