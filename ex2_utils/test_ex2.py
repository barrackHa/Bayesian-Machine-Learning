import pytest
from ex2 import *
from matplotlib import pyplot as plt

def test_LinearRegression():
    # test that the linear regression model works
    deg = 1
    model = LinearRegression(polynomial_basis_functions(deg))
    X = np.arange(1,7)
    y = np.array(X/deg)

    H = model.H(X)
    assert np.array_equal(H, np.array([[1, i] for i in range(1, 7)]))
    
    model.fit(X, y)
    t = model.theta
    assert np.allclose(t, np.array([0, 1]), atol=1e-5)

    new_X = np.array([[7], [8], [9], [10]])
    predictied = model.predict(new_X)
    assert np.allclose(predictied, np.array([7, 8, 9, 10]), atol=1e-5)
    
    # # Uncoment to plot and see results
    # fig, ax = plt.subplots()
    # ax.plot(X, y, 'o')
    # ax.plot(X, model.predict(X))
    # xx = np.arange(-6, 6, .05)
    # new_X = xx.reshape((xx.size, 1))
    # ax.plot(xx, model.predict(new_X))
    # plt.show()
    
def test_deg2_LinearRegression():
    # test that the linear regression model works
    deg = 2
    model = LinearRegression(polynomial_basis_functions(deg))
    X = np.arange(1,6)
    print(X)
    # y = x^2
    y = np.array((X/deg)**2)
    H = model.H(X)

    assert np.array_equal(H, np.array([[1, i/deg, (i/deg)**2] for i in range(1, 6)]))
    
    model.fit(X, y)
    t = model.theta
    assert np.allclose(t, np.array([0, 0, 1]), atol=1e-5)

    test = np.array([-1, -2, -3, -4])
    predictied = model.predict(test)
    assert np.allclose(predictied, (test/deg)**2, atol=1e-5)
    
    # # Uncoment to plot and see results
    # fig, ax = plt.subplots()
    # ax.plot(X, y, 'o')
    # ax.plot(X, model.predict(X))
    # xx = np.arange(-6, 6, .05)
    # new_X = xx.reshape((xx.size, 1))
    # ax.plot(xx, model.predict(new_X))
    # plt.show()

def test_BayesianLinearRegression():
    deg = 1
    baseis_funcs = polynomial_basis_functions(deg)
    theta_mean = np.array([0,1])
    # theta_cov = np.array([1]) 
    theta_cov = np.eye(2)
    sig = 1
    blr = BayesianLinearRegression(theta_mean, theta_cov, sig, baseis_funcs)
    X = np.arange(1,7)
    y = np.arange(1,7)

    blr.fit(X, y)
    S, mu = blr.cov_theta_D, blr.mu_theta_D
    test = blr.predict(X)
    assert np.allclose(test, y, atol=np.sqrt(sig))    

    _, ax = plt.subplots()

    ax.scatter(x=X, y=y, s=20, marker='o', c='r')
    ax.plot(X, test, color='b', lw=2, label='orig')
    xx = np.arange(-6, 8, .05)
    ax.set_title(f'Bayesian Linear Regression\nmu={mu}\nS={S}')
    mean = blr.predict(xx)
    ax.plot(xx, mean, 'k', lw=3, label='mean')
    std = blr.predict_std(xx)
    ax.fill_between(xx, mean-std, mean+std, alpha=.5, label='confidence interval')
    
    for _ in range(4):
        rand = blr.posterior_sample(xx) 
        ax.plot(xx, rand, alpha=.5)
    ax.legend()    
    # plt.show()

def test_polynomial_basis_functions():
    deg, N = 2, 5
    H = polynomial_basis_functions(deg)
    x = np.ones(N)
    H = H(x)
    tester = np.array([[1, 0.5, 0.25]]*N)
    assert np.array_equal(H, tester)
    assert np.array_equal(H[:, 0], np.ones(N))
    assert H.shape == (N, deg+1)

def test_gaussian_basis_functions():
    N = 5
    x = np.ones(N)
    beta = 2
    centers = np.array([1,2,3])
    H = gaussian_basis_functions(centers=centers, beta=beta)
    H = H(centers)

    assert np.array_equal(H[:, 1:], H[:, 1:].T)
    assert np.array_equal(np.diag(H[:, 1:]), np.ones_like(centers))
    assert np.array_equal(H[:, 0], np.ones_like(centers))
    assert H[0,2] == np.exp(np.power(1-2,2) / (-2 * np.power(beta,2)))
    assert H[0,3] == np.exp(np.power(1-3,2) / (-2 * np.power(beta,2)))
    assert H.shape == (centers.shape[0], centers.shape[0] + 1)

    H = gaussian_basis_functions(centers=centers, beta=beta)
    H = H(x)
    assert H[0,2] == np.exp(np.power(1-2,2) / (-2 * np.power(beta,2)))
    assert H[0,3] == np.exp(np.power(1-3,2) / (-2 * np.power(beta,2)))
    assert H.shape == (N, centers.shape[0] + 1)

def test_spline_basis_functions():
    N = 5
    x = 2*np.ones(N)
    x[-1] = 8
    knots = np.array([6,12,18])
    H = spline_basis_functions(knots)
    H = H(x)
    assert H.shape == (N, knots.size + 4)
    tester = np.array([1, 8, 64, 512, 8, 0, 0])
    assert np.array_equal(H[-1], tester)
    tester = np.array([1, 2, 4, 8, 0, 0, 0])
    for i in range(N-1):
        assert np.array_equal(H[i], tester)
    assert np.array_equal(H[:, 0], np.ones(N))

def test_learn_prior():
    deg = 1
    pbf = polynomial_basis_functions(1)
    temps = np.load(Path(__file__).parent / 'jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)
    tmp = np.sum(temps, axis=0) / temps.shape[0]
    print(temps[1], temps.shape, tmp.shape, hours.shape)
    fig, ax = plt.subplots()
    variance = np.var(temps, axis=0)
    ax.bar(hours, tmp, yerr=variance)
    ax.set_xticks(hours)
    mu, var = np.mean(tmp), np.var(tmp)
    ax.set_title('Mean: {:.2f}, Variance: {:.2f}'.format(mu, var))
    # plt.show()
    
    mu, cov = learn_prior(hours, temps, pbf)
    print(mu.shape, cov.shape)
    # assert False



if __name__ == '__main__':
    pytest.main()