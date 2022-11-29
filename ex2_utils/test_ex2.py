import pytest
from ex2 import *
from matplotlib import pyplot as plt

def test_ex2():
    assert True

def test_linear_regression():
    # test that the linear regression model works
    model = LinearRegression([lambda x: 1, lambda x: x])
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([1, 2, 3, 4, 5, 6])

    assert np.array_equal(model.h(X[1]), np.array([1, 2]))
    assert np.array_equal(model.H(X), np.array([[1, i] for i in range(1, 7)]))
    
    t = model.fit(X, y)
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
    
def test_deg2_linear_regression():
    # test that the linear regression model works
    model = LinearRegression([lambda x: 1, lambda x: x, lambda x: x**2])
    X = np.array([[1], [2], [3], [4], [5]])
    # y = x^2
    y = np.array([1, 4, 9, 16, 25])

    assert np.array_equal(model.h(X[1]), np.array([1, 2, 4]))
    assert np.array_equal(model.H(X), np.array([[1, i, i*i] for i in range(1, 6)]))
    
    t = model.fit(X, y)
    assert np.allclose(t, np.array([0, 0, 1]), atol=1e-5)

    new_X = np.array([[-1], [-2], [-3], [-4]])
    predictied = model.predict(new_X)
    assert np.allclose(predictied, np.array([1, 4, 9, 16]), atol=1e-5)
    
    # # Uncoment to plot and see results
    # fig, ax = plt.subplots()
    # ax.plot(X, y, 'o')
    # ax.plot(X, model.predict(X))
    # xx = np.arange(-6, 6, .05)
    # new_X = xx.reshape((xx.size, 1))
    # ax.plot(xx, model.predict(new_X))
    # plt.show()


if __name__ == '__main__':
    pytest.main()