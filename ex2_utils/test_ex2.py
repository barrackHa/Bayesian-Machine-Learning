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
    print(model.h(X[1]), model.h(X[1]).shape)
    # t = model.fit(X, y)
    # print(t, t.shape)

    # fig, ax = plt.subplots()
    # ax.plot(X, y, 'o')
    # ax.plot(X, model.predict(X))
    # xx = np.arange(0, 7, .05)
    # ax.plot(xx, model.predict(xx))
    # plt.show()
    assert False
    

if __name__ == '__main__':
    pytest.main()