import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from pathlib import Path


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        # <your code here>
        return None
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        # <your code here>
        return None
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        # <your code here>
        return None
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        # todo <your code here>
        thetas.append(None)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        # <your code here>
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions

        # x is a vector in R_d
        # h(x) = [h1(x),...,hn(x)] and foreach i, hi(x) is a function from R_d to R
        # h.shape = (n,) though h doesn't really have shape as it's a function
        self.h = lambda x: np.array(
            [np.squeeze(h(x)) for h in basis_functions]
        )
        
        # X is a matix in R_dxm such that it has m vectors in R_d (as x above)
        # H(X) = [h(x1),...,h(xm)]^T foreach xi row of the matrix X
        self.H = lambda X: np.apply_along_axis(self.h, 1, X)

        return

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # <your code here>
        H = self.H(X)
        theta_cov_inv = np.linalg.pinv(self.theta_cov)
        cov_theta_D = np.linalg.pinv(theta_cov_inv + (1 / self.sig**2) * (H.T @ H))
        mu_theta_D = cov_theta_D @ (theta_cov_inv @ self.theta_mean + (1 / self.sig**2) * (H.T @ y))
        self.mu_theta_D = mu_theta_D
        self.cov_theta_D = cov_theta_D

        # # theta_cov_det = np.linalg.det(self.theta_cov)
        # # normslizer = 1 / np.sqrt(np.li)
        # # pdf = np.exp()
        return cov_theta_D, mu_theta_D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        return self.H(X).dot(self.mu_theta_D)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # <your code here>
        return None

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        return None


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions: a function that receives data points as inputs and returns a design matrix
        """
        # <your code here>
        self.basis_functions = basis_functions
        
        # x is a vector in R_d
        # h(x) = [h1(x),...,hn(x)] and foreach i, hi(x) is a function from R_d to R
        # h.shape = (n,) though h doesn't really have shape as it's a function
        self.h = lambda x: np.array(
            [np.squeeze(h(x)) for h in basis_functions]
        )
        
        # X is a matix in R_dxm such that it has m vectors in R_d (as x above)
        # H(X) = [h(x1),...,h(xm)]^T foreach xi row of the matrix X
        self.H = lambda X: np.apply_along_axis(self.h, 1, X)
        return 

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # <your code here>
        H = self.H(X)
        theta = np.linalg.pinv(H.T @ H) @ H.T @ y
        self.theta = theta
        return theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here> 
        # H(X)*theta is a vector in which each element is 
        # the prediction for the corresponding x in X, 
        # i.e. evalute the function h(x) times the wieght given by theta 
        return self.H(X).dot(self.theta)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def main():
    # load the data for November 16 2020
    nov16 = np.load(Path(__file__).parent / 'nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours))**2):.2f}')

        # plot graphs for linear regression part
        # <your code here>

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 2.5  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([3, 6, 9, 12, 15, 18, 21])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>

    # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)

        # plot prior graphs
        # <your code here>

        # plot posterior graphs
        # <your code here>


if __name__ == '__main__':
    main()
