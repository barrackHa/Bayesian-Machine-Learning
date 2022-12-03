import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from pathlib import Path


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to 
    (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and 
            returns the design matrix of the polynomial basis functions, 
            a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        # <your code here>
        H = [np.power((x/degree), i) for i in range(0, degree+1)]
        return np.array(H).T
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and 
            returns the design matrix of the Gaussian basis functions, 
            a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        # <your code here>
        phi0 = np.ones_like(x)
        phi = np.array([
            np.exp(np.power((x - c*np.ones_like(x)), 2) / (-2*np.power(beta, 2))) 
            for c in centers
        ])
        return np.vstack((phi0, phi)).T
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline 
    basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N 
            and returns the design matrix of the cubic regression 
            spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        # <your code here>
        prefix = np.array([x**i for i in range(0, 4)])
        suffix = np.array([(x-xi) for xi in knots])
        suffix = np.where(suffix < 0, 0, suffix**3)
        return np.vstack((prefix, suffix)).T
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, 
                  as loaded from 'jerus_daytemps.npy', with shape [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis 
                       functions to be used
    :return: the mean and covariance of the learned covariance - 
             the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], 
             where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func)
        theta = ln.fit(hours, t)
        # todo <your code here>
        thetas.append(theta)  # append learned parameters here

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
        self.H = self.basis_functions = basis_functions

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

        # Using Woodbury identity from Rec4
        M_inverse = np.linalg.pinv(
            self.sig**2 * np.eye(H.shape[0]) + \
            H @ self.theta_cov @ H.T
        )
         
        mu_theta_D =  self.theta_cov @ H.T @ M_inverse
        mu_theta_D = mu_theta_D @ (y - H @ self.theta_mean)
        mu_theta_D += self.theta_mean

        cov_theta_D = self.theta_cov - \
                    self.theta_cov @ H.T @ M_inverse @ H @ self.theta_cov

        self.mu_theta_D = mu_theta_D # AKA theta MMSE
        self.cov_theta_D = cov_theta_D
        self.chol = np.linalg.cholesky(cov_theta_D)

        return cov_theta_D, mu_theta_D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        return self.H(X).dot(self.mu_theta_D)

    def prior_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X using the prior
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.H(X).dot(self.theta_mean)

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
        H = self.H(X)
        std = np.sqrt(np.diagonal(H@self.cov_theta_D@H.T))

        return std

    def prior_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the Priors's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.H(X)
        std = np.sqrt(np.diagonal(H@self.theta_cov@H.T))

        return std

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        H = self.H(X)
        # print(self.chol)
        # rand_theta = np.random.normal(self.theta_mean, self.chol)
        # print(self.cov_theta_D)
        # print(np.linalg.eigvals(self.cov_theta_D))
        # rand_theta = np.random.normal(self.mu_theta_D, self.cov_theta_D)
        rand_theta = self.mu_theta_D + \
                self.chol@np.random.randn(self.chol.shape[-1]) 

        return (H @ rand_theta)

    def praior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the prior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # <your code here>
        H = self.H(X)
        chol = np.linalg.cholesky(self.theta_cov)
        # print(chol)
        # rand_theta = np.random.normal(self.theta_mean, chol)
        rand_theta = self.theta_mean + \
                chol@np.random.randn(chol.shape[-1]) 

        return (H @ rand_theta)


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions: a function that receives data points as inputs and returns a design matrix
        """
        # <your code here>
        self.H = self.basis_functions = basis_functions
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

def plot_results(
    ax: plt.axes, train_hours: np.ndarray, train: np.ndarray, 
    x: np.ndarray, model: np.ndarray, model_lbl: str,
    test_hours: np.ndarray, test: np.ndarray, pred: np.ndarray,
    ax_title: str
    ):
    """
    Plot the results of the regression
    """

    ax.scatter(train_hours, train, label='Train')
    ax.scatter(test_hours, test, label='Test')
    ax.scatter(test_hours, pred, label='Predictions')
    ax.plot(x, model, label=model_lbl, alpha=1, lw=2, c='black')
    ax.set_xlabel('Hours')
    ax.set_ylabel('temperature [C]')
    ax.set_title(f'{ax_title}')
    
    ax.legend()
    ax.grid()

    return ax
    
def add_filling_and_samples(
    ax: plt.axes, x: np.ndarray, model: np.ndarray, 
    std: np.ndarray, samples_func: np.ndarray
    ):
    ax.fill_between(x, model-std, model+std, alpha=.5, label='confidence interval', color='#A4DBE8')
    ax.legend()

    for i in range(5):
        tmp = samples_func(x)
        ax.plot(x, tmp, alpha=.5, label=f'Sample #{i}')
    
    return ax


def main(show=False):
    # load the data for November 16 2020
    nov16 = np.load(Path(__file__).parent / 'nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]


    # setup the model parameters
    degrees = [3, 7]
    x = np.arange(0, 24, .1)
    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d))
        ln.fit(train_hours, train)

        # print average squared error performance
        pred = ln.predict(test_hours)
        
        avg_sq_err = np.mean((test - pred)**2)
        print(f'Average squared error with LR and d={d} is {avg_sq_err:.2f}')

        # plot graphs for linear regression part
        # <your code here>
        model = ln.predict(x)
        model_lbl = f'Linear Regression Model of degree {d}'
        fig, ax = plt.subplots()
        fig.suptitle(f'Linear Regression Model\n Polynomial of degree {d}')
        ax_title = f'Average squared error with LR and d={d} is {avg_sq_err:.2f}'
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred, ax_title
        )

    
    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load(Path(__file__).parent / 'jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = np.sqrt(0.25)
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 3  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([2, 4, 8, 12, 16, 20, 22])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for i, deg in enumerate(degrees):
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        blr_cov_theta_D, blr_mu_theta_D = blr.fit(train_hours, train)
        pred_post = blr.predict(test_hours)
        pred_prior = blr.prior_predict(test_hours)
        avg_sq_err = np.mean((test - pred_post)**2)
        print(f'Average squared error with Bayesian Linear Regression Model and d={d} is {avg_sq_err:.2f}')

        # plot prior graphs
        # <your code here>
        fig, ax = plt.subplots()
        fig.suptitle(f'Bayesian Linear Regression Model\nDegree {deg} Polynomial\n ')
        
        model_lbl = f'Prior'
        ax_title = f'Prior Graph'
        model = blr.prior_predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_prior, ax_title
        )

        std = blr.prior_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.praior_sample)

        # plot posterior graphs
        # <your code here>
        fig, ax = plt.subplots()
        fig.suptitle(f'Bayesian Linear Regression Model\nDegree {deg} Polynomial\n ')
        
        model_lbl = f'Postior'
        ax_title = f'Postior Graph\n'
        ax_title += f'Average squared error with BLR and d={d} is {avg_sq_err:.2f}'
        model = blr.predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_post, ax_title
        )

        std = blr.predict_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.posterior_sample)

    # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        blr_cov_theta_D, blr_mu_theta_D = blr.fit(train_hours, train)
        pred_post = blr.predict(test_hours)
        pred_prior = blr.prior_predict(test_hours)
        avg_sq_err = np.mean((test - pred_post)**2)
        basis_type = f'S_{ind+1} Gaussian basis functions (Beta={beta})'
        print(f'Average squared error with Bayesian Linear Regression Model {basis_type} is {avg_sq_err:.2f}')

        # plot prior graphs
        # <your code here>
        fig, ax = plt.subplots()
        
        fig.suptitle(f'Bayesian Linear Regression Model\n{basis_type}')
        
        model_lbl = f'Prior'
        ax_title = f'Prior Graph'
        model = blr.prior_predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_prior, ax_title
        )

        std = blr.prior_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.praior_sample)

        # plot posterior graphs
        # <your code here>
        fig, ax = plt.subplots()
        fig.suptitle(f'Bayesian Linear Regression Model\n{basis_type}')
        
        model_lbl = f'Postior'
        ax_title = f'Postior Graph\n'
        ax_title += f'Average squared error with BLR and {basis_type} is {avg_sq_err:.2f}'
        model = blr.predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_post, ax_title
        )

        std = blr.predict_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.posterior_sample)


    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)

        blr_cov_theta_D, blr_mu_theta_D = blr.fit(train_hours, train)
        pred_post = blr.predict(test_hours)
        pred_prior = blr.prior_predict(test_hours)
        avg_sq_err = np.mean((test - pred_post)**2)
        basis_type = f'K_{ind+1} Cubic Regression Splines'
        print(f'Average squared error with Bayesian Linear Regression Model {basis_type} is {avg_sq_err:.2f}')

        # plot prior graphs
        # <your code here>
        fig, ax = plt.subplots()
        
        fig.suptitle(f'Bayesian Linear Regression Model\n{basis_type}')
        
        model_lbl = f'Prior'
        ax_title = f'Prior Graph'
        model = blr.prior_predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_prior, ax_title
        )

        std = blr.prior_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.praior_sample)

        # plot posterior graphs
        # <your code here>
        fig, ax = plt.subplots()
        fig.suptitle(f'Bayesian Linear Regression Model\n{basis_type}')
        
        model_lbl = f'Postior'
        ax_title = f'Postior Graph\n'
        ax_title += f'Average squared error with BLR and {basis_type} is {avg_sq_err:.2f}'
        model = blr.predict(x)
        ax = plot_results(
            ax, train_hours, train, x, model, model_lbl, 
            test_hours, test, pred_post, ax_title
        )

        std = blr.predict_std(x)
        ax = add_filling_and_samples(ax, x, model, std, blr.posterior_sample)

    if show:
        plt.show()


if __name__ == '__main__':
    main(show=True)
