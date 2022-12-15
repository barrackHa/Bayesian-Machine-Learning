import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    sig_inv = model.prec
    noise = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    H = model.h(X)
    a = 0.5 * np.log(
        np.linalg.det(map_cov) / np.linalg.det(sig)
    )
    mahala_means = (map - mu).T @ sig_inv @ (map - mu)
    mahala_y = 1/noise * np.linalg.norm(y - H @ map)**2
    b = 0.5 * (mahala_means + mahala_y + (X.shape[0] * np.log(noise)))
    c = H.shape[-1] * np.log(2 * np.pi) / 2 

    model_log_evidence = a - b - c
    
    return model_log_evidence


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x**2 - 1
    f2 = lambda x: -x**4 + 3*x**2 + 50*np.sin(x/6)
    f3 = lambda x: .05*x**6 - .75*x**4 + 2.75*x**2
    f4 = lambda x: 5 / (1 + np.exp(-4*x)) - (x - 2 > 0)*x
    f5 = lambda x: np.cos(x*4) + 4*np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            # <your code here>

        # plot evidence versus degree and predicted fit
        # <your code here>

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        # <your code here>

    # plot log-evidence versus amount of sample noise
    # <your code here>


if __name__ == '__main__':
    main()



