from pathlib import Path
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
    
    ## Tow way to calculate the log determinant of a matrix:
    ## The secode way is theoreticly more stable but I see no difference in the results

    # a = 0.5 * np.log(
    #     np.linalg.det(map_cov) / np.linalg.det(sig)
    # )
    a = 0.5 *(np.linalg.slogdet(map_cov)[1] - np.linalg.slogdet(sig)[1])
    mahala_means = (map - mu).T @ sig_inv @ (map - mu)
    mahala_y = 1/noise * np.linalg.norm(y - H @ map)**2
    b = 0.5 * (mahala_means + mahala_y + (X.shape[0] * np.log(noise)))
    c = H.shape[-1] * np.log(2 * np.pi) / 2 

    model_log_evidence = a - b - c
    
    return model_log_evidence


def main(save=False, show=True):
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
    tot_evidence = []

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        evidence_dict = {}
        evidence_lst = []

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            # <your code here>
            evidence_dict[d] = ev
            evidence_lst.append(ev)
        
        # For each function in functions we get a dictionary 
        # of evidence values per degree
        # looks like - 
        # [function1: {degree1: evidence1, degree2: evidence2, ...},...]
        tot_evidence.append(evidence_dict)


        # plot evidence versus degree and predicted fit
        # <your code here>
        plt.figure()
        plt.plot(degrees, evidence_lst, 'k', lw=2)
        argMax = degrees[np.argmax(evidence_lst)]
        argMin = degrees[np.argmin(evidence_lst)]
        plt.xlabel('$x$')
        plt.ylabel(r'log-evidence')
        plt.suptitle(f'log-evidence versus degree for function {i+1}')
        plt.title(f'Best Fit: {argMax}, Worst Fit: {argMin}')
        plt.grid()

        if save:
            p = Path(__file__).parents[1]/f'tmp_figs/q3_f_{i}_log_evidence.png'
            plt.savefig(p)

        best = polynomial_basis_functions(argMax)
        worst = polynomial_basis_functions(argMin)
        best_mean, best_cov = np.zeros(argMax + 1), np.eye(argMax + 1) * alpha
        worst_mean, worst_cov = np.zeros(argMin + 1), np.eye(argMin + 1) * alpha

        best_blr = BayesianLinearRegression(best_mean, best_cov, noise_var, best)
        worst_blr = BayesianLinearRegression(worst_mean, worst_cov, noise_var, worst)
        
        best_pred = best_blr.fit_predict(x, y)
        best_std = best_blr.predict_std(x)
        worst_pred = worst_blr.fit_predict(x, y)
        worst_std = worst_blr.predict_std(x)

        plt.figure()
        plt.plot(x, y, 'o',  lw=0.1, alpha=0.3, label='data', color='k')
        plt.plot(x, best_pred, lw=2, label=f'best fit d={argMax}')
        plt.fill_between(x, best_pred-best_std, best_pred+best_std, alpha=.5)
        plt.plot(x, worst_pred, lw=2, label=f'worst fit d={argMin}')
        plt.fill_between(x, worst_pred-worst_std, worst_pred+worst_std, alpha=.5)
        plt.title(f'Function {i+1} - Best And Worst Evidence Compared To Data')
        plt.legend()

        if save:
            p = Path(__file__).parents[1]/f'tmp_figs/q4_f_{i}_best_and_worst.png'
            plt.savefig(p)


    if show:     
        plt.show()
    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load(Path(__file__).parent / 'nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    # temps_log_evidence = []
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        # <your code here>
        evs[i] = (ev)

    # plot log-evidence versus amount of sample noise
    # <your code here>
    plt.figure()
    plt.plot(noise_vars, evs)
    noise_argMax = noise_vars[np.argmax(evs)]
    plt.title(f'Sample Noise With Highest Evidence: {noise_argMax:.4f}')
    plt.xlabel('Sample Noise')
    plt.ylabel('Log-Evidence')
    plt.suptitle('Log-Evidence Score As A Function Of Sample Noise')
    plt.grid()

    if save:
        p = Path(__file__).parents[1]/f'tmp_figs/q6_log_evidence_per_noise.png'
        plt.savefig(p)
    
    if show:
        plt.show()

if __name__ == '__main__':
    main(show=True, save=False)



