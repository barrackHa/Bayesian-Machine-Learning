from pathlib import Path
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Gibbs': r'Gibbs, $\alpha={}$, $\beta={}$, $\delta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        # todo <your code here>
        k_x_y = alpha * np.exp(
            -1 * beta * np.power(np.linalg.norm(x-y), 2)
        )
        return k_x_y
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        # todo <your code here>
        operand = np.sum(np.abs(x-y))
        k_x_y = alpha * np.exp(
            -1 * beta * np.power(operand, 2)
        )
        return k_x_y
    return kern


def Gibbs_kernel(alpha: float, beta: float, delta: float, gamma: float) -> Callable:
    """
    An implementation of the Gibbs kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    l = lambda x: gamma + alpha * np.exp(
        -1 * beta * np.power(np.linalg.norm(x-delta), 2)
    )
    def kern(x, y):
        # todo <your code here>
        num = 2 * l(x) * l(y)
        denom = l(x)**2 + l(y)**2
        var = np.sqrt(num / denom)
        exp_num = np.power(np.linalg.norm(x-y), 2)
        exp = np.exp(-1 * (exp_num/denom))
        return var * exp
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        # todo <your code here>
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        if not isinstance(y, np.ndarray):
            y = np.array([y])

        num = 2 * beta * (x @ y + 1)
        denom_x = 1 + (2 * beta * (x @ x + 1))
        denom_y = 1 + (2 * beta * (y @ y + 1))
        denom = np.sqrt(denom_x * denom_y)
        k_x_y = alpha * (2 / np.pi) * np.arcsin(num / denom)
        return k_x_y
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        # todo <your code here>
        self.kernel = kernel
        self.noise = noise
        self.k_star_operator = None
        self.I = None
        self.cov = None
        self.cov_inv = None
        self.alpha = None

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # todo <your code here>
        dim_X = X.shape[0]
        self.I = dim_X
        k = self.kernel
        range(dim_X)
        cov = np.array([ 
            [k(X[i],X[j]) for i in range(dim_X)]
            for j in range(dim_X)
        ]) 
        cov = cov + self.noise * np.eye(dim_X)
        self.cov = cov
        self.cov_inv = np.linalg.pinv(cov) 
        self.alpha = self.cov_inv @ y

        # k_star is an operator that takes a sample and returns the kernel 
        # values between it and all the training samples
        # The len is (n_training_samples)
        self.k_star_operator = lambda x: np.array([k(x, x_) for x_ in X])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        if self.k_star_operator is not None:
            K = np.array([self.k_star_operator(x) for x in X]) 
            f_z = self.alpha @ K.T
            self.fitted_mean = f_z
            return f_z
        
        return np.zeros_like(X)

        
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        # todo <your code here>
        # if model was not fitted, sample from the prior
        if self.k_star_operator is None:
            K = np.array([[self.kernel(x,x_) for x in X] for x_ in X]) 
            K = K + self.noise * np.eye(X.shape[0])
            chol = np.linalg.cholesky(K)
            return chol @ np.random.randn(chol.shape[-1]) 

        # sample from the posterior
        K = np.array([self.k_star_operator(x) for x in X])
        C_z = np.array([[self.kernel(x,x_) for x in X] for x_ in X])
        mean = K @ self.alpha
        cov = C_z - K @ self.cov_inv @ K.T
        sample = np.random.multivariate_normal(mean, cov)

        return sample

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # todo <your code here>
        # if model was not fitted, std of the prior
        if self.k_star_operator is None:
            K = np.array([[self.kernel(x,x_) for x in X] for x_ in X]) 
            K = K + self.noise * np.eye(X.shape[0])
            return np.sqrt(np.diagonal(K) + self.noise)

        # std of the posterior
        # K = self.k_star_operator(X)
        K = np.array([self.k_star_operator(x) for x in X])
        C_z = np.array([[self.kernel(x,x_) for x in X] for x_ in X])
        cov = C_z - K @ self.cov_inv @ K.T
        return np.sqrt(np.diagonal(cov))

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        # todo <your code here>
        N = X.shape[0]
        a = -1/2 * y @ self.alpha
        b = -1/2 * np.linalg.slogdet(self.cov)[1]
        c = -1 * N/2 * np.log(2*np.pi)
        return a + b + c


def main(show=True, save=False):
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([-2.1, -4.3, 0.7, 1.2, 3.9])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 1, 1],        # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 10, 1],        # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 1, 1],                    # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 10, 1],                    # insert your parameters, order: alpha, beta

        # Gibbs kernels
        # a, b, d, g
        ['Gibbs', Gibbs_kernel, 1, 0.5, 0, 0.1],             # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 1, 0.1, 0, 0.1],    # insert your parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 10, 0.1, 0, 0.1],    # insert your parameters, order: alpha, beta, delta, gamma

        # Neurel network kernels
        # a, b
        ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 0.5, 1],                      # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 0.5, 10],                      # insert your parameters, order: alpha, beta
    ]
    noise = 0.05

    # plot all of the chosen parameter settings
    j = 0
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise)

        # plot prior variance and samples from the priors
        plt.figure()
        # todo <your code here>
        s = 2*gp.predict_std(xx)
        m = gp.predict(xx)
        plt.plot(xx, m, lw=1, label='Prior', linestyle='--' , color='black')
        for _ in range(5):
            plt.plot(xx, gp.sample(xx), lw=1)
        f = gp.sample(xx)
        plt.plot(xx, gp.sample(xx), lw=1)
        plt.fill_between(xx, -1*s, s, alpha=.3)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.legend()

        if save:
            f_name = f'kernel_{p[0]}_{(j%3)+1}_prior.png'
            f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
            plt.savefig(f_p)

        
        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        if save:
            f_name = f'kernel_{p[0]}_{(j%3)+1}_post.png'
            f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
            plt.savefig(f_p)
        
        j += 1
    
    # if show:
    #     plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(.1, 15, 101)
    noise = .15

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    
    if save:
        f_name = f'log-evidence.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)

    # if show:
    #   plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise).fit(x, y).predict(xx), lw=2, label=r'min evidence $\beta={}$'.format(min_ev))
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise).fit(x, y).predict(xx), lw=2, label=r'median evidence $\beta={}$'.format(median_ev))
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise).fit(x, y).predict(xx), lw=2, label=r'max evidence $\beta={}$'.format(max_ev))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()

    if save:
        f_name = f'funcs_by_log_evidence.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)
    # if show:
    #   plt.show()

    # ------------------------------------------------------ section 2.2
    # define function and parameters
    f = lambda x: np.sin(x*3)/2 - np.abs(.75*x) + 1
    xx = np.linspace(-3, 3, 100)
    noise = .25
    beta = 2

    # calculate the function values
    np.random.seed(0)
    y = f(xx) + np.sqrt(noise)*np.random.randn(len(xx))

    # ------------------------------ question 5
    # fit a GP model to the data
    gp = GaussianProcess(kernel=RBF_kernel(1, beta=beta), noise=noise).fit(xx, y)

    # calculate posterior mean and confidence interval
    m, s = gp.predict(xx), 2*gp.predict_std(xx)
    print(f'Average squared error of the GP is: {average_error(m, y):.2f}')

    # plot the GP prediction and the data
    plt.figure()
    plt.fill_between(xx, m-s, m+s, alpha=.5)
    plt.plot(xx, m, lw=2)
    plt.scatter(xx, y, 30, 'k', alpha=.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim([-3, 3])
    
    if save:
        f_name = f'last.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)
    if show:
      plt.show()


if __name__ == '__main__':
    main(show=True, save=True)



