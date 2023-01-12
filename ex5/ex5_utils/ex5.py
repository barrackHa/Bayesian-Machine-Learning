import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, GaussianProcess, RBF_kernel, accuracy, Gaussian, plot_ims
from pathlib import Path

def calc_post(mu, sig, sig_0, X):
        N = X.shape[0]
        num_mean = ((1/sig) * np.sum(X)) + ((1/sig_0) * mu)
        denum_mean= (N * (1/sig)) + (1/sig_0)
        mean_mmse = num_mean / denum_mean

        cov = 1 / ((N * (1/sig)) + (1/sig_0)) * np.eye(X[0].shape[0])
        return mean_mmse, cov

def calc_decision_boundary(a, b):
        num_y_0 = np.linalg.norm(a)**2 - np.linalg.norm(b)**2
        denum_y_0 = 2 * (a - b)[1]
        y_0 = num_y_0 / denum_y_0
        l = (a - b)[0] / (a - b)[1]
        return lambda x: y_0 - (l * x) 

def main(save=False):
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(1)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    
    # <your code here>
    
    post_mmse_p, post_cov_p = calc_post(mu_p, sig, sig_0, x_p)
    post_mmse_m, post_cov_m = calc_post(mu_m, sig, sig_0, x_m)

    xx = np.linspace(-1, 1, 100)

    num_of_samples = 10
    samp_p = np.random.multivariate_normal(post_mmse_p, post_cov_p, num_of_samples)
    samp_m = np.random.multivariate_normal(post_mmse_m, post_cov_m, num_of_samples)

    sample_descision_boundaries = [
        calc_decision_boundary(samp_p[i], samp_m[i]) for i in range(num_of_samples)
    ]

    plt.figure()
    plt.scatter(x_p[:,0], x_p[:,1], c='b', label='+')
    plt.scatter(x_m[:,0], x_m[:,1], c='r', label='-')
    
    plt.scatter(post_mmse_p[0], post_mmse_p[1], c='purple', label='POST_Mean_+')
    plt.scatter(post_mmse_m[0], post_mmse_m[1], c='orange', label='POST_Mean_-')

    plt.plot(
        xx, calc_decision_boundary(post_mmse_p, post_mmse_m)(xx), 
        c='black', label='Mean Decision Boundary', linestyle='--', lw=3
    )

    for line in sample_descision_boundaries:
        plt.plot(xx, line(xx), c='gray', linestyle='--', lw=2)

    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')

    if save:
        f_name = f'q1.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)

    plt.show()  

    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    preds = []
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)
        
        # <your code here>
        dogs_model = Gaussian(beta, nu).fit(train[:dogs.shape[0]])
        frogs_model = Gaussian(beta, nu).fit(train[dogs.shape[0]:])

        pred = lambda x: np.where(
            np.sign(
                dogs_model.log_likelihood(x) - frogs_model.log_likelihood(x)
            ) == 0, 1, np.sign(dogs_model.log_likelihood(x) - frogs_model.log_likelihood(x))
        )

        preds.append(pred(train))
        # print train and test accuracies
        train_score[i] = accuracy(pred(train), labels)
        test_score[i] = accuracy(pred(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')

    if save:
        f_name = f'q2.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    kern, sigma = RBF_kernel(.009), .1
    Ns = [250, 500, 1000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ns)), np.zeros(len(Ns))

    gp = None
    for i, N in enumerate(Ns):
        print(f'GP using {N} samples', end='', flush=True)

        # <your code here>
        X = np.concatenate([dogs[:N], frogs[:N]], axis=0)
        y = np.concatenate([np.ones(dogs[:N].shape[0]), -np.ones(frogs[:N].shape[0])])
        gp = GaussianProcess(kern, sigma).fit(X, y)
        
        # print train and test accuracies
        train_score[i] = accuracy(gp.predict(X), y)
        test_score[i] = accuracy(gp.predict(test), labels_t)
        
        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ns, train_score, lw=2, label='train')
    plt.plot(Ns, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')

    if save:
        f_name = f'q3.png'
        f_p = Path(__file__).parents[0]/f'tmp_figs/{f_name}'
        plt.savefig(f_p)
    
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(gp.predict(dogs_t) / gp.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')

if __name__ == '__main__':
    main(save=False)







