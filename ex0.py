import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# %%

def f3(c, h, y):
    if not isinstance(h, np.ndarray):
        h = np.array(h)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    g1 = lambda lamda, theta: -c * np.log(1 / lamda)
    g2 = lambda lamda, theta: (lamda / 2) * np.sum(np.power(h * theta - y, 2))
    func = lambda lamda, theta: g1(lamda, theta) - g2(lamda, theta)
    return func


# b = np.arange(-3, 3.2, 0.02)
# d = np.arange(-3, 3.2, 0.02)

# lamds, thetas = np.meshgrid(b, d)
# h = np.array([1.8, 7.4, 8.2])
# y = np.array([7.7, 9.3, 8.2])
# f = f3(1, h, y)
# nu = f(lamds, thetas)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(lamds, thetas, nu)
# plt.xlabel('b')
# plt.ylabel('d')
# plt.show()

# %%

def max_theta(H: np.ndarray, y: np.ndarrat):
    hh_t = np.transpose(H) @ H
    max_theta = np.linalg.inv(hh_t) @ np.transpose(H) @ y 
    return max_theta

def max_lambda(H: np.ndarray, y: np.ndarray, c: float):
    theta = max_theta(H, y)
    denum =  np.linalg.norm(H @ theta - y)
    denum = denum ** 2
    num = 2 * c
    return num / denum

def max_lamda_theta(H: np.ndarray, y: np.ndarray, c: float):
    return max_theta(H, y), max_lambda(H, y, c)
# %%

def rain_prob(p_r, p_fn, p_fp):
    num = p_r * p_fn
    dnum = 1 - p_r * (1 - p_fn) + p_fp * (1 - p_r)
    return num / dnum


# p_r = 0.03
# p_fn = 0.09
# p_fp = 0.16
# rain_prob(p_r, p_fn, p_fp)

# %%

def uniform_mean(m, d):
    return m ** 2


def uniform_var(m, d):
    return ((d ** 2) / 12)