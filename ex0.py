import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_f3_for_ploting(c, h, y):
    # not tested be cafeful
    if not isinstance(h, np.ndarray):
        h = np.array(h)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    g1 = lambda lamda, theta: -c * np.log(1 / lamda)
    g2 = lambda lamda, theta: (lamda / 2) * np.sum(np.power(h * theta - y, 2))
    func = lambda lamda, theta: g1(lamda, theta) - g2(lamda, theta)
    return func

def max_theta(H, y):
    hh_t = np.transpose(H) @ H
    A = np.linalg.inv(hh_t) @ np.transpose(H)
    max_theta = np.dot(A, y)
    return max_theta

def max_lambda(H, y, c):
    theta = max_theta(H, y)
    denum =  np.linalg.norm(H @ theta - y)
    denum = denum ** 2
    num = 2 * c
    return num / denum

def max_lamda_theta(H, y, c: float):
    return max_theta(H, y), max_lambda(H, y, c)

def rain_prob(p_r, p_fn, p_fp):
    num = p_r * p_fn
    dnum = 1 - p_r * (1 - p_fn) + p_fp * (1 - p_r)
    return num / dnum

def uniform_mean(m, d):
    return m

def uniform_var(m, d):
    return ((d ** 2) / 12)

if __name__ == '__main__':
   
    # Assign values as in the question in the quiz and print results
    # Q2-3
    H = np.array([[8.4], [5.7], [7.2]])
    y = np.array([8.1, 5.7, 7.7])
    c = 1

    print(f'Quiz Q2 - max theta: {max_theta(H,y)}')
    print(f'Quiz Q3: max lambda: {max_lambda(H,y,c)}')

    # Q4
    p_fn = 0.04
    p_fp = 0.28
    p_r = 0.04
    print(f'Quiz Q4: rain for rain: {rain_prob(p_r, p_fn, p_fp)}')

    # Q5-6
    m = 3.6
    d = 5.7

    print(f'Quiz Q5: mean: {uniform_mean(m, d)}')
    print(f'Quiz Q6: var: {uniform_var(m, d)}')

    # # Plot f3
    # b = np.arange(-3, 3.2, 0.02)
    # d = np.arange(-3, 3.2, 0.02)

    # lamds, thetas = np.meshgrid(b, d)
    # h = np.array([1.8, 7.4, 8.2])
    # y = np.array([7.7, 9.3, 8.2])
    # f3= get_f3_for_ploting(1, h, y)
    # nu = f3(lamds, thetas)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(lamds, thetas, nu)
    # plt.xlabel('b')
    # plt.ylabel('d')
    # plt.show()
