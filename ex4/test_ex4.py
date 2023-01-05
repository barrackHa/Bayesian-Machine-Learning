import pytest
import numpy as np
from matplotlib import pyplot as plt
from ex4 import RBF_kernel, Laplacian_kernel, Gibbs_kernel, NN_kernel
from pathlib import Path
# import sys
# import os


p = Path(__file__)
# sys.path.append(str(p.parents[2]))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ex4')))

def test_RBF_kernel():
    x = np.linspace(-3, 3, 100)
    y = np.zeros_like(x)

    k = RBF_kernel(alpha=1, beta=1)
    k_x_y = np.array([k(x_i, y_i) for x_i, y_i in zip(x, y)])
    test = np.exp(-1 * x**2)
    assert np.array_equal(k_x_y, test)

def test_Laplacian_kernel():
    x = np.linspace(-3, 3, 100)
    y = np.zeros_like(x)

    k = Laplacian_kernel(alpha=1, beta=1)
    k_x_y = k(1,0)
    assert np.array_equal(k_x_y, np.exp(-1))

def test_Gibbs_kernel():
    pass
    # assert False

def test_NN_kernel():
    x = np.linspace(-3, 3, 100)
    y = np.zeros_like(x)
    k = NN_kernel(alpha=1, beta=1)
    k_x_y = k(1,0)
    print(k_x_y)
    assert False
    
if __name__ == '__main__':
    pytest.main([str(p)])

