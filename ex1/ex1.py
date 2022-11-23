import numpy as np

def multiply_uniforms(a, b, y, d):
    return [min(a, y-d), max(b, y+d)]

def multiply_gaussions(m,ss, h, y, ll):
    mean = ((ll*m) + (ss*h*y)) / (ll + ss*h*h)
    return mean

if __name__ == '__main__':
    a, b = 2, 4
    y, d = 2.5, 1
    new_a, new_b = multiply_uniforms(a, b, y, d)
    print(f'U[{new_a}, {new_b}]')

    m, ss = 1, 0.66
    h, ll = 2.3, 0.14
    y = 2.3
    mean = multiply_gaussions(m, ss, h, y, ll)
    print(f'mean = {mean}')