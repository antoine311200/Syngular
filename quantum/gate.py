import numpy as np


I = np.array([
    [1., 0.],
    [0., 1.]
])

X = np.array([
    [0., 1.],
    [1., 0.]
])

Y = np.array([
    [0., -1.j],
    [1.j, 0.]
])

Z = np.array([
    [1., 0.],
    [0., -1.]
])

H = 1/np.sqrt(2) * np.array([
    [1., 1.],
    [1., -1.]
])