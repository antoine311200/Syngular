from colorlog import WARNING
import syngular as syn

from syngular.tensor import MatrixProductOperator
from syngular.tensor import MatrixProductState

import numpy as np


def test_mul():

    x = np.arange(2**2)
    X = MatrixProductState(x.reshape((2,2)), bond_shape=(2,)).decompose()

    w = np.arange(2**4).reshape((4,4))
    W = MatrixProductOperator(w.reshape((2,2,2,2)), bond_shape=(2,)).decompose()
    
    print('------')
    print(np.matmul(x, w))
    print(syn.mul(W, X, mode="standard").to_tensor())
    print('------')
    print(np.matmul(x, w))
    print(syn.mul(X, W, mode="standard").to_tensor())
    print('------')
    print(np.matmul(x, x))
    print(syn.mul(X, X, mode="standard"))
    print('------')


test_mul()