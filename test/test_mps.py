from syngular.tensor import MatrixProductState
from syngular.tensor import MatrixProductOperator

import numpy as np
import time

np.set_printoptions(suppress=True)

def test_apply():
    x = np.arange(8).reshape((2,2,2))
    X = MatrixProductState(x, bond_shape=(2,2)).decompose()

    from syngular.quantum import gate

    g = gate.CX.reshape((2,2,2,2))
    # print(g)
    X.apply(g, 1)

    print(X)


def test_augment():
    x = np.arange(8).reshape((2,2,2))
    X = MatrixProductState(x, bond_shape=(2,2)).decompose()
    
    Y = MatrixProductState.zeros((2,2,2),(2,2,))

    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = X + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y + Y
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(Z)
    print(Z.to_tensor())
    print(X.to_tensor())
    print((Z >> 2).to_tensor())


def test_zeros():
    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = MatrixProductState.zeros((2,2,2),(2,2,))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")
    print(Z.to_tensor())
    print(Z.shape)

def test_add():
    x = np.arange(8).reshape((2,2,2))
    y = np.arange(8).reshape((2,2,2))
    X = MatrixProductState(x, bond_shape=(2,2)).decompose()
    Y = MatrixProductState(y, bond_shape=(2,2)).decompose()
    
    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = X + Y
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    z = x + y

    print(z[1,1,1])
    print(Z[1,1,1])

    print(z)
    print(Z.to_tensor())

def test_compress():
    x = np.arange(4**3).reshape((4,4,4))
    X = MatrixProductState(x, bond_shape=(4,4)).decompose()
    
    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = X >> 2
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(X)
    print(Z)

    # print(X.to_tensor())
    # print(Z.to_tensor())
def test_random():

    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = MatrixProductState.random((4,4), (4,))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(Z)
    print(Z.to_tensor())

# test_add()
# test_compress()
# test_random()
# test_zeros()

# test_augment()
test_apply()